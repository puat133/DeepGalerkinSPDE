import argparse
import inspect
import os
import sys
from datetime import datetime
import jax
import jax.numpy as np
import jax.experimental.optimizers as jopt
import jax.random as jrandom
from jax import jit, value_and_grad, partial
from tqdm import trange

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from utils import add_boolean_argument, set_plot
from utils.hdf_io import save_to_hdf
from DeepSPDE import Fourier, RandomGenerator, LMatrixGenerator
from Optimizers import hessian_free, adahessian, value_grad_and_hessian
from post_processing import process_nwd_results
from utils.loss_function_and_support import from_complex_w_halfs_to_ravelled_reals, from_two_reals_ravelled_to_complex, \
    non_analytical_mean_and_std as non_analytical_mean_and_std, \
    loss, output_fun
from Measurements import DenoisingOneD, OneDSampleSignalShape
from Optimizers import OptimizerType

os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/local/cuda"


def __run_simulation(fourier_basis_number: int,
                     fourier_target_number: int,
                     measurement_samples_number: int,
                     n_samples: int,
                     n_prior_layers: int,
                     n_steps: int,
                     cg_max_iter: int,
                     random_seed: int,
                     random_loop_length: int,
                     measurement_variance: float,
                     initial_lm_damping: float,
                     step_scale: float,
                     kappa0: float,
                     required_max_loss_val: float,
                     momentum: float,
                     use_randomized_initialization: bool,
                     use_gnm: bool,
                     use_soft_max: bool,
                     use_cuda: bool,
                     signal_shape: OneDSampleSignalShape,
                     optimizer: OptimizerType,
                     ):
    if not use_cuda:
        print('Not using CUDA ...')
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    print('Initialization ...')
    f = Fourier(fourier_basis_number, fourier_target_number, 1)
    dod = DenoisingOneD(num_points=2 * f.target_basis_number - 1, basis_number=f.basis_number,
                        std_dev=np.sqrt(measurement_variance),
                        number_of_sample=measurement_samples_number,
                        signal_shape=signal_shape)

    if use_soft_max:
        kappa_fun = lambda x: np.log(np.exp(x) + 1)
        l_matrix = LMatrixGenerator(f, kappa0=kappa0, kappa_fun=kappa_fun)
    else:
        l_matrix = LMatrixGenerator(f, kappa0=kappa0)

    identity = np.eye(2 * f.basis_number - 1)
    identity_m = np.eye(dod.num_points)
    meas_samples = dod.measure()
    normalized_meas_samples = np.asarray(meas_samples, dtype=np.float32) / dod.std_dev
    h_matrix = np.asarray(dod.measurement_matrix, dtype=np.complex64) / dod.std_dev
    h_conj_t = h_matrix.conj().T

    @partial(jit, static_argnums=(1,))
    def wrapped_loss(whalfs_real, whalfs_shape, sigmas_in, y):
        w_halfs_in = from_two_reals_ravelled_to_complex(whalfs_real, whalfs_shape)
        return loss(w_halfs_in, sigmas_in, y, h_conj_t, identity_m, l_matrix)

    @partial(jit, static_argnums=(2,))
    def wrapped_output_fun(sigmas_in, whalfs_real, whalfs_shape, y):
        w_halfs_in = from_two_reals_ravelled_to_complex(whalfs_real, whalfs_shape)
        return output_fun(w_halfs_in, sigmas_in, y, h_conj_t, identity_m, l_matrix)

    rg = RandomGenerator(f.basis_number, f.dimension, prngkey_num=random_seed)
    if optimizer in [OptimizerType.HESSIAN_FREE]:
        picked_whalfs = rg.construct_w_half(n_prior_layers)
    picked_whalfs = np.zeros((n_prior_layers, f.basis_number), dtype=np.complex64)
    picked_log_sigmas = np.zeros(n_prior_layers + 1)
    sigmas = np.ones(n_prior_layers + 1)

    if use_randomized_initialization:
        print('Use a randomized initial condition ...')
        picked_val = np.inf
        loss_val = np.nan
        picked_log_sigmas = 0
        picked_whalfs = 0

        picked_log_sigmas = np.log(sigmas)
        prngkey = jrandom.PRNGKey(random_seed)
        w_halfs_many = rg.construct_w_half(n_prior_layers * random_loop_length)
        w_halfs_many = w_halfs_many.reshape((random_loop_length, n_prior_layers, l_matrix.fourier.basis_number))
        prngkey, subkeys = jrandom.split(prngkey)
        log_sigmas = jrandom.normal(subkeys, shape=(random_loop_length, n_prior_layers + 1))
        for i in trange(random_loop_length):
            sigmas = np.exp(picked_log_sigmas + log_sigmas[i])
            w_halfs = picked_whalfs + w_halfs_many[i]
            loss_val = loss(w_halfs, sigmas,
                            normalized_meas_samples[0],
                            h_conj_t, identity_m, l_matrix).block_until_ready()  # test with first measurement

            if loss_val < 0.:  # below zero seems an invalid value
                continue

            if loss_val < picked_val:
                picked_val = loss_val
                picked_whalfs = w_halfs
                w_halfs_real, w_halfs_shape = from_complex_w_halfs_to_ravelled_reals(w_halfs)
                state_init = np.concatenate((sigmas, w_halfs_real))
                print('update picked_val to {}'.format(picked_val))
            if picked_val < required_max_loss_val:
                break

    if optimizer in [OptimizerType.HESSIAN_FREE, OptimizerType.ADA_HESSIAN]:
        if optimizer == OptimizerType.HESSIAN_FREE:
            picked_whalfs = rg.construct_w_half(n=n_prior_layers)

        w_halfs_real, w_halfs_shape = from_complex_w_halfs_to_ravelled_reals(picked_whalfs)
        state_init = w_halfs_real.copy()
        if optimizer == OptimizerType.ADA_HESSIAN:
            opt = adahessian(step_size=step_scale)
            rng = jax.random.PRNGKey(1)
            opt_state = opt.init_fn(state_init)

    else:
        state_init = (picked_whalfs.copy(),)
        if optimizer == OptimizerType.ADAM:
            opt = jopt.adam(step_size=step_scale, b1=0.01, b2=0.01)
            opt_state = opt.init_fn(state_init)
        elif optimizer == OptimizerType.RMSPROP:
            opt = jopt.rmsprop(step_size=step_scale)
            opt_state = opt.init_fn(state_init)
        elif optimizer == OptimizerType.RMSPROP_MOMENTUM:
            opt = jopt.rmsprop_momentum(step_size=step_scale, momentum=momentum)
            opt_state = opt.init_fn(state_init)

    simulation_time_string = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    simulation_id = simulation_time_string + '_' + dod.signal_shape.name + '_n_prior_layers' + str(n_prior_layers)

    loss_val_list = []
    nu_val_list = []
    v_mean_list = []
    v_std_list = []
    kappa_list = []
    end_state_list = []

    print('Executing the main loop ...')
    print('Computing {} optimization ...'.format(optimizer.name.lower()))

    # loop to simulate using all measurement samples:
    last_loss_value = np.inf
    with trange(normalized_meas_samples.shape[0]) as tr:
        tr.set_description(desc="last loss = {}".format(last_loss_value))
        # for i in trange(normalized_meas_samples.shape[0], ):
        for i in tr:
            y = normalized_meas_samples[i]
            if optimizer in [OptimizerType.HESSIAN_FREE, OptimizerType.ADA_HESSIAN]:
                def re_wrapped_loss(state_real, whalfs_shape=(2, n_prior_layers, f.basis_number),
                                    num_sigmas=n_prior_layers + 1):
                    sigmas_in = np.ones(num_sigmas)
                    w_real = state_real
                    return wrapped_loss(w_real, whalfs_shape, sigmas_in, y)

                def re_wrapped_output_fun(state_real, whalfs_shape=(2, n_prior_layers, f.basis_number),
                                          num_sigmas=n_prior_layers + 1):
                    sigmas_in = np.ones(num_sigmas)
                    w_real = state_real
                    return wrapped_output_fun(sigmas_in, w_real, whalfs_shape, y)

                if use_gnm:
                    fun = re_wrapped_output_fun
                else:
                    fun = re_wrapped_loss

                if optimizer == OptimizerType.HESSIAN_FREE:
                    state_hist, nu_hist, value_hist = hessian_free(fun, state_init, n_steps=n_steps,
                                                                   use_gnm=use_gnm,
                                                                   cg_max_iter=cg_max_iter,
                                                                   learning_rate=step_scale)
                    w_halfs_real = state_hist[-1, :]
                    end_state_list.append(w_halfs_real)

                # ADA_HESSIAN
                else:
                    def optimizer_step(step, opt_state, rng):
                        val, gradient, hessian = value_grad_and_hessian(re_wrapped_loss, (opt.params_fn(opt_state),),
                                                                        rng,
                                                                        argnum=0)
                        if np.isnan(val):
                            raise StopIteration
                        if np.any(np.isnan(gradient)) or np.any(np.isnan(hessian)):
                            raise StopIteration
                        opt_state = opt.update_fn(step, gradient, hessian, opt_state)
                        return val, opt_state

                    value_hist = []
                    opt_state = opt.init_fn(state_init)
                    try:
                        with trange(n_steps) as tr_:
                            for step in tr_:
                                rng, rng_param = jax.random.split(rng)
                                val, opt_state = optimizer_step(step, opt_state, rng_param)
                                value_hist.append(val)
                                tr_.set_description('Current loss value = {:.4f}'.format(val))
                    except StopIteration:
                        print('Nan value found in the gradient. Stopping the optimization loop')
                    finally:
                        w_halfs = opt.params_fn(opt_state)
                        end_state_list.append(w_halfs)

                    if not value_hist:
                        value_hist = np.array([np.inf])
                    else:
                        value_hist = np.stack(value_hist)

                    w_halfs_real = opt.params_fn(opt_state)

                w_halfs = from_two_reals_ravelled_to_complex(w_halfs_real, (2, n_prior_layers, f.basis_number))

            elif optimizer in [OptimizerType.ADAM, OptimizerType.RMSPROP, OptimizerType.RMSPROP_MOMENTUM]:
                def re_wrapped_loss(w_halfs_in):
                    val = loss(w_halfs_in, sigmas, y, h_conj_t, identity_m, l_matrix)
                    return val

                loss_grad_and_val = jit(value_and_grad(re_wrapped_loss, argnums=(0,)))

                def optimizer_step(step, state):
                    val, grads = loss_grad_and_val(*opt.params_fn(state))
                    if np.isnan(val):
                        raise StopIteration
                    if np.any(np.isnan(grads[0])):
                        raise StopIteration
                    state = opt.update_fn(step, grads, state)
                    return val, state

                value_hist = []
                state_init = (picked_whalfs.copy(),)
                opt_state = opt.init_fn(state_init)
                (w_halfs,) = opt.params_fn(opt_state)
                try:
                    with trange(n_steps) as tr_:
                        for step in tr_:
                            val, opt_state = optimizer_step(step, opt_state)
                            value_hist.append(val)
                            tr_.set_description('Current loss value = {:.4f}'.format(val))
                except StopIteration:
                    print('Nan value found in the gradient. Stopping the optimization loop')
                finally:
                    (w_halfs,) = opt.params_fn(opt_state)
                    end_state_list.append(w_halfs)

                if not value_hist:
                    value_hist = np.array([np.inf])
                else:
                    value_hist = np.stack(value_hist)

            # Non analytical
            v_mean, v_std = non_analytical_mean_and_std(w_halfs, sigmas, h_matrix, y, n_samples, l_matrix, f, rg,
                                                        use_lancos=False)

            kappa = l_matrix.kappa_fun(f.irfft(l_matrix.u_half_min_1))
            v_mean_list.append(v_mean)
            v_std_list.append(v_std)
            kappa_list.append(kappa)

            if value_hist.shape[0] < n_steps:
                temp = np.concatenate((value_hist, np.nan * np.ones(n_steps - value_hist.shape[0])))
                value_hist = temp
            loss_val_list.append(value_hist)
            last_loss_value = value_hist[-1]
            tr.set_description(desc="last loss = {}".format(last_loss_value))
            if optimizer == OptimizerType.HESSIAN_FREE:
                nu_val_list.append(nu_hist)

    print('Stacking result ...')
    if v_mean_list:
        v_mean_all = np.stack(v_mean_list)
    else:
        v_mean_all = np.nan

    if v_std_list:
        v_std_all = np.stack(v_std_list)
    else:
        v_std_all = np.nan

    if kappa_list:
        kappa_all = np.stack(kappa_list)
    else:
        kappa_all = np.nan

    if loss_val_list:
        loss_val_all = np.stack(loss_val_list)
    else:
        loss_val_all = np.nan

    if end_state_list:
        end_state_all = np.stack(end_state_list)
    else:
        end_state_all = np.nan

    if optimizer == OptimizerType.HESSIAN_FREE:
        nu_all = np.stack(nu_val_list)

    del v_mean_list
    del v_std_list
    del kappa_list
    del loss_val_list
    del nu_val_list
    del end_state_list

    print('Saving local variables ...')
    save_to_hdf('./.simulation_results/{}.hdf'.format(simulation_id), locals())

    print('post processing ...')
    process_nwd_results(simulation_id + '.hdf')

    print('Finished ...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--basis', default=64, type=int, help='fourier_basis_number')
    parser.add_argument('--target-basis', default=128, type=int, help='fourier_target_number')
    parser.add_argument('--meas-samples', default=2, type=int, help='measurement_samples_number')
    parser.add_argument('--samples', default=1000, type=int, help='n_samples')
    parser.add_argument('--prior', default=1, type=int, help='n_prior_layers')
    parser.add_argument('--steps', default=100, type=int, help='n_steps')
    parser.add_argument('--cg', default=100, type=int, help='cg_max_iter')
    parser.add_argument('--seed', default=0, type=int, help='random_seed')
    parser.add_argument('--rloop', default=10000, type=int, help='random_loop_length')

    parser.add_argument('--var', default=0.01, type=float, help='measurement_variance')
    parser.add_argument('--learn', default=1e-2, type=float, help='step_scale')
    parser.add_argument('--kappa0', default=1., type=float, help='kappa0')
    parser.add_argument('--momentum', default=0.1, type=float, help='momentum')
    parser.add_argument('--damping', default=1., type=float, help='initial_lm_damping')
    parser.add_argument('--maxloss', default=500., type=float, help='required_max_loss_val')

    add_boolean_argument(parser, 'randomized', default=False, messages='whether to use_randomized_initialization')
    add_boolean_argument(parser, 'gnm', default=False, messages='whether to use_gnm')
    add_boolean_argument(parser, 'softmax', default=False, messages='whether to use_soft_max')
    add_boolean_argument(parser, 'cuda', default=True, messages='whether to use_cuda')

    parser.add_argument('--shape', default='Smooth', type=str, help='signal_shape')
    parser.add_argument('--optimizer', default='hf', type=str, help='optimizer')

    args = parser.parse_args()
    shape = OneDSampleSignalShape.SMOOTH_DISCONTINUOUS
    optimizer = OptimizerType.HESSIAN_FREE

    shape_str = args.shape.lower()
    if shape_str == 'rect':
        shape = OneDSampleSignalShape.RECT
    elif shape_str == 'rect2':
        shape = OneDSampleSignalShape.RECT2
    elif shape_str == 'box':
        shape = OneDSampleSignalShape.BOX

    optimizer_str = args.optimizer.lower()
    if optimizer_str == 'hf':
        optimizer = OptimizerType.HESSIAN_FREE
    elif optimizer_str == "adam":
        optimizer = OptimizerType.ADAM
    elif optimizer_str == "rmsprop":
        optimizer = OptimizerType.RMSPROP
    elif optimizer_str == "adahessian":
        optimizer = OptimizerType.ADA_HESSIAN
    elif optimizer_str == "rmspropmomentum":
        optimizer = OptimizerType.RMSPROP_MOMENTUM

    __run_simulation(fourier_basis_number=args.basis,
                     fourier_target_number=args.target_basis,
                     measurement_samples_number=args.meas_samples,
                     n_samples=args.samples,
                     n_prior_layers=args.prior,
                     n_steps=args.steps,
                     cg_max_iter=args.cg,
                     random_seed=args.seed,
                     random_loop_length=args.rloop,
                     measurement_variance=args.var,
                     initial_lm_damping=args.damping,
                     step_scale=args.learn,
                     kappa0=args.kappa0,
                     momentum=args.momentum,
                     required_max_loss_val=args.maxloss,
                     use_randomized_initialization=args.randomized,
                     use_gnm=args.gnm,
                     use_soft_max=args.softmax,
                     use_cuda=args.cuda,
                     signal_shape=shape,
                     optimizer=optimizer
                     )
