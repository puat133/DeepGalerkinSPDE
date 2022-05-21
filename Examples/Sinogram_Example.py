import argparse
import inspect
import os
import pathlib
import sys
from datetime import datetime

import jax
import jax.experimental.optimizers as jopt
import jax.numpy as np
import jax.random as jrandom
import numpy as onp
from jax import jit, partial, value_and_grad
from tqdm import trange

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from post_processing import process_tomography_results
from utils import symmetrize, metric
from utils import add_boolean_argument, plot_image_and_save
from utils.hdf_io import save_to_hdf
from DeepSPDE import Fourier, RandomGenerator, LMatrixGenerator
from Optimizers import adahessian, value_grad_and_hessian
from utils.loss_function_and_support import from_complex_w_halfs_to_ravelled_reals, from_two_reals_ravelled_to_complex, \
    non_analytical_mean_and_std, loss, output_fun
from Measurements import Sinogram
from Phantom import contrast_shepp_logan_ellipses, multi_ellipses_fun
from Optimizers import OptimizerType

os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/local/cuda"


def __run_simulation(fourier_basis_number: int,
                     fourier_target_number: int,
                     restoration_upscale: int,
                     n_theta: int,
                     n_samples: int,
                     n_prior_layers: int,
                     capture_interval: int,
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
                     b1_momentum: float,
                     b2_momentum: float,
                     tikhonov_relative_const: float,
                     use_randomized_initialization: bool,
                     learn_sigmas: bool,
                     use_gnm: bool,
                     use_soft_max: bool,
                     use_cuda: bool,
                     optimizer: OptimizerType,
                     compute_tikhonov_regularization: bool
                     ):
    if not use_cuda:
        print('Not using CUDA ...')
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    f = Fourier(fourier_basis_number, fourier_target_number, 2)
    f_upscaled = Fourier(fourier_basis_number, restoration_upscale * fourier_target_number, 2)

    # domain mask
    mask_ = onp.nan * onp.ones(f_upscaled.expected_shape)
    r_ = mask_.shape[0] / 2
    t_ = onp.linspace(-r_, r_, mask_.shape[0], endpoint=True)
    x_, y_ = onp.meshgrid(t_, t_)
    mask_[x_ ** 2 + y_ ** 2 <= r_ ** 2] = 1
    mask = np.array(mask_)

    if use_soft_max:
        kappa_fun = lambda x: np.log(np.exp(x) + 1)
        l_matrix = LMatrixGenerator(f, kappa0=kappa0, kappa_fun=kappa_fun)
    else:
        l_matrix = LMatrixGenerator(f, kappa0=kappa0)

    n_sigmas = n_prior_layers + 1
    n_p_prime = 2 * f.target_basis_number - 1
    scale = n_p_prime
    std_dev = np.sqrt(measurement_variance)
    thetas = onp.linspace(0, 1, n_theta, endpoint=False) * onp.pi
    sino = Sinogram(contrast_shepp_logan_ellipses, f, thetas, n_p_prime, std_dev=std_dev)
    identity = np.eye(2 * f.basis_number - 1)
    meas_samples = sino.measure().ravel(sino.ravel_order)
    identity_m = np.eye(meas_samples.shape[0])
    normalized_meas_samples = np.asarray(meas_samples, dtype=np.float32) / sino.std_dev
    sino.measurement_matrix
    h_matrix = np.asarray(sino.measurement_matrix, dtype=np.complex64) / sino.std_dev
    h_conj_t = h_matrix.conj().T

    t_points = onp.linspace(-0.5, 0.5, 2 * f_upscaled.target_basis_number - 1)  # this is the default settings
    x_points, y_points = onp.meshgrid(t_points, t_points)
    ground_truth_phantom = multi_ellipses_fun(x_points, y_points, contrast_shepp_logan_ellipses)

    @partial(jit, static_argnums=(3,))
    def wrapped_loss(sigmas_in, whalfs_real, y, whalfs_shape):
        w_halfs_in = from_two_reals_ravelled_to_complex(whalfs_real, whalfs_shape)
        return loss(w_halfs_in, sigmas_in, y, h_conj_t, identity_m, l_matrix)

    @partial(jit, static_argnums=(2,))
    def wrapped_output_fun(sigmas_in, whalfs_real, y, whalfs_shape):
        w_halfs_in = from_two_reals_ravelled_to_complex(whalfs_real, whalfs_shape)
        return output_fun(w_halfs_in, sigmas_in, y, h_conj_t, identity_m, l_matrix)

    rg = RandomGenerator(f.essential_basis_number, f.dimension, prngkey_num=1)
    picked_whalfs = np.zeros((n_prior_layers, f.basis_number), dtype=np.complex64)
    if optimizer in [OptimizerType.ADA_HESSIAN, OptimizerType.RMSPROP_MOMENTUM, OptimizerType.ADAM,
                     OptimizerType.RMSPROP]:
        picked_whalfs = rg.construct_w_half(n_prior_layers)
    picked_log_sigmas = np.zeros(n_prior_layers + 1)
    sigmas = np.ones(n_prior_layers + 1)

    if use_randomized_initialization:
        print('Use a randomized initial condition ...')
        picked_val = np.inf
        loss_val = np.nan
        prngkey = jrandom.PRNGKey(random_seed)
        w_halfs_many = rg.construct_w_half(n_prior_layers * random_loop_length)
        w_halfs_many = w_halfs_many.reshape((random_loop_length, n_prior_layers, l_matrix.fourier.basis_number))
        prngkey, subkeys = jrandom.split(prngkey)
        log_sigmas = jrandom.normal(subkeys, shape=(random_loop_length, n_prior_layers + 1))
        for i in trange(random_loop_length):
            sigmas = np.exp(picked_log_sigmas + log_sigmas[i])
            w_halfs = picked_whalfs + w_halfs_many[i]
            loss_val = loss(w_halfs, sigmas,
                            normalized_meas_samples,
                            h_conj_t, identity_m, l_matrix).block_until_ready()  # test with first measurement

            if loss_val < 0.:  # below zero seems an invalid value
                continue

            if loss_val < picked_val:
                picked_val = loss_val
                picked_whalfs = w_halfs
                picked_log_sigmas = np.log(sigmas)
                print('update picked_val to {}'.format(picked_val))
            if picked_val < required_max_loss_val:
                break
        # Directly delete after that so it wont be saved
        del w_halfs_many
        del log_sigmas

    if optimizer in [OptimizerType.ADA_HESSIAN]:
        sigmas = np.exp(picked_log_sigmas)
        w_halfs_real, w_halfs_shape = from_complex_w_halfs_to_ravelled_reals(picked_whalfs)
        if learn_sigmas:
            state_init = np.concatenate((sigmas, w_halfs_real))
        else:
            state_init = w_halfs_real

        if optimizer == OptimizerType.ADA_HESSIAN:
            opt = adahessian(step_size=step_scale, b1=b1_momentum, b2=b2_momentum)
            rng = jax.random.PRNGKey(1)
            opt_state = opt.init_fn(state_init)

    else:
        if learn_sigmas:
            state_init = (sigmas, picked_whalfs.copy())
        else:
            state_init = (picked_whalfs.copy(),)

        if optimizer == OptimizerType.ADAM:
            opt = jopt.adam(step_size=step_scale, b1=b1_momentum, b2=b2_momentum)
            opt_state = opt.init_fn(state_init)
        elif optimizer == OptimizerType.RMSPROP:
            opt = jopt.rmsprop(step_size=step_scale)
            opt_state = opt.init_fn(state_init)
        elif optimizer == OptimizerType.RMSPROP_MOMENTUM:
            opt = jopt.rmsprop_momentum(step_size=step_scale, momentum=momentum)
            opt_state = opt.init_fn(state_init)

    simulation_time_string = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    simulation_id = simulation_time_string + '_sino_n_prior_layers_' + str(n_prior_layers)

    loss_val_list = []
    nu_val_list = []
    v_mean_list = []
    v_std_list = []
    kappa_list = []
    end_state_list = []

    y = normalized_meas_samples
    if optimizer in [OptimizerType.ADA_HESSIAN]:
        def re_wrapped_loss(state_real, whalfs_shape=(2, n_prior_layers, f.basis_number),
                            num_sigmas=n_sigmas):
            sigmas_in = np.ones(num_sigmas)
            w_real = state_real
            return wrapped_loss(sigmas_in, w_real, y, whalfs_shape)

        def re_wrapped_loss_b(state_real, whalfs_shape=(2, n_prior_layers, f.basis_number),
                              num_sigmas=n_sigmas):
            sigmas_in = state_real[:num_sigmas]
            w_real = state_real[num_sigmas:]

            return wrapped_loss(sigmas_in, w_real, y, whalfs_shape)

        def optimizer_step(step_, opt_state_, rng_):
            if learn_sigmas:
                fun_ = re_wrapped_loss_b
            else:
                fun_ = re_wrapped_loss

            val, gradient, hessian = value_grad_and_hessian(fun_, (opt.params_fn(opt_state_),),
                                                            rng_, argnum=0)
            if np.isnan(val):
                raise StopIteration
            if np.any(np.isnan(gradient)) or np.any(np.isnan(hessian)):
                raise StopIteration
            opt_state_ = opt.update_fn(step_, gradient, hessian, opt_state_)
            return val, opt_state_

        value_hist = []
        opt_state = opt.init_fn(state_init)

        try:
            with trange(n_steps) as tr:
                for step in tr:
                    rng, rng_param = jax.random.split(rng)
                    val, opt_state = optimizer_step(step, opt_state, rng_param)
                    value_hist.append(val)
                    tr.set_description('Current loss value = {:.4f}'.format(val))
                    if step % capture_interval == 0:
                        w_halfs, sigmas = __retrieve_w_halfs_and_sigmas(opt,
                                                                        optimizer,
                                                                        opt_state,
                                                                        learn_sigmas,
                                                                        n_sigmas,
                                                                        n_prior_layers, f)

                        v_mean, v_std, kappa = __retrieve_mean_std_kappa(w_halfs, sigmas, h_matrix, y,
                                                                         1,
                                                                         l_matrix, f_upscaled, rg, sino)
                        __temp_plot(simulation_id, step, v_mean, v_std, kappa, extension='png'
                                    , mask=mask)

        except StopIteration:
            print('Nan value found in the gradient. Stopping the optimization loop')
        finally:
            end_state = opt.params_fn(opt_state)
            if learn_sigmas:
                sigmas = end_state[:n_sigmas]
                w_halfs_real = end_state[n_sigmas:]
            else:
                w_halfs_real = end_state
            end_state_list.append(end_state)
            w_halfs = from_two_reals_ravelled_to_complex(w_halfs_real, (2, n_prior_layers, f.basis_number))

        if not value_hist:
            value_hist = np.array([np.inf])
        else:
            value_hist = np.stack(value_hist)

    elif optimizer in [OptimizerType.ADAM, OptimizerType.RMSPROP, OptimizerType.RMSPROP_MOMENTUM]:
        def re_wrapped_loss(w_halfs_in):
            val = loss(w_halfs_in, sigmas, y, h_conj_t, identity_m, l_matrix)
            return val

        def re_wrapped_loss_b(w_halfs_in, sigmas_in):
            val = loss(w_halfs_in, sigmas_in, y, h_conj_t, identity_m, l_matrix)
            return val

        if learn_sigmas:
            state_init = (picked_whalfs.copy(), sigmas.copy())
            opt_state = opt.init_fn(state_init)
            (w_halfs, sigmas) = opt.params_fn(opt_state)
            fun = re_wrapped_loss_b
            loss_grad_and_val = jit(value_and_grad(fun, argnums=(0, 1)))
        else:
            state_init = (picked_whalfs.copy(),)
            opt_state = opt.init_fn(state_init)
            (w_halfs,) = opt.params_fn(opt_state)
            fun = re_wrapped_loss
            loss_grad_and_val = jit(value_and_grad(fun, argnums=(0,)))

        def optimizer_step(step, state):
            val, grads = loss_grad_and_val(*opt.params_fn(state))
            if np.isnan(val):
                raise StopIteration
            if np.any(np.isnan(grads[0])):
                raise StopIteration
            state = opt.update_fn(step, grads, state)
            return val, state

        value_hist = []
        try:
            with trange(n_steps) as tr:
                for step in tr:
                    val, opt_state = optimizer_step(step, opt_state)
                    value_hist.append(val)
                    tr.set_description('Current loss value = {:.4f}'.format(val))

                    if step % capture_interval == 0:
                        w_halfs, sigmas = __retrieve_w_halfs_and_sigmas(opt,
                                                                        optimizer,
                                                                        opt_state,
                                                                        learn_sigmas,
                                                                        n_sigmas,
                                                                        n_prior_layers, f)

                        v_mean, v_std, kappa = __retrieve_mean_std_kappa(w_halfs, sigmas, h_matrix, y,
                                                                         1,
                                                                         l_matrix, f_upscaled, rg, sino)
                        __temp_plot(simulation_id, step, v_mean, v_std, kappa, extension='png'
                                    , mask=mask)
        except StopIteration:
            print('Nan value found in the gradient. Stopping the optimization loop')
        finally:
            if learn_sigmas:
                (w_halfs, sigmas) = opt.params_fn(opt_state)
            else:
                (w_halfs,) = opt.params_fn(opt_state)
            end_state_list.append(w_halfs)

        if not value_hist:
            value_hist = np.array([np.inf])
        else:
            value_hist = np.stack(value_hist)

    # print('post processing ...')
    # process_nwd_results(simulation_id + '.hdf')

    print('Optimization completed ...')
    print('Now performing the Gaussian inference ...')
    w_halfs, sigmas = __retrieve_w_halfs_and_sigmas(opt,
                                                    optimizer,
                                                    opt_state, learn_sigmas,
                                                    n_sigmas,
                                                    n_prior_layers, f)

    v_mean, v_std, kappa = __retrieve_mean_std_kappa(w_halfs, sigmas,
                                                     h_matrix, y,
                                                     n_samples, l_matrix,
                                                     f_upscaled, rg, sino)
    error = ground_truth_phantom - v_mean
    # put a mask in the error
    mask_nan_to_zero = np.nan_to_num(mask)
    error = mask_nan_to_zero * error


    # print('Computing metrics ...')
    rmse_val = metric.rmse(ground_truth_phantom.ravel(), v_mean.ravel())
    l2_error_val = metric.l_2_error(ground_truth_phantom.ravel(), v_mean.ravel())
    psnr_val = metric.psnr(ground_truth_phantom.ravel(), v_mean.ravel())
    rmlse_val = metric.rmlse(ground_truth_phantom.ravel(), v_mean.ravel())
    mae_val = metric.mean_absolute(error.ravel())
    medae_val = metric.median_absolute(error.ravel())
    mlse_val = metric.mean_log_squared(error.ravel())

    # removes unnecessary matrices
    del h_matrix
    del h_conj_t
    del identity
    del identity_m

    print('Saving local variables ...')
    save_to_hdf('./.simulation_results/{}.hdf'.format(simulation_id), locals())

    print('Post processing ...')
    process_tomography_results(file_name=simulation_id+'.hdf',
                               compute_tikhonov_regularization=compute_tikhonov_regularization,
                               compute_fbp=compute_tikhonov_regularization,
                               tikhonov_relative_const=tikhonov_relative_const)

    print('Finished ...')


def __retrieve_w_halfs_and_sigmas(opt_,
                                  optimizer_: OptimizerType,
                                  opt_state_,
                                  learn_sigmas_: bool,
                                  n_sigmas_: int,
                                  n_prior_layers_: int,
                                  f_: Fourier):
    if optimizer_ in [OptimizerType.ADA_HESSIAN]:
        a_state = opt_.params_fn(opt_state_)
        if learn_sigmas_:
            sigmas_ = a_state[:n_sigmas_]
            w_halfs_real_ = a_state[n_sigmas_:]
        else:
            w_halfs_real_ = a_state
            sigmas_ = np.ones(n_sigmas_)
        w_halfs_ = from_two_reals_ravelled_to_complex(w_halfs_real_, (2, n_prior_layers_, f_.basis_number))
    else:
        if learn_sigmas_:
            (w_halfs_, sigmas_) = opt_.params_fn(opt_state_)
        else:
            sigmas_ = np.ones(n_sigmas_)
            (w_halfs_,) = opt_.params_fn(opt_state_)

    return w_halfs_, sigmas_


def __retrieve_mean_std_kappa(w_halfs_: np.ndarray,
                              sigmas_: np.ndarray, h_matrix_: np.ndarray,
                              y_: np.ndarray,
                              n_samples_: int,
                              l_matrix_: LMatrixGenerator,
                              f_: Fourier,
                              rg_: RandomGenerator,
                              sino_: Sinogram):
    v_mean, v_std = non_analytical_mean_and_std(w_halfs_, sigmas_, h_matrix_, y_, n_samples_, l_matrix_, f_, rg_,
                                                use_lancos=False)

    u_min_1 = symmetrize(l_matrix_.u_half_min_1, False).reshape(f_.symmetric_shape, order=sino_.ravel_order)

    # scaling
    scale = 2 * f_.target_basis_number - 1
    v_mean = v_mean * scale
    v_std = v_std * (scale ** 2)

    kappa = l_matrix_.kappa_fun(f_.irfft(u_min_1[:, f_.essential_basis_number - 1:]))
    return np.flipud(v_mean), np.flipud(v_std), np.flipud(kappa)


def __temp_plot(simulation_id: str, iteration_index: int, v_mean_temp: np.ndarray, v_std_temp: np.ndarray,
                kappa_temp: np.ndarray, extension: str = "pdf", mask=None):
    plot_dir = pathlib.Path('./.plots/' + simulation_id)
    length_scale = (1 / kappa_temp)
    if not plot_dir.exists():
        plot_dir.mkdir()
    if isinstance(mask, np.ndarray):
        v_mean_temp = mask * v_mean_temp
        v_std_temp = mask * v_std_temp
        length_scale = mask * length_scale
    else:
        pass
    plot_image_and_save(v_mean_temp, './.plots/' + simulation_id + '/mean_{}.{}'.format(iteration_index, extension))
    plot_image_and_save(v_std_temp, './.plots/' + simulation_id + '/std_{}.{}'.format(iteration_index, extension))
    plot_image_and_save(length_scale, './.plots/' + simulation_id + '/ell_{}.{}'.format(iteration_index, extension))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture', default=50, type=int, help='capture_interval')
    parser.add_argument('--basis', default=16, type=int, help='fourier_basis_number')
    parser.add_argument('--target-basis', default=64, type=int, help='fourier_target_number')
    parser.add_argument('--upscale', default=1, type=int, help='restoration_upscale')
    parser.add_argument('--samples', default=1000, type=int, help='n_samples')
    parser.add_argument('--ntheta', default=45, type=int, help='n_theta')
    parser.add_argument('--prior', default=1, type=int, help='n_prior_layers')
    parser.add_argument('--steps', default=100, type=int, help='n_steps')
    parser.add_argument('--cg', default=100, type=int, help='cg_max_iter')
    parser.add_argument('--seed', default=0, type=int, help='random_seed')
    parser.add_argument('--rloop', default=10000, type=int, help='random_loop_length')

    parser.add_argument('--var', default=0.01, type=float, help='measurement_variance')
    parser.add_argument('--learn', default=1e-1, type=float, help='step_scale')
    parser.add_argument('--b1', default=0.9, type=float, help='b1_momentum')
    parser.add_argument('--b2', default=0.999, type=float, help='b2_momentum')
    parser.add_argument('--kappa0', default=1., type=float, help='kappa0')
    parser.add_argument('--momentum', default=0.1, type=float, help='momentum')
    parser.add_argument('--damping', default=1., type=float, help='initial_lm_damping')
    parser.add_argument('--maxloss', default=10000., type=float, help='required_max_loss_val')
    parser.add_argument('--tikoconst', default=5e-2, type=float, help='tikhonov_relative_const')

    add_boolean_argument(parser, 'tikhonov', default=False, messages='whether to compute_tikhonov_regularization')
    add_boolean_argument(parser, 'randomized', default=False, messages='whether to use_randomized_initialization')
    add_boolean_argument(parser, 'gnm', default=False, messages='whether to use_gnm')
    add_boolean_argument(parser, 'softmax', default=False, messages='whether to use_soft_max')
    add_boolean_argument(parser, 'cuda', default=True, messages='whether to use_cuda')
    add_boolean_argument(parser, 'learn-sigmas', default=False, messages='whether to learn_sigmas')

    parser.add_argument('--optimizer', default='adahessian', type=str, help='optimizer')

    args = parser.parse_args()
    optimizer = OptimizerType.HESSIAN_FREE
    optimizer_str = args.optimizer.lower()

    if optimizer_str == "adam":
        optimizer = OptimizerType.ADAM
    elif optimizer_str == "rmsprop":
        optimizer = OptimizerType.RMSPROP
    elif optimizer_str == "adahessian":
        optimizer = OptimizerType.ADA_HESSIAN
    elif optimizer_str == "rmspropmomentum":
        optimizer = OptimizerType.RMSPROP_MOMENTUM

    __run_simulation(fourier_basis_number=args.basis,
                     fourier_target_number=args.target_basis,
                     restoration_upscale=args.upscale,
                     capture_interval=args.capture,
                     n_samples=args.samples,
                     n_prior_layers=args.prior,
                     n_steps=args.steps,
                     n_theta=args.ntheta,
                     cg_max_iter=args.cg,
                     random_seed=args.seed,
                     random_loop_length=args.rloop,
                     measurement_variance=args.var,
                     initial_lm_damping=args.damping,
                     step_scale=args.learn,
                     kappa0=args.kappa0,
                     momentum=args.momentum,
                     b1_momentum=args.b1,
                     b2_momentum=args.b2,
                     required_max_loss_val=args.maxloss,
                     use_randomized_initialization=args.randomized,
                     learn_sigmas=args.learn_sigmas,
                     use_gnm=args.gnm,
                     use_soft_max=args.softmax,
                     use_cuda=args.cuda,
                     optimizer=optimizer,
                     compute_tikhonov_regularization=args.tikhonov,
                     tikhonov_relative_const=args.tikoconst
                     )
