import matplotlib.pyplot as plt
import numpy as onp
import jax.numpy as np
import argparse
import os
import sys
import inspect

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import utils.metric as metric
from utils import set_plot, add_boolean_argument
from utils.hdf_io import load_hdf_file
from DeepSPDE import LMatrixGenerator, Fourier, RandomGenerator
from utils.loss_function_and_support import non_analytical_mean_and_std as non_analytical_mean_and_std

fig_size, lw, marker_size = set_plot(use_latex=True)


def __get_statistics(signals):
    return onp.min(signals), onp.mean(signals), onp.median(signals), onp.max(signals), onp.std(signals)


def __plot_estimation(simulation_id, v_mean, v_std, t, y, ground_truth):
    # plot only the last sample
    plt.figure(figsize=fig_size)
    plt.fill_between(t, v_mean + 2 * v_std, v_mean - 2 * v_std, color='b', alpha=.25)
    plt.plot(t, v_mean, linewidth=lw)
    plt.plot(t, ground_truth, '-.r', linewidth=lw)
    plt.plot(t, y, '.k', markersize=6, linewidth=lw)
    plt.grid()
    plt.ylabel('$v$')
    plt.xlabel('$t$')
    plt.tight_layout()
    plt.savefig('./.plots/v_{}.pdf'.format(simulation_id))
    plt.close()
    # compute the last layer value


def __plot_length_scale(simulation_id, t, kappa):
    plt.figure(figsize=fig_size)
    plt.semilogy(t, onp.flip(1 / kappa))
    plt.grid()
    plt.ylabel('$\ell$')
    plt.xlabel('$t$')
    plt.tight_layout()
    plt.savefig('./.plots/ell_{}.pdf'.format(simulation_id))
    plt.close()


def __plot_loss_val(simulation_id, loss):
    plt.figure(figsize=fig_size)
    plt.semilogy(loss.T, linewidth=0.1, alpha=0.2, color='blue')
    plt.semilogy(onp.mean(loss, axis=0), linewidth=1)
    plt.grid()
    plt.ylabel('loss')
    plt.xlabel('step')
    plt.tight_layout()
    plt.savefig('./.plots/loss_{}.pdf'.format(simulation_id))
    plt.close()


def process_hdf_file(file_name: str,
                     relative_path: str = './.simulation_results',
                     simulation_index: int = -1,
                     redo_statistics: bool = False,
                     use_lancos: bool = False):
    if not redo_statistics:
        data_set_names = ['v_mean_all', 'v_std_all', 'kappa_all', 'loss_val_all', 'meas_samples',
                          '/dod/_DenoisingOneD__t', '/dod/_DenoisingOneD__ground_truth_signal',
                          'loss_val_all']
        array_names = ['v_mean_all', 'v_std_all', 'kappa_all', 'loss_val_all', 'meas_samples',
                       't', 'ground_truth_signal', 'loss_val_all']
        var_ = load_hdf_file(file_name, relative_path, data_set_names, array_names)
    else:
        print('Recalculating again the inverse Fourier and statistics')

        data_set_names = ['end_state_all', 'sigmas', 'h_matrix', 'normalized_meas_samples', 'kappa0', 'use_soft_max',
                          'fourier_basis_number', 'fourier_target_number', 'meas_samples', 'n_prior_layers',
                          '/dod/_DenoisingOneD__t', '/dod/_DenoisingOneD__ground_truth_signal',
                          '/dod/_DenoisingOneD__num_points',
                          'loss_val_all']
        array_names = ['end_state_all', 'sigmas', 'h_matrix', 'normalized_meas_samples', 'kappa0', 'use_soft_max',
                       'fourier_basis_number', 'fourier_target_number', 'meas_samples', 'n_prior_layers',
                       't', 'ground_truth_signal', 'num_points', 'loss_val_all']
        var_ = load_hdf_file(file_name, relative_path, data_set_names, array_names)
        f = Fourier(var_['fourier_basis_number'], var_['fourier_target_number'], 1)
        rg = RandomGenerator(f.basis_number, f.dimension)
        n_samples = var_['normalized_meas_samples'].shape[0]
        num_points = var_['num_points']
        if var_['use_soft_max']:
            kappa_fun = lambda x: np.log(np.exp(x) + 1)
            l_matrix = LMatrixGenerator(f, var_['kappa0'], kappa_fun=kappa_fun)
        else:
            l_matrix = LMatrixGenerator(f, var_['kappa0'])

        v_mean_list = []
        v_std_list = []
        kappa_list = []

        if var_['end_state_all'].dtype == np.float32:
            n_prior_layers = var_['n_prior_layers']
            end_state_all = var_['end_state_all']
            whalfs_shape = (end_state_all.shape[0], 2, n_prior_layers, f.basis_number)
            state_real_shaped = end_state_all.reshape(whalfs_shape)
            state_real_shaped = np.swapaxes(state_real_shaped, 0, 1)
            var_['end_state_all'] = state_real_shaped[0] + 1j * state_real_shaped[1]

        for i in range(n_samples):
            y = var_['normalized_meas_samples'][i]
            w_halfs = var_['end_state_all'][i]
            sigmas = var_['sigmas']
            h_matrix = var_['h_matrix']

            v_mean, v_std = non_analytical_mean_and_std(w_halfs, sigmas, h_matrix, y, n_samples, l_matrix, f, rg,
                                                        use_lancos=use_lancos)
            kappa = l_matrix.kappa_fun(f.irfft(l_matrix.u_half_min_1))
            v_mean_list.append(v_mean)
            v_std_list.append(v_std)
            kappa_list.append(kappa)

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
        del v_mean_list
        del v_std_list
        del kappa_list
        var_['v_mean_all'] = v_mean_all
        var_['v_std_all'] = v_std_all
        var_['kappa_all'] = kappa_all

    simulation_id = file_name.replace('.hdf', '')
    if simulation_index > var_['meas_samples'].shape[0]:
        raise IndexError('list index out of range')

    print('recalculating some metrics ...')
    rmse_val = metric.rmse(var_['ground_truth_signal'], var_['v_mean_all'])
    l2_error_val = metric.l_2_error(var_['ground_truth_signal'], var_['v_mean_all'])
    psnr_val = metric.psnr(var_['ground_truth_signal'], var_['v_mean_all'])
    rmlse_val = metric.rmlse(var_['ground_truth_signal'], var_['v_mean_all'])
    mae_val = metric.mean_absolute(var_['ground_truth_signal'] - var_['v_mean_all'])
    medae_val = metric.median_absolute(var_['ground_truth_signal'] - var_['v_mean_all'])
    mlse_val = metric.mean_log_squared(var_['ground_truth_signal'] - var_['v_mean_all'])

    __plot_estimation(simulation_id,
                      var_['v_mean_all'][simulation_index],
                      var_['v_std_all'][simulation_index],
                      var_['t'],
                      var_['meas_samples'][simulation_index],
                      var_['ground_truth_signal'])

    __plot_length_scale(simulation_id,
                        var_['t'],
                        var_['kappa_all'][simulation_index])

    __plot_loss_val(simulation_id, var_['loss_val_all'])

    template_string = '\t: min {:.3f}\t| mean {:.3f}\t| median {:.3f} \t| max {:.3f}\t| stdev {:.3f}'
    print('L2 Error' + template_string.format(*__get_statistics(l2_error_val)))
    print('PSNR\t' + template_string.format(*__get_statistics(psnr_val)))
    print('RMSE\t' + template_string.format(*__get_statistics(rmse_val)))
    print('RMLSE\t' + template_string.format(*__get_statistics(rmlse_val)))
    print('Mean-AE\t' + template_string.format(*__get_statistics(mae_val)))
    print('Median-AE' + template_string.format(*__get_statistics(medae_val)))
    print('MLSE\t' + template_string.format(*__get_statistics(mlse_val)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', default='Smooth', type=str, help='file_name')
    parser.add_argument('--path', default='./.simulation_results', type=str, help='relative_path')
    parser.add_argument('--index', default=-1, type=int, help='simulation_index')
    add_boolean_argument(parser, 'redo', default=False, messages='whether to redo_statistics')
    add_boolean_argument(parser, 'lancos', default=False, messages='whether to use_lancos')
    args = parser.parse_args()
    process_hdf_file(file_name=args.file,
                     relative_path=args.path,
                     simulation_index=args.index,
                     redo_statistics=args.redo,
                     use_lancos=args.lancos)
