import argparse
import inspect
import os
import pathlib
import sys

import jax.numpy as np
import numpy as onp

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import utils.metric as metric
from utils import plot_image_and_save, add_boolean_argument
from utils.hdf_io import load_hdf_file
from DeepSPDE import Fourier
from skimage.transform import iradon
from Phantom import contrast_shepp_logan_ellipses, multi_ellipses_fun

os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/local/cuda"


def print_metrics(ground_truth_phantom: np.ndarray, v_mean: np.ndarray, error: np.ndarray):
    rmse_val = metric.rmse(ground_truth_phantom.ravel(), v_mean.ravel())
    l2_error_val = metric.l_2_error(ground_truth_phantom.ravel(), v_mean.ravel())
    psnr_val = metric.psnr(ground_truth_phantom.ravel(), v_mean.ravel())
    rmlse_val = metric.rmlse(ground_truth_phantom.ravel(), v_mean.ravel())
    mae_val = metric.mean_absolute(error.ravel())
    medae_val = metric.median_absolute(error.ravel())
    mlse_val = metric.mean_log_squared(error.ravel())
    print('rmse_val \t = {:.6f}\t'.format(rmse_val))
    print('l2_error_val \t = {:.6f}\t'.format(l2_error_val))
    print('psnr_val \t = {:.6f}\t'.format(psnr_val))
    print('rmlse_val \t = {:.6f}\t'.format(rmlse_val))
    print('mae_val \t = {:.6f}\t'.format(mae_val))
    print('medae_val \t = {:.6f}\t'.format(medae_val))
    print('mlse_val \t = {:.6f}\t'.format(mlse_val))


def compute_thikonov_regularization(y: np.ndarray,
                                    h_matrix: np.ndarray,
                                    identity: np.ndarray,
                                    tikhonov_const: float,
                                    fo: Fourier):
    y_ = np.concatenate((y, np.zeros(h_matrix.shape[1])))
    h_ = np.vstack((h_matrix, tikhonov_const * identity))
    res = np.linalg.lstsq(h_, y_)
    # h_t_h = h_matrix.conj().T@h_matrix
    # h_t = h_matrix.conj().T
    # res = np.linalg.lstsq(h_t_h+tikhonov_const*identity, h_t@y)
    v_sym = res[0]
    v_sym_2d = v_sym.reshape(fo.symmetric_shape, order='F')
    scale = 2 * fo.target_basis_number - 1
    v_sym_2d = np.flipud(v_sym_2d)
    tikhonov_reconstruction = fo.irfft(v_sym_2d[:, fo.essential_basis_number - 1:])
    return tikhonov_reconstruction * scale


def compute_iradon(y: np.ndarray, n_theta: int, n_p_prime: int):
    scale = n_p_prime
    sino = onp.array(y.reshape((n_p_prime, n_theta), order='F')) * scale
    thetas = onp.linspace(0, 180., n_theta, endpoint=False)
    return iradon(sino, thetas)


def process_hdf_file(file_name: str,
                     relative_path: str = './.simulation_results',
                     compute_tikhonov_regularization: bool = False,
                     compute_fbp: bool = False,
                     tikhonov_relative_const: float = 5e-2):
    data_set_names = ['y', 'v_mean', 'v_std', 'kappa', 'error', 'fourier_basis_number', 'fourier_target_number',
                      'restoration_upscale', 'ground_truth_phantom', 'n_steps', '/sino/_Sinogram__measurement_matrix',
                      '/sino/_Sinogram__ground_truth',
                      'n_theta', 'n_p_prime', 'std_dev']
    array_names = ['y', 'v_mean', 'v_std', 'kappa', 'error', 'fourier_basis_number', 'fourier_target_number',
                   'restoration_upscale', 'ground_truth_phantom', 'n_steps', 'h_matrix', 'sino_truth',
                   'n_theta', 'n_p_prime', 'std_dev']
    var_ = load_hdf_file(file_name, relative_path, data_set_names, array_names)

    # f = Fourier(var_['fourier_basis_number'], var_['fourier_target_number'], 2)
    f_upscaled = Fourier(var_['fourier_basis_number'], var_['restoration_upscale'] * var_['fourier_target_number'], 2)

    # domain mask
    mask_ = onp.nan * onp.ones(f_upscaled.expected_shape)
    r_ = mask_.shape[0] / 2
    t_ = onp.linspace(-r_, r_, mask_.shape[0], endpoint=True)
    x_grid, y_grid = onp.meshgrid(t_, t_)
    mask_[x_grid ** 2 + y_grid ** 2 <= r_ ** 2] = 1
    mask = np.array(mask_)
    mask_nan_to_zero = np.nan_to_num(mask)

    simulation_id = file_name.replace('.hdf', '')

    print('Computing metrics ...')
    ground_truth_phantom = var_['ground_truth_phantom']
    v_mean = var_['v_mean']
    v_std = var_['v_std']
    kappa = var_['kappa']
    n_steps = var_['n_steps']
    error = ground_truth_phantom - v_mean
    print_metrics(v_mean=mask_nan_to_zero * v_mean,
                  ground_truth_phantom=mask_nan_to_zero * ground_truth_phantom,
                  error=mask_nan_to_zero * error)

    plot_dir = pathlib.Path('./.plots/' + simulation_id)
    extension = 'png'
    iteration_index = n_steps
    length_scale = (1 / kappa)
    if not plot_dir.exists():
        plot_dir.mkdir()
    # if isinstance(mask, np.ndarray):
    #     v_mean = mask * v_mean
    #     v_std = mask * v_std
    #     length_scale = mask * length_scale
    # else:
    #     pass
    plot_image_and_save(mask * v_mean, './.plots/' + simulation_id + '/mean_{}.{}'.format(iteration_index, extension))
    plot_image_and_save(mask * v_std, './.plots/' + simulation_id + '/std_{}.{}'.format(iteration_index, extension))
    plot_image_and_save(mask * ground_truth_phantom, './.plots/' + simulation_id +
                        '/ground_truth_{}.{}'.format("", extension))
    plot_image_and_save(mask * (ground_truth_phantom - v_mean), './.plots/'
                        + simulation_id + '/error_{}.{}'.format("", extension))
    plot_image_and_save(length_scale, './.plots/' + simulation_id + '/ell_{}.{}'.format(iteration_index, extension))

    if compute_tikhonov_regularization:
        h_matrix = var_['h_matrix']/var_['std_dev']
        identity = np.eye(h_matrix.shape[1], dtype=np.complex64)
        y = var_['y']
        tikhonov_const = tikhonov_relative_const * np.linalg.norm(h_matrix)
        tikhonov_reconstruction = compute_thikonov_regularization(y=y, h_matrix=h_matrix, identity=identity,
                                                                  tikhonov_const=tikhonov_const, fo=f_upscaled)
        error_tikhonov = (ground_truth_phantom - tikhonov_reconstruction)
        print('Tikhonov metrics: ')
        print_metrics(v_mean=mask_nan_to_zero * tikhonov_reconstruction,
                      ground_truth_phantom=mask_nan_to_zero * ground_truth_phantom,
                      error=mask_nan_to_zero * error_tikhonov)
        plot_image_and_save(mask * tikhonov_reconstruction, './.plots/'
                            + simulation_id + '/mean_tikhonov.{}'.format(extension))
        plot_image_and_save(mask * error_tikhonov, './.plots/'
                            + simulation_id + '/error_tikhonov.{}'.format(extension))

    if compute_fbp:
        y = var_['y'] * var_['std_dev']
        y_truth = var_['sino_truth'].ravel('F')
        n_theta = var_['n_theta']
        n_p_prime = var_['n_p_prime']
        mask_small = onp.nan * onp.ones((n_p_prime, n_p_prime))

        t_points = onp.linspace(-0.5, 0.5, n_p_prime)  # this is the default settings
        x_points, y_points = onp.meshgrid(t_points, t_points)
        ground_truth_phantom_small = multi_ellipses_fun(x_points, y_points, contrast_shepp_logan_ellipses)

        r_small = mask_small.shape[0] / 2
        t_small = onp.linspace(-r_small, r_small, mask_small.shape[0], endpoint=True)
        x_grid_small, y_grid_small = onp.meshgrid(t_small, t_small)
        mask_small[x_grid_small ** 2 + y_grid_small ** 2 <= r_small ** 2] = 1
        mask_small_nan_to_zero = onp.nan_to_num(mask_small)
        fbp_image_truth = compute_iradon(y_truth, n_theta, n_p_prime)
        fbp_image = compute_iradon(y, n_theta, n_p_prime)
        error_fbp = ground_truth_phantom_small - fbp_image
        print('FBP metrics: ')
        print_metrics(v_mean=mask_small_nan_to_zero * fbp_image,
                      ground_truth_phantom=mask_small_nan_to_zero * ground_truth_phantom_small,
                      error=mask_small_nan_to_zero * error_fbp)
        plot_image_and_save(mask_small * fbp_image_truth, './.plots/'
                            + simulation_id + '/fbp_true.{}'.format(extension))
        plot_image_and_save(mask_small * fbp_image, './.plots/'
                            + simulation_id + '/fbp.{}'.format(extension))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', default='', type=str, help='file_name')
    parser.add_argument('--path', default='./.simulation_results', type=str, help='relative_path')
    parser.add_argument('--tikoconst', default=5e-2, type=float, help='tikhonov_relative_const')
    add_boolean_argument(parser, 'tikhonov', default=False, messages='whether to compute_tikhonov_regularization')
    add_boolean_argument(parser, 'fbp', default=False, messages='whether to compute_fbp')
    args = parser.parse_args()

    process_hdf_file(file_name=args.file,
                     relative_path=args.path,
                     tikhonov_relative_const=args.tikoconst,
                     compute_tikhonov_regularization=args.tikhonov,
                     compute_fbp=args.fbp)
