from typing import Tuple

import jax.numpy as np
import jax.scipy.linalg as jsl
import numpy as onp
from jax import jit, partial

from DeepSPDE import Fourier, LMatrixGenerator, RandomGenerator
from utils import eigen_function_one_d, squared_norm, sum_log_positive_array


def non_analytical_mean_and_std(w_halfs: np.ndarray,
                                sigmas: np.ndarray,
                                h_matrix: np.ndarray,
                                y: np.ndarray,
                                n_samples: int,
                                l_matrix: LMatrixGenerator,
                                f: Fourier,
                                rg: RandomGenerator,
                                use_lancos: bool = True
                                ):
    """

    Parameters
    ----------
    use_lancos
    w_halfs
    sigmas
    h_matrix
    y
    n_samples
    l_matrix
    f
    rg

    Returns
    -------

    """
    numpoints = y.shape[0]
    l_res = l_matrix.generate_last_l_matrix_from_u_half_sequence(w_halfs, sigmas)
    y_bar = np.concatenate((y, np.zeros(2 * f.basis_number - 1)))
    if n_samples > 1:
        w_sym = rg.construct_w(n_samples)
        e_ = onp.random.randn(n_samples, numpoints)
        w_sym_ = onp.array(w_sym)
        w_bar_ = onp.hstack((e_, w_sym_))
        w_bar = np.array(w_bar_)

    lbar = np.vstack((h_matrix, l_res))
    if n_samples > 1:
        v_sym_res, _, _, _ = np.linalg.lstsq(lbar, (y_bar - w_bar).T)
    else:
        v_sym_res, _, _, _ = np.linalg.lstsq(lbar, y_bar)

    lancos_sigmas = l_matrix.fourier.lancos_sigmas

    if f.dimension == 1:
        if n_samples >= 1:
            v_half_res = v_sym_res[f.basis_number - 1:, :]
            if use_lancos:
                v_half_res = np.swapaxes(v_half_res, 0, 1) * lancos_sigmas
            else:
                v_half_res = np.swapaxes(v_half_res, 0, 1)
        else:
            v_half_res = v_sym_res[f.basis_number - 1:]
            if use_lancos:
                v_half_res = v_half_res * lancos_sigmas

    elif f.dimension == 2:
        if n_samples > 1:
            v_sym_res = v_sym_res.swapaxes(0, 1).reshape((n_samples, *f.symmetric_shape), order='F')
            if use_lancos:
                v_sym_res = v_sym_res * lancos_sigmas

            v_half_res = v_sym_res[:, :, f.essential_basis_number - 1:]
        else:
            v_sym_res_ = v_sym_res.reshape(f.symmetric_shape, order='F')
            if use_lancos:
                v_sym_res = v_sym_res_ * lancos_sigmas

            v_half_res = v_sym_res_[:, f.essential_basis_number - 1:]

    else:
        raise ValueError("Fourier dimension not supported")

    v_estim = f.irfft(v_half_res)
    if n_samples > 1:
        v_mean = onp.mean(v_estim, axis=0).real
        v_std = onp.std(v_estim, axis=0).real
    else:
        v_mean = v_estim
        v_std = np.nan * np.zeros(v_estim.shape)
    return v_mean, v_std


def analytical_mean_and_std(w_halfs: np.ndarray,
                            sigmas: np.ndarray,
                            h_matrix: np.ndarray,
                            h_conj_t_matrix: np.ndarray,
                            identity_m: np.ndarray,
                            y: np.ndarray,
                            t: np.ndarray,
                            l_matrix: LMatrixGenerator,
                            f: Fourier
                            ):
    """

    Parameters
    ----------
    w_halfs
    sigmas
    h_matrix
    h_conj_t_matrix
    identity_m
    y
    l_matrix
    t
    f

    Returns
    -------

    """
    phi = eigen_function_one_d(np.arange(-l_matrix.fourier.basis_number + 1, l_matrix.fourier.basis_number),
                               t) / np.sqrt(t.shape[0])
    l_res = l_matrix.generate_last_l_matrix_from_u_half_sequence(w_halfs, sigmas)
    y_bar = np.concatenate((y, np.zeros(2 * f.basis_number - 1)))
    lbar = np.vstack((h_matrix, l_res))

    # compute the mean
    v_sym, _, _, _ = np.linalg.lstsq(lbar, y_bar)
    v_mean = l_matrix.fourier.irfft(v_sym[l_matrix.fourier.basis_number - 1:])

    # compute the standard deviation
    l_t_l = l_res.conj().T @ l_res
    l_t_l = 0.5 * (l_t_l + l_t_l.conj().T)
    l_t_l_h_t = l_t_l @ h_conj_t_matrix
    q = l_t_l - l_t_l_h_t @ np.linalg.solve(identity_m + h_matrix @ l_t_l_h_t, l_t_l_h_t.conj().T)
    q = 0.5 * (q + q.conj().T)
    q_spatial = phi.conj().T @ q @ phi
    q_spatial = 0.5 * (q_spatial + q_spatial.conj().T)
    v_std = np.sqrt(np.diag(q_spatial.real))

    return v_mean, v_std


@partial(jit, static_argnums=(1,))
def from_two_reals_ravelled_to_complex(state_real: np.ndarray, shape: Tuple):
    """

    Parameters
    ----------
    state_real
    shape

    Returns
    -------

    """
    state_real_shaped = state_real.reshape(shape)
    whalf_out = state_real_shaped[0] + 1j * state_real_shaped[1]
    return whalf_out


@jit
def from_complex_w_halfs_to_ravelled_reals(w_halfs_in: np.ndarray):
    """

    Parameters
    ----------
    w_halfs_in

    Returns
    -------

    """
    state_real = np.stack((w_halfs_in.real, w_halfs_in.imag))
    return state_real.ravel(), state_real.shape


# @checkpoint
@jit
def _compute_c_and_log_det(h_conj_t, identity_m, l_res):
    """

    Parameters
    ----------
    complex_w_halfs
    sigma_values
    h_conj_t
    identity_m
    l_matrix

    Returns
    -------

    """
    z = np.linalg.solve(l_res.conj().T, h_conj_t)
    c = np.linalg.cholesky(z.conj().T @ z + identity_m)
    log_det = sum_log_positive_array(np.diag(c).real)
    return c, log_det


@partial(jit, static_argnums=(4,))
def __generate_l_res_and_compute_c_and_log_det(complex_w_halfs, sigma_values, h_conj_t, identity_m, l_matrix):
    """

    Parameters
    ----------
    complex_w_halfs
    sigma_values
    h_conj_t
    identity_m
    l_matrix

    Returns
    -------

    """
    l_res = l_matrix.generate_last_l_matrix_from_u_half_sequence(complex_w_halfs, sigma_values)
    return _compute_c_and_log_det(h_conj_t, identity_m, l_res)


@partial(jit, static_argnums=(5,))
def loss(complex_w_halfs, sigma_values, y, h_conj_t, identity_m, l_matrix):
    """

    Parameters
    ----------
    complex_w_halfs
    sigma_values
    y
    h_conj_t
    identity_m
    l_matrix

    Returns
    -------

    """
    c, log_det = __generate_l_res_and_compute_c_and_log_det(complex_w_halfs, sigma_values, h_conj_t, identity_m,
                                                            l_matrix)
    return 0.5 * squared_norm(jsl.solve_triangular(c, y, lower=True)) + log_det


@partial(jit, static_argnums=(5,))
def output_fun(complex_w_halfs, sigma_values, y, h_conj_t, identity_m, l_matrix):
    """

    Parameters
    ----------
    complex_w_halfs
    sigma_values
    y
    h_conj_t
    identity_m
    l_matrix

    Returns
    -------

    """
    c, log_det_ = __generate_l_res_and_compute_c_and_log_det(complex_w_halfs, sigma_values, h_conj_t, identity_m,
                                                             l_matrix)

    log_det = log_det_ + 1j * 0
    return np.concatenate((jsl.solve_triangular(c, y, lower=True), np.array([np.sqrt(log_det)])))
