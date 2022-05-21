from functools import partial
from typing import Tuple
import jax.numpy as jnp
from jax import jit
import numpy as onp

import jax.numpy as np
import jax.scipy.linalg as jsl
import jax.profiler

import inspect
import os
import sys
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from DeepSPDE import Fourier, LMatrixGenerator, RandomGenerator
from Measurements import Sinogram
from Optimizers import hessian_free
from Phantom import contrast_shepp_logan_ellipses
from utils import *
import matplotlib.pyplot as plt
sys.path.append("../Optimizers/AdaHessianJax/")
from adahessianJax.jax import adahessian
from adahessianJax import value_grad_and_hessian

@partial(jit, static_argnums=(1,))
def from_two_reals_ravelled_to_complex(state_real: np.ndarray, shape: Tuple):
    state_real_shaped = state_real.reshape(shape)
    whalf_out = state_real_shaped[0] + 1j * state_real_shaped[1]
    return whalf_out


@jit
def from_complex_w_halfs_to_ravelled_reals(w_halfs_in: np.ndarray):
    state_real = np.stack((w_halfs_in.real, w_halfs_in.imag))
    return state_real.ravel(), state_real.shape


fourier = Fourier(32, 128, 2)
n_theta = 45
n_p_prime = 2 * fourier.target_basis_number - 1
std_dev = 3e-1
thetas = onp.linspace(0, 1, n_theta, endpoint=False) * onp.pi
sino = Sinogram(contrast_shepp_logan_ellipses, fourier, thetas, n_p_prime, std_dev=std_dev)

I = np.eye(2 * fourier.basis_number - 1)
y = np.asarray(sino.measure().ravel(sino.ravel_order), dtype=jnp.float32) / sino.std_dev
y_ = np.concatenate((y, jnp.zeros(2 * fourier.basis_number - 1)))

rg = RandomGenerator(fourier.essential_basis_number, fourier.dimension, prngkey_num=1)

L = LMatrixGenerator(fourier)
L.kappa0 = 1.0
n_prior_layers = 1

w_halfs = rg.construct_w_half(n_prior_layers)
sigmas = np.ones(n_prior_layers + 1)

H_dummy = sino.measurement_matrix

H = np.asarray(sino.measurement_matrix, dtype=np.complex64) / sino.std_dev
H_conj_T = H.conj().T
norm_H = np.linalg.norm(H)
Im = np.eye(sino.ground_truth.shape[0] * sino.ground_truth.shape[1])


@jit
def loss(complex_w_halfs, sigmas):
    l_res = L.generate_last_l_matrix_from_u_half_sequence(complex_w_halfs, sigmas)
    Z = jnp.linalg.solve(l_res.conj().T, H_conj_T)
    C = jnp.linalg.cholesky(Z.conj().T @ Z + Im)
    return 0.5 * squared_norm(jsl.solve_triangular(C, y, lower=True)) + sum_log_positive_array(jnp.diag(C).real)


@partial(jit, static_argnums=(2,))
def wrapped_loss(sigmas_in, whalfs_real, whalfs_shape):
    w_halfs_in = from_two_reals_ravelled_to_complex(whalfs_real, whalfs_shape)
    return loss(w_halfs_in, sigmas_in)


def re_wrapped_loss(state_real, whalfs_shape=(2, n_prior_layers, fourier.basis_number), n_sigmas=n_prior_layers + 1):
    sigmas_in = state_real[:n_sigmas]
    u_halfs_real = state_real[n_sigmas:]
    return wrapped_loss(sigmas_in, u_halfs_real, whalfs_shape)

def optimizer_step(step,opt_state, rng):
    val, gradient, hessian = value_grad_and_hessian(re_wrapped_loss,(opt.params_fn(opt_state), ),rng, argnum=0)
    if jnp.isnan(val):
        raise StopIteration
    if jnp.any(jnp.isnan(gradient)) or jnp.any(jnp.isnan(hessian)):
        raise StopIteration
    opt_state = opt.update_fn(step, gradient, hessian, opt_state)
    return val, opt_state

w_halfs_real, w_halfs_shape = from_complex_w_halfs_to_ravelled_reals(w_halfs)
state_init = np.concatenate((sigmas, w_halfs_real))


num_step = 10000
opt = adahessian(step_size=1e-1)
opt_state = opt.init_fn(state_init)

rng = jax.random.PRNGKey(1)
value_hist = []
for step in range(num_step):
    rng, rng_param = jax.random.split(rng)
    val, opt_state = optimizer_step(step, opt_state, rng_param)
    if step % 50 == 0:
        print('Current loss = {}'.format(val))

        state_hist = opt.params_fn(opt_state)
        n_sigmas = n_prior_layers+1
        sigmas = state_hist[:n_sigmas]
        w_halfs_real = state_hist[n_sigmas:]
        w_halfs_again = from_two_reals_ravelled_to_complex(w_halfs_real,(2,n_prior_layers,fourier.basis_number))

        y_bar = onp.concatenate((y,onp.zeros(2*fourier.basis_number-1)))
        l_res = L.generate_l_matrices_from_u_half_sequence(w_halfs_again,sigmas)[0]
        Lbar = onp.vstack((onp.asarray(H,dtype=onp.complex64),onp.asarray(l_res,dtype=onp.complex64)))
        v_spde,_,_,_ = onp.linalg.lstsq(Lbar,y_bar)
        v_spde_2d = v_spde.reshape((2*fourier.essential_basis_number-1,
                          2*fourier.essential_basis_number-1),
                         order=sino.ravel_order)
        reconstructed_spde = fourier.irfft(v_spde_2d[:,fourier.essential_basis_number-1:])
        fig = plt.figure(figsize=(10,10))
        im = plt.imshow(reconstructed_spde,cmap=plt.cm.seismic_r,origin='upper')
        fig.colorbar(im)
        plt.savefig('result{0}.png'.format(step))

        #compute the last layer value
        u_min_1 = symmetrize(L.u_half_min_1,False).reshape((2*fourier.essential_basis_number-1,2*fourier.essential_basis_number-1),
                                      order=sino.ravel_order)
        res = fourier.irfft(u_min_1[:,fourier.essential_basis_number-1:])

        kappa = L.kappa_fun(res)
        plt.figure(figsize=(10,10))
        plt.imshow(1/kappa)
        plt.savefig('kappa{0}.png'.format(step))
