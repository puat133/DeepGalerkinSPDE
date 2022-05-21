from typing import Callable, Tuple, Union

import jax.lax as lax
import jax.numpy as np
import jax.scipy.special as jss
from jax import jit, partial

from DeepSPDE import Fourier
from utils import create_toeplitz_indices, pad_symmetric, symmetrize


@jit
def default_kappa_fun(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return np.exp(-x)


class LMatrixGenerator(object):
    def __init__(self, f: Fourier, kappa0: float = 1.,
                 kappa_fun: Callable[[Union[float, np.ndarray]], Union[float, np.ndarray]] = default_kappa_fun):
        """

        Parameters
        ----------
        f : fourier
            Instance of Fourier class.
        """
        self.__fourier: Fourier = f
        self.__id_matrix: np.ndarray = np.eye(2 * self.__fourier.basis_number - 1)
        self.__nu = 2 - self.__fourier.dimension / 2
        self.__alpha = 2
        sqrt_beta_per_sigma = np.sqrt(
            2 ** self.__fourier.dimension * np.pi ** (self.__fourier.dimension / 2)) * np.exp(
            0.5 * (jss.gammaln(self.__alpha) - jss.gammaln(self.__nu)))
        self.__sqrt_beta_per_sigma = float(sqrt_beta_per_sigma)
        self.__kappa_0 = kappa0
        self.__kappa_fun = kappa_fun
        self.__kappa_pow_d_min_nu_fun = lambda x: self.__kappa_fun(x * (self.__fourier.dimension - self.__nu))
        self.__kappa_pow_min_nu_fun = lambda x: self.__kappa_fun(-self.__nu * x)
        self.__u_half = np.empty(self.__fourier.dimension)
        self.__ravel_order = 'F'
        self.__i_y = None
        self.__i_x = None

        if self.__fourier.dimension == 1:
            self.__d_diag: np.ndarray = -(2 * np.pi) ** 2 * np.arange(-(self.__fourier.basis_number - 1),
                                                                      self.__fourier.basis_number) ** 2
        elif self.__fourier.dimension == 2:
            self.__i_y, self.__i_x = create_toeplitz_indices(self.__fourier.essential_basis_number)

            temp = np.arange(-(self.__fourier.essential_basis_number - 1), self.__fourier.essential_basis_number,
                             dtype=np.int32)

            ix, iy = np.meshgrid(temp, temp)
            self.__d_diag: np.ndarray = -(2 * np.pi) ** 2 * (ix.ravel(self.__ravel_order) ** 2 +
                                                             iy.ravel(self.__ravel_order) ** 2)

    @property
    def nu(self):
        return self.__nu

    @property
    def sqrt_beta_per_sigma(self):
        return self.__sqrt_beta_per_sigma

    @property
    def fourier(self):
        return self.__fourier

    @property
    def d_diag(self):
        return self.__d_diag

    @property
    def id_matrix(self):
        return self.__id_matrix

    @property
    def kappa_fun(self):
        return self.__kappa_fun

    @property
    def kappa0(self):
        return self.__kappa_0

    @kappa0.setter
    def kappa0(self, value):
        # if value < 0:
        #     raise ValueError('Invalid value for kappa_0. Only positive real value is allowed.')
        # else:
        self.__kappa_0 = value

    @property
    def u_half_min_1(self):
        return self.__u_half

    @partial(jit, static_argnums=(0,))
    def toeplitz_form(self, u_half: np.ndarray) -> np.ndarray:
        @jit
        def __toeplitz_body_scan(carry_, inputs_) -> Tuple:
            u_sym_extended = inputs_
            n_ = 1 + u_sym_extended.shape[0] // 4
            i = carry_
            middle = (u_sym_extended.shape[0]) // 2
            u_row = lax.dynamic_slice_in_dim(u_sym_extended, middle - i, (2 * n_ - 1))
            i += 1
            return i, u_row

        if self.__fourier.dimension != u_half.ndim:
            raise ValueError('u_half.ndim does not match with \
                                    Fourier dimension. Expected {} given {}'.format(u_half.ndim,
                                                                                    self.__fourier.dimension))
        if self.__fourier.dimension == 1:
            n = u_half.shape[0]
            if n != self.__fourier.basis_number:
                raise ValueError('u_half.shape[0] does not match with \
                                        Fourier basis_number. Expected {} given {}'.format(n,
                                                                                           self.__fourier.basis_number))
            u_sym_extendeds = np.tile(pad_symmetric(symmetrize(u_half, False), 2 * n - 1), (2 * n - 1, 1))
            inputs = u_sym_extendeds
            carry = 0
            _, u_matrix = lax.scan(__toeplitz_body_scan, carry, inputs, length=2 * n - 1)
            return u_matrix
        elif self.__fourier.dimension == 2:
            u_sym = symmetrize(u_half, False).reshape(self.__fourier.symmetric_shape)
            u_sym_2d_extended = pad_symmetric(u_sym, 2 * self.__fourier.essential_basis_number - 1)
            return u_sym_2d_extended[self.__i_y, self.__i_x]

    @partial(jit, static_argnums=(0, 1))
    def __apply_fun_on_original_domain(self, fun: Callable[[np.ndarray], np.ndarray],
                                       u_half: np.ndarray) -> np.ndarray:
        """
        Take an inverse FFT of a vector with Fourier elements, then apply a function in temporal/spatial function
        then transform back to the FFT domain.

        Parameters
        ----------
        fun: Callable[[np.ndarray], np.ndarray]
            Function to be applied
        u_half: np.ndarray
            input vector


        Returns
        -------
        out: np.ndarray
            The output vector
        """

        # @checkpoint
        def __sandwich(w_half):
            return self.__fourier.rfft(fun(self.__fourier.irfft(w_half)))

        if u_half.shape[0] != self.__fourier.basis_number:
            raise ValueError('u_half.shape[0] does not match with \
                            Fourier basis_number. Expected {} given {}'.format(u_half.shape[0],
                                                                               self.__fourier.basis_number))
        u_half_to_be_processed = u_half
        if self.__fourier.dimension == 2:
            u_sym = symmetrize(u_half, False).reshape(self.__fourier.symmetric_shape)
            u_half_to_be_processed = u_sym[:, self.__fourier.essential_basis_number - 1:]
        out = __sandwich(u_half_to_be_processed)
        return out

    @partial(jit, static_argnums=(0,))
    def __compute_L_from_u_half_and_sqrt_beta(self, u_half: np.ndarray, sqrt_beta: float):
        kappa_pow_d_min_nu = self.toeplitz_form(self.__apply_fun_on_original_domain(self.__kappa_pow_d_min_nu_fun,
                                                                                    u_half))
        kappa_u_pow_min_nu = self.toeplitz_form(self.__apply_fun_on_original_domain(self.__kappa_pow_min_nu_fun,
                                                                                    u_half))
        l_matrix = (kappa_u_pow_min_nu * self.__d_diag - kappa_pow_d_min_nu) / sqrt_beta
        return l_matrix

    @partial(jit, static_argnums=(0,))
    def generate_static_l_matrix_diagonal_elements(self, sigma_0: float) -> np.ndarray:
        """
        Calculate static L matrix diagonal element from a given `sigma_0` value.

        Parameters
        ----------
        sigma_0: float
            The strength of the random field.

        Returns
        ------
        out: jax.numpy.ndarray
            Diagonal elements of L matrix
        """

        sqrt_beta_0 = sigma_0 * self.__sqrt_beta_per_sigma
        l_matrix_diag_elem = (self.__d_diag * self.__kappa_0 ** (-self.__nu) -
                              self.__kappa_0 ** (self.__fourier.dimension - self.__nu)) / sqrt_beta_0
        return l_matrix_diag_elem

    def generate_last_l_matrix_from_u_half_sequence(self, w_halfs: np.ndarray, sigmas: np.ndarray) -> np.ndarray:
        """
        Apply chains of mapping from w_halfs and sigma to obtain the L matrix, see equation... of the
        paper. Here `w_halfs` is a `N`x`J` complex array, where `N` is the Fourier basis dimension and `J` is
        the number of hyper-prior layers, sigma is `J` real valued array. The output is a `2N-1` x `2N-1` array,
        which represent the `L` matrix of the `J-1`-th layer

        Parameters
        ----------
        w_halfs : jax.numpy.ndarray
            input array.
        sigmas : jax.numpy.ndarray
            input sigmas

        Returns
        -------
        out: jax.numpy.ndarray
            The output L matrix of each layers.
        """
        l_matrices = self.generate_l_matrices_from_u_half_sequence(w_halfs, sigmas)

        return l_matrices[-1]

    def generate_l_matrices_from_u_half_sequence(self, w_halfs: np.ndarray, sigmas: np.ndarray) -> np.ndarray:
        """
        Apply chains of mapping from w_halfs and sigma to obtain the L matrix, see equation... of the
        paper. Here `w_halfs` is a `N`x`J` complex array, where `N` is the Fourier basis dimension and `J` is
        the number of hyper-prior layers, sigma is `J` real valued array. The output is a `2N-1` x `2N-1` array,
        which represent the `L` matrix of the `J-1`-th layer

        Parameters
        ----------
        w_halfs : jax.numpy.ndarray
            input array.
        sigmas : jax.numpy.ndarray
            input sigmas

        Returns
        -------
        out: jax.numpy.ndarray
            The output L matrix of each layers.
        """

        def __generator_body_scan(car: Tuple, inp: Tuple) -> Tuple:
            def __generator_true_fun(operand_: Tuple) -> np.ndarray:
                l_matrix_, w_sym_, u_half__ = operand_
                u_sym_ = np.linalg.solve(l_matrix_, w_sym_)
                return u_sym_[self.__fourier.basis_number - 1:]

            def __generator_false_fun(operand_: Tuple) -> np.ndarray:
                _, _, u_half__ = operand_
                return u_half__

            w_sym, sqrt_beta, len_w_halfs = inp
            u_half_, i_ = car
            l_matrix = self.__compute_L_from_u_half_and_sqrt_beta(u_half_, sqrt_beta)
            operand = (l_matrix, w_sym, u_half_)
            u_half_ = lax.cond(i_ < len_w_halfs - 1, __generator_true_fun, __generator_false_fun, operand)
            i_ += 1
            return (u_half_, i_), l_matrix

        w_syms = symmetrize(w_halfs, True)
        sqrt_betas = sigmas * self.__sqrt_beta_per_sigma
        l_matrix_diag = self.generate_static_l_matrix_diagonal_elements(sigmas[0])
        u_half = w_halfs[0] / l_matrix_diag[self.__fourier.basis_number - 1:]
        iteration_limit = w_halfs.shape[0] * np.ones(w_halfs.shape[0], dtype=np.int32)
        inputs = (w_syms, sqrt_betas[:-1], iteration_limit)
        i = 0
        carry = (u_half, i)
        (u_half, i), l_matrices = lax.scan(__generator_body_scan, carry, inputs, length=w_halfs.shape[0])

        #  set the last layer here
        self.__u_half = u_half

        return l_matrices

    def tree_flatten(self):
        auxiliaries = default_kappa_fun
        return ((self.__fourier, self.__kappa_fun), auxiliaries)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, aux_data)
