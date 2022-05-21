import jax.numpy as np
import jax.numpy.fft as jfft
import numpy as onp
from jax import jit, partial
from jax.tree_util import register_pytree_node_class

from utils import pad_symmetric, symmetrize


def lancos_sigmas(dimension: int, essential_basis_number: int) -> np.ndarray:
    """
    sigma Lancos coefficients for calculating inverse Fourier Transforms
    """

    if dimension == 1:
        k = np.arange(1, essential_basis_number + 1)
        temp = np.sin(np.pi * (k / (essential_basis_number + 1)))
        sigmas = temp / (np.pi * (k / (essential_basis_number + 1)))

    elif dimension == 2:
        temp = onp.zeros(2 * essential_basis_number - 1)
        for i in onp.arange(2 * essential_basis_number - 1):
            k = i - (essential_basis_number - 1)
            if k == 0:
                temp[i] = 1
                continue
            else:
                temp[i] = onp.sin(onp.pi * (k / essential_basis_number)) / (
                        onp.pi * (k / essential_basis_number))

        sigmas = np.array(onp.outer(temp, temp), dtype=np.float32)
    else:
        sigmas = np.ones(essential_basis_number)

    return sigmas


@register_pytree_node_class
class Fourier(object):
    """
    A class that facilitates the computation of Fourier transform and inverse Fourier transform for
    `dimension` dimensional random fields, where for one axis, the Fourier basis number is given by
    `essential_basis_number`. Only supported at the moment for `dimension` equals to one or two.
    The domain is assumed to be :math:`\Omega = (0,1)^d` where `d` equals to `dimension`.
    """

    def __init__(self, essential_basis_number: int, target_basis_number: int, dimension: int):
        self.__essential_basis_number = essential_basis_number
        self.__target_basis_number = target_basis_number
        self.__dimension = dimension
        assert self.__dimension <= 2
        if self.__dimension == 1:
            self.__basis_number = self.__essential_basis_number
            self.__symmetric_shape = (2 * self.__basis_number - 1,)
            self.__half_shape = (self.__essential_basis_number,)
            self.__expected_shape = (2 * self.__target_basis_number - 1,)
            self.__irfft_signature = '(n)->(m)'
            self.__rfft_signature = '(m)->(n)'
        elif self.__dimension == 2:
            self.__basis_number = 2 * self.__essential_basis_number * self.__essential_basis_number \
                                  - 2 * self.__essential_basis_number + 1
            self.__symmetric_shape = (2 * self.__essential_basis_number - 1,
                                      2 * self.__essential_basis_number - 1)
            self.__half_shape = (2 * self.__essential_basis_number - 1,
                                 self.__essential_basis_number)
            self.__expected_shape = (2 * self.__target_basis_number - 1,
                                     2 * self.__target_basis_number - 1)
            self.__irfft_signature = '(n,m)->(k,k)'
            self.__rfft_signature = '(k,k)->(n,m)'
        self.__sigmas = np.zeros(self.__essential_basis_number)
        self.__sigmas_lanscoz_computed = False

        self.irfft = np.vectorize(self.__irfft, signature=self.__irfft_signature)
        self.rfft = np.vectorize(self.__rfft, signature=self.__rfft_signature)

    @property
    def settings(self):
        return self.__settings

    @property
    def symmetric_shape(self):
        return self.__symmetric_shape

    @property
    def expected_shape(self):
        return self.__expected_shape

    @property
    def half_shape(self):
        return self.__half_shape

    @property
    def dimension(self):
        return self.__dimension

    @property
    def essential_basis_number(self):
        return self.__essential_basis_number

    @property
    def basis_number(self):
        return self.__basis_number

    @property
    def target_basis_number(self):
        return self.__target_basis_number

    @partial(jit, static_argnums=(0,))
    def __rfft(self, z: np.ndarray) -> np.ndarray:
        """
        Compute the 2-dimensional FFT of a real array.

        Parameters
        ----------
        self :
        z : jax.numpy.ndarray
            Input array, taken to be real.

        Returns
        -------
        out: jax.numpy.ndarray
            The result of the real 2-D FFT.
        """
        if z.ndim == self.dimension:
            if self.dimension == 2:
                expected_shape = (2 * self.target_basis_number - 1,
                                  2 * self.target_basis_number - 1)
                if z.shape == expected_shape:
                    m = z.shape[0]
                    temp = jfft.rfft2(z)
                    temp = jfft.fftshift(temp, axes=0)
                    out = temp[m // 2 - (self.essential_basis_number - 1):m // 2 + self.essential_basis_number,
                               : self.essential_basis_number] / (2 * self.target_basis_number - 1)
                else:
                    raise ValueError('Invalid input shape. Expected {} given {}'.format(
                        expected_shape, z.shape))

            elif self.dimension == 1:
                expected_shape = (2 * self.target_basis_number - 1,)
                if z.shape == expected_shape:
                    temp = jfft.rfft(z)
                    out = temp[:self.essential_basis_number] / (2 * self.target_basis_number - 1)

                else:
                    raise ValueError('Invalid input shape. Expected {} given {}'.format(
                        expected_shape, z.shape))
        else:
            raise ValueError('Invalid input dimension. Expected {} given {}'.format(self.dimension,
                                                                                    z.ndim))
        return out

    @partial(jit, static_argnums=(0,))
    def __irfft(self, u_half: np.ndarray) -> np.ndarray:
        """
        Compute the 2-dimensional inverse FFT of a real array.
        Parameters
        ----------
        self

        u_half : the Fourier coefficients from the output of rfft

        Returns
        -------
        out: jax.numpy.ndarray
            The result of the real inverse FFT.
        """
        if u_half.ndim == self.dimension:
            if self.dimension == 2:
                expected_shape = (2 * self.essential_basis_number - 1, self.essential_basis_number)
                if u_half.shape == expected_shape:
                    u_sym_shifted = jfft.ifftshift(pad_symmetric(symmetrize(u_half, False), self.target_basis_number),
                                                   axes=(0, 1))
                    out = jfft.ifft2(u_sym_shifted).real * (2 * self.target_basis_number - 1)
                else:
                    raise ValueError('Invalid input shape. Expected {} given {}'.format(
                        expected_shape, u_half.shape))

            elif self.dimension == 1:
                expected_shape = (self.basis_number,)
                if u_half.shape == expected_shape:
                    u_sym_shifted = jfft.ifftshift(pad_symmetric(symmetrize(u_half, False), self.target_basis_number),
                                                   axes=0)
                    out = jfft.ifft(u_sym_shifted).real * (2 * self.target_basis_number - 1)
                else:
                    raise ValueError('Invalid input shape. Expected {} given {}'.format(
                        expected_shape, u_half.shape))
        else:
            raise ValueError('Invalid input dimension. Expected {} given {}'.format(self.dimension,
                                                                                    u_half.ndim))

        return out

    @property
    def lancos_sigmas(self):
        """
        sigma Lancos coefficients for calculating inverse Fourier Transforms
        """
        if not self.__sigmas_lanscoz_computed:
            self.__sigmas = lancos_sigmas(self.__dimension, self.__essential_basis_number)
            self.__sigmas_lanscoz_computed = True

        return self.__sigmas

    def tree_flatten(self):
        # auxiliaries = (self.__basis_number, self.__symmetric_shape, self.__half_shape, self.__expected_shape,
        #                self.__rfft_signature, self.__irfft_signature)
        auxiliaries = None
        return ((self.__essential_basis_number, self.__target_basis_number, self.__dimension), auxiliaries)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)
