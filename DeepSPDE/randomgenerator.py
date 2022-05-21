import jax.random as jrandom
import jax.numpy as np
from jax.ops import index_update, index

from utils import symmetrize
from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
class RandomGenerator(object):
    """
    THis class facilitates a construction of complex-valued Fourier coefficients of a random field `w`, from frequency
    :math:`-(N-1)` up to :math:`N-1` . For the case of dimension :math:`d` of the field `w` is greater than one,
    :math:`N = ((2n-1)^d-1)/2`. The class also support construction of the half portion version.
    """

    def __init__(self, essential_basis_number: int, dimension: int, prngkey_num: int = 0):
        """

        Parameters
        ----------
        essential_basis_number : int
            Number of Fourier coefficients in one dimension.
        dimension: int
            Dimension of the random field. Support only one and two.
        prngkey_num: int, optional
            Pseudo random generator seed to be passed to Jax.
        """
        self.__essential_basis_number = essential_basis_number
        self.__dimension = dimension

        self.__sqrt2 = np.sqrt(2)
        self.__prngkey_num = prngkey_num
        self.__PRNGKey = jrandom.PRNGKey(self.__prngkey_num)
        assert self.__dimension <= 2
        if self.__dimension == 1:
            self.__basis_number = self.__essential_basis_number
        elif self.__dimension == 2:
            self.__basis_number = 2 * self.__essential_basis_number * self.__essential_basis_number \
                                  - 2 * self.__essential_basis_number + 1

    @property
    def dimension(self):
        return self.__dimension

    @property
    def essential_basis_number(self):
        return self.__essential_basis_number

    @property
    def basis_number(self):
        return self.__basis_number

    # @partial(jit, static_argnums=(0,))
    def construct_w_half(self, n: int = None) -> np.ndarray:
        """
        Construct the half portion of complex-valued Fourier coefficients of a random field `w`, from frequency
               :math:`0` up to :math:`N-1` .
        Parameters
        ----------
        n : int
            How many columns, default is None, which return a one dimensional array. If specified, then
            the output is `N`x`n` array.
        Returns
        -------
        w_half: jax.numpy.ndarray
            A half portion of complex-valued Fourier coefficients of a random field `w`.

        """

        self.__PRNGKey, *subkeys = jrandom.split(self.__PRNGKey, 3)
        if not n:
            w_half = jrandom.normal(subkeys[0], shape=(self.__basis_number,)) + \
                     1j * jrandom.normal(subkeys[1], shape=(self.__basis_number,))
            w_half = index_update(w_half, index[0], np.sqrt(2) * w_half[0].real)
        else:
            w_half = jrandom.normal(subkeys[0], shape=(n, self.__basis_number)) + \
                     1j * jrandom.normal(subkeys[1], shape=(n, self.__basis_number))
            w_half = index_update(w_half, index[:, 0], np.sqrt(2) * w_half[:, 0].real)

        return w_half / np.sqrt(2)

    # @partial(jit, static_argnums=(0,))
    def construct_w(self, n: int = None) -> np.ndarray:
        """
        Construct the half portion of complex-valued Fourier coefficients of a random field `w`, from frequency
        :math:`-(N-1)` up to :math:`N-1` .
        Returns
        -------
        w: jax.numpy.ndarray
            A symmetrized complex-valued Fourier coefficients of a random field `w`.
        """
        w_half = self.construct_w_half(n)
        if not n:
            w = symmetrize(w_half, False)
        else:
            w = symmetrize(w_half, True)
        return w

    def tree_flatten(self):
        return ((self.__essential_basis_number, self.__dimension, self.__prngkey_num), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)
