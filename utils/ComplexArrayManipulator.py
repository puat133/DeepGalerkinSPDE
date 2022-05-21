import jax.numpy as np
import numpy as onp
from jax import jit, custom_jvp, partial
from typing import Tuple


@custom_jvp
@jit
def sum_log_positive_array(x: np.ndarray):
    epsilon = 0.  # np.finfo(np.float32).eps
    return np.sum(np.log(np.maximum(x, epsilon)))


@sum_log_positive_array.defjvp
@jit
def __sum_log_positive_array_jvp(primals: Tuple, tangents: Tuple):
    """
    Custom implementation of jacobian vector product of sum_log_positive_array function.

    Parameters
    ----------
    primals: Tuple

    tangents: Tuple

    Returns
    -------
    out: Tuple
    """
    x, = primals
    x_dot, = tangents
    primal_out = sum_log_positive_array(x)
    tangent_out = 1 / x
    return primal_out, np.dot(x_dot, tangent_out).real


@custom_jvp
@jit
def squared_norm(x: np.ndarray):
    """
    Return squared of L^2 norm
    Parameters
    ----------
    x: numpy.array
        input array.
    Returns
    -------
    out: float
        the squared norm.
    """
    x_square = np.square(x)
    return np.sum(x_square.real + x_square.imag)


@squared_norm.defjvp
@jit
def __squared_norm_jvp(primals: Tuple, tangents: Tuple):
    """
    Custom implementation of jacobian vector product of squared_norm function.

    Parameters
    ----------
    primals: Tuple

    tangents: Tuple

    Returns
    -------
    out: Tuple
    """
    x, = primals
    x_dot, = tangents
    primal_out = squared_norm(x)
    tangent_out = x.conj()
    return primal_out, np.dot(x_dot, tangent_out).real


@partial(jit, static_argnums=(1,))
def symmetrize(w_half: np.ndarray, multi: bool) -> np.ndarray:
    """
    If dimension of the half portion of Fourier coefficients `w_half` is one,
    then it will symmetrize  from frequency :math:`0` up to :math:`N-1` ,
    and gives a full spectrum from frequency :math:`-(N-1)` up to :math:`N-1`.
    In other words it transform an array from (n,) to (2n-1,)

    If the dimension is equal to two, then it will ...
    w_half is in the shape of (2n-1,n), and it will be returned to (2n-1,2n-1)
    Parameters
    ----------
    w_half : jax.numpy.ndarray
        A half portion of complex-valued Fourier coefficients of a random field `w`.
    multi: bool, optional
        Whether the symmetrization is taken for one dimensional Fourier coefficients,
        but multiple rows. Default is false.
    Returns
    -------
    w: jax.numpy.ndarray
                A symmetrized complex-valued Fourier coefficients of a random field `w`.
    """
    if w_half.ndim == 1:
        w = np.concatenate((w_half[:0:-1].conj(), w_half))
    elif w_half.ndim == 2:
        if not multi:
            temp = w_half[:, 1:]
            temp_c = temp[::-1, :][:, ::-1].conj()
            w = np.hstack((temp_c, w_half))
        else:
            temp = w_half[:, 1:]
            w = np.hstack((temp[:, ::-1].conj(), w_half))
    return w


@partial(jit, static_argnums=(1,))
def pad_symmetric(u_sym: np.ndarray, m: int) -> np.ndarray:
    """
    Extend a complex array:

    - if `u_sym.ndim=1` then it is extended from shape `(2n-1)` to `(2m-1)`
    - if `u_sym.ndim=2` then it is extended from shape `(2n-1x2n-1)` to `(2m-1x2m-1)`
    where the shape of u_sym is `(2n-1)` for `ndim=1` or `(2n-1,2n-1)` for `ndim=2`.
    it is similar to padding

    Parameters
    ----------
    u_sym : jax.numpy.ndarray
        Symmetric input array
    m : int
    Returns
    -------
    z : jax.numpy.ndarray
        Output array
    """
    if u_sym.ndim == 1:
        n = (u_sym.shape[0] + 1) // 2
        if m > n:
            z = np.pad(u_sym, (m - n, m - n))
        else:
            z = u_sym
    elif u_sym.ndim == 2:
        n = (u_sym.shape[0] + 1) // 2
        if m > n:
            z = np.pad(u_sym, ((m - n, m - n), (m - n, m - n)))
        else:
            z = u_sym
    return z


@partial(jit, static_argnums=(1,))
def pad(u_half: np.ndarray, m: int) -> np.ndarray:
    """
    Apply padding to an array `uHalf` which is the output of `jax.numpy.fft.rfft` or `jax.numpy.fft.rfft2`.
    If `u_half.ndim` is one, it will extend `uHalf` from shape `(n,)` to shape `(m,)` while the last `m-n`
    is zero.
    Similarly, if `u_half.ndim` is two, it will extend `uHalf` from shape `(2n-1,n)` to shape `(2m-1,m)`,
    where the center of `2n-1,n` are occupied by the `uHalf` and remaining entries are zero.


    Parameters
    ----------
    u_half : jax.numpy.ndarray
        Input array
    m : int
        Target

    Returns
    -------

    """
    if u_half.ndim == 1:
        if m > u_half.shape[0]:
            u_half_padded = np.pad(u_half, (0, m - u_half.shape[0]))
        else:
            u_half_padded = u_half
    elif u_half.ndim == 2:
        if m > u_half.shape[1]:
            u_half_padded = np.pad(u_half, ((m - u_half.shape[1], m - u_half.shape[1]), (0, m - u_half.shape[1])))
        else:
            u_half_padded = u_half
    else:
        raise ValueError('Only supported one and two dimensional array')

    return u_half_padded


def create_toeplitz_indices(n: int) -> Tuple:
    """
    Create two index matrices used in the creation of Toeplitz from for two dimensional Fourier coefficient matrix.

    Parameters
    ----------
    n : int
        The Fourier essential basis number.

    Returns
    -------
    out: Tuple
        The index matrices, iy, and ix, both are with size `2n-1 x 2n-1`.
    """
    inner_length = (2 * n - 1)
    length = inner_length ** 2
    shape = (length, length)
    i_x = onp.zeros(shape, dtype=onp.int8)
    i_y = onp.zeros(shape, dtype=onp.int8)
    for i in range(inner_length):
        for j in range(inner_length):
            i_shift = i * inner_length
            j_shift = j * inner_length

            i_y[i_shift:i_shift + inner_length, j_shift:j_shift + inner_length] = (i - j) + (
                    inner_length - 1)
            for k in range(inner_length):
                for m in range(inner_length):
                    i_x[k + i_shift, m + j_shift] = (k - m) + (inner_length - 1)

    # because i_y is row index, i_x is column index
    return np.asarray(i_y, dtype=np.int8), np.asarray(i_x, dtype=np.int8)


@partial(np.vectorize, signature='(),(t)->(t)')
def eigen_function_one_d(i: int, t: np.ndarray) -> np.ndarray:
    """
    one dimensional eigen function of a Laplacian operator in (0,1)

    Parameters
    ----------
    i : int
        Frequency.
    t : numpy.ndarray
        Time signal from (0,1).

    Returns
    -------
    out: numpy.ndarray
        The function output.
    """
    return np.exp(2 * np.pi * 1j * i * t)
