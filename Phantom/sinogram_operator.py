import numpy as np


def sinogram_matrix(r: np.ndarray, theta: np.ndarray, ix: np.ndarray, iy: np.ndarray):
    """
    Compute a sinogram measurement matrix based on equation xxx of the paper:
    Nonlineaer multi-layered Gaussian process prior

    Parameters
    ----------
    r
    theta
    ix
    iy

    Returns
    -------
    out: numpy.ndarray
        the matrix
    """
    # theta = -theta+0.5*np.pi
    s_theta = np.sin(theta)
    c_theta = np.cos(theta)
    one_r = np.ones_like(r, dtype=np.complex64)
    one_i = np.ones_like(ix, dtype=np.complex64)
    k_tilde_u = np.outer(c_theta, ix) + np.outer(s_theta, iy)
    k_tilde_v = -np.outer(s_theta, ix) + np.outer(c_theta, iy)
    el = np.sqrt(0.25 - r * r)
    res = np.exp(1j * np.pi * (np.outer(one_r, (ix + iy)) + 2 * k_tilde_u * np.outer(r, one_i)))
    factor = (np.sin(2 * np.pi * k_tilde_v * np.outer(el, one_i))) / (np.pi * k_tilde_v)
    limiting = 2 * np.outer(el, one_i)
    mask = np.isnan(factor)
    factor[mask] = limiting[mask]
    return res * factor
