import jax.numpy as np
from jax import partial


@partial(np.vectorize, signature='(n)->()')
def mean_squared(signals):
    """
    compute mean squared value of a signals in shape (m_1,m_2,..., n), where the mean squared
    is computed along the last axis.

    Parameters
    ----------
    signals: np.ndarray
        Input signals

    Returns
    -------
    output: np.ndarray
        The meansquared value.

    """
    return np.sum(np.square(signals)) / signals.shape[-1]


@partial(np.vectorize, signature='(n)->()')
def mean_absolute(signals):
    """
    compute mean absolute value of a signals in shape (m_1,m_2,..., n), where the mean squared
    is computed along the last axis.

    Parameters
    ----------
    signals: np.ndarray
        Input signals

    Returns
    -------
    output: np.ndarray
        The mean absolute value.

    """
    return np.sum(np.abs(signals)) / signals.shape[-1]


@partial(np.vectorize, signature='(n)->()')
def median_absolute(signals):
    """
    median mean absolute value of a signals in shape (m_1,m_2,..., n), where the mean squared
    is computed along the last axis.

    Parameters
    ----------
    signals: np.ndarray
        Input signals

    Returns
    -------
    output: np.ndarray
        The median absolute value.

    """
    return np.median(np.abs(signals))


@partial(np.vectorize, signature='(n),(n)->()')
def mean_absolute_percentage_error(x_true, x_pred):
    """Mean absolute percentage error regression loss.
    Note here that we do not represent the output as a percentage in range
    [0, 100]. Instead, we represent it in range [0, 1/eps]. Read more in the
    :ref:`User Guide <mean_absolute_percentage_error>`.
    .. versionadded:: 0.24
    Parameters
    ----------
    x_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    x_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.
    Returns
    -------
    loss : float or ndarray of floats in the range [0, 1/eps]
        MAPE output is non-negative floating point. The best value is 0.0.
        But note the fact that bad predictions can lead to arbitarily large
        MAPE values, especially if some y_true values are very close to zero.
        Note that we return a large value instead of `inf` when y_true is zero.
    """
    epsilon = np.finfo(np.float64).eps
    mape = np.abs(x_pred - x_true) / np.maximum(np.abs(x_true), epsilon)
    output_errors = np.average(mape, axis=0)
    return np.average(output_errors)


@partial(np.vectorize, signature='(n)->()')
def mean_log_squared(signals):
    """
    compute mean log squared value of a signals in shape (m_1,m_2,..., n), where the mean squared
    is computed along the last axis.

    Parameters
    ----------
    signals: np.ndarray
        Input signals

    Returns
    -------
    output: np.ndarray
        The meansquared value.

    """
    return np.sum(np.log(np.square(signals))) / signals.shape[-1]


def rmlse(x_true, x_pred, negative_sign: bool = True):
    """
        compute root mean log squared value of a signals in shape (m_1,m_2,..., n), where the RMSE
        is computed along the last axis.
        Parameters
        ----------
        negative_sign:bool
            whether to multiply the mean_log_squared with minus one

        x_true: np.ndarray
            true array
        x_pred: np.ndarray
            pred array


        Returns
        -------
        out: np.ndarray
            The RMLSE
        """
    multiplier = 1
    if negative_sign:
        multiplier = -1

    error = x_true - x_pred
    return np.sqrt(multiplier * mean_log_squared(error))


def rmse(x_true, x_pred):
    """
    compute root mean squared value of a signals in shape (m_1,m_2,..., n), where the RMSE
    is computed along the last axis.
    Parameters
    ----------
    x_true: np.ndarray
        x_true x_pred
    x_pred: np.ndarray
        the x_pred
        

    Returns
    -------
    out: np.ndarray
        The RMSE
    """
    error = x_true - x_pred
    return np.sqrt(mean_squared(error))


def l_2_error(x_true, x_pred):
    """
    compute L_2 error value of a signals in shape (m_1,m_2,..., n), where the L2 error
    is computed along the last axis.
    Parameters
    ----------
    x_true: np.ndarray
        x_true x_pred
    x_pred: np.ndarray
        the x_pred


    Returns
    -------
    out: np.ndarray
        The L2_error
    """
    error = x_true - x_pred
    return np.linalg.norm(error, axis=-1)


def psnr(x_true, x_pred):
    """
    Compute PSNR value of a signals in shape (m_1,m_2,..., n), where the PSNR
    is computed along the last axis.
    Parameters
    ----------
    x_true: np.ndarray
        x_true x_pred
    x_pred: np.ndarray
        the x_pred


    Returns
    -------
    out: np.ndarray
        The L2_error
    """
    error = x_true - x_pred
    mse = mean_squared(error)
    psnr = 10 * np.log10(np.max(x_pred, axis=-1) ** 2 / mse)
    return psnr
