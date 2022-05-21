import pytest
import jax.numpy as np
import os
from jax import jit
from Optimizers import levenberg_marquardt, hessian_free, adaptive_hessian_free

os.environ["CUDA_VISIBLE_DEVICES"] = ""


@jit
def fun_test_1(x: np.ndarray):
    """
    The test function in the equation 3.1.3 from Fletcher book.
    .:math: x[0]*x[0]*x[0]*x[0] + x[0]*x[1] + (1+x[1])*(1+x[1])


    Parameters
    ----------
    x: numpy.ndarray
        the `x`.

    Returns
    -------
    out:
        output of the function.
    """
    return x[0] * x[0] * x[0] * x[0] + x[0] * x[1] + (1 + x[1]) * (1 + x[1])


@jit
def fun_test_2(x: np.ndarray):
    """
    The Rosenborg function
    .:math: (a-x[0])^2 + b(x[1]-x[0]^2)^2

    where `a = 1` `b = 100`.

    Parameters
    ----------
    x: numpy.ndarray
        the `x`.

    Returns
    -------
    out:
        output of the function.
    """
    return (1. - x[0]) ** 2 + 100 * (x[1] - x[0] * x[0]) ** 2


@jit
def fun_test_3(x: np.ndarray):
    """
    A function f, such that :math: 1/2 z^2 \circ f is equal to the Rosenborg function
    .:math: (a-x[0])^2 + b(x[1]-x[0]^2)^2

    where `a = 1` `b = 100`.

    Parameters
    ----------
    x: numpy.ndarray
        the `x`.

    Returns
    -------
    out:
        output of the function.
    """
    return np.sqrt(2) * np.array([1 - x[0], 10 * (x[1] - x[0] * x[0])])


@pytest.mark.parametrize("fun_test", [fun_test_1, fun_test_2])
@pytest.mark.parametrize("x_init", [np.array([0.75, -1.25])])
@pytest.mark.parametrize("step_scale", [0.1])
@pytest.mark.parametrize("n_steps", [1000])
def test_levenberg_marquardt(fun_test, x_init, n_steps: int, step_scale: float):
    nu_init = 1.
    state_hist, nu_hist, value_hist = levenberg_marquardt(fun_test, x_init, nu_init, n_steps, step_scale)
    assert state_hist.shape[0] == n_steps
    assert nu_hist.shape[0] == n_steps
    assert value_hist.shape[0] == n_steps


@pytest.mark.parametrize("fun_test", [fun_test_1, fun_test_2])
@pytest.mark.parametrize("x_init", [np.array([0.75, -1.25])])
@pytest.mark.parametrize("learning_rate", [1.])
@pytest.mark.parametrize("n_steps", [1000])
def test_hessian_free(fun_test, x_init, n_steps: int, learning_rate: float):
    nu_init = 1.
    state_hist, nu_hist, value_hist = hessian_free(fun_test, x_init, n_steps=n_steps,
                                                   use_gnm=False, cg_max_iter=40,
                                                   learning_rate=learning_rate)
    assert state_hist.shape[0] == n_steps
    assert nu_hist.shape[0] == n_steps
    assert value_hist.shape[0] == n_steps


@pytest.mark.parametrize("fun_test", [fun_test_3])
@pytest.mark.parametrize("x_init", [np.array([0.75, -1.25])])
@pytest.mark.parametrize("learning_rate", [1.])
@pytest.mark.parametrize("n_steps", [1000])
def test_hessian_free_with_gnm(fun_test, x_init, n_steps: int, learning_rate: float):
    nu_init = 1.
    state_hist, nu_hist, value_hist = hessian_free(fun_test, x_init, n_steps=n_steps,
                                                   use_gnm=True, cg_max_iter=10,
                                                   learning_rate=learning_rate)
    assert state_hist.shape[0] == n_steps
    assert nu_hist.shape[0] == n_steps
    assert value_hist.shape[0] == n_steps


@pytest.mark.parametrize("fun_test", [fun_test_2])
@pytest.mark.parametrize("x_init", [np.array([0.75, -1.25])])
@pytest.mark.parametrize("n_steps", [1000])
def test_adaptive_hessian_free(fun_test, x_init, n_steps: int):
    state_hist, nu_hist, value_hist = adaptive_hessian_free(fun_test,
                                                            x_init,
                                                            damping=1.,
                                                            beta_2=1-1e-2,
                                                            # cg_sigma=1e-5,
                                                            # cg_epsilon=1e-8,
                                                            n_steps=n_steps,
                                                            use_gnm=False,
                                                            cg_max_iter=100)
    assert state_hist.shape[0] == n_steps
    assert nu_hist.shape[0] == n_steps
    assert value_hist.shape[0] == n_steps


