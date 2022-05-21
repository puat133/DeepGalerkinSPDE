from typing import Callable, Tuple

import jax.numpy as np
from jax import jacfwd, jacrev, jit, jvp, lax, vjp
from jax.scipy.sparse.linalg import cg

from Optimizers.levenberg_marquardt import cond_1_false_l_m_constant, cond_1_l_m_constant, \
    cond_1_true_l_m_constant, cond_r_greater_zero
from utils import squared_norm


def adaptive_hessian_free(fun: Callable,
                          initial_state: np.ndarray,
                          damping: float = 0.5,
                          cg_max_iter: int = 50,
                          n_steps: int = 50,
                          use_gnm: bool = False,
                          epsilon: float = np.finfo(np.float32).eps,
                          # beta_1:float,
                          beta_2: float = 0.999,
                          cg_epsilon: float = np.finfo(np.float32).eps,
                          cg_sigma: float = 30,
                          cg_el_smoothness: float = 100) -> Tuple:
    # Set beta_1
    beta_1 = cg_epsilon / (cg_epsilon + 2 * cg_sigma)

    @jit
    def _jvp_fun(x, v):
        """
        Calculate jacobian vector product

        Parameters
        ----------
        x : np.ndarray
            primal
        v : np.ndarray
            tangent

        Returns
        -------
        tangents_out: np.ndarray
            tangent output
        """

        return jvp(fun, (x,), (v,))[1]

    @jit
    def _vjp_fun(x):
        """
        perform vector jacobian product of fun at x. Just a wrapper from jax vjp
        Parameters
        ----------
        x : np.ndarray
            primal
        Returns
        -------
        cotangent_out: fun
            Cotangent output
        """

        return vjp(fun, x)[1]

    @jit
    def _hvp(x, v):
        """
        Calculate the Hessian vector product at x
        Parameters
        ----------
        x : np.ndarray
            primal
        v : np.ndarray
            tangent

        Returns
        -------
        tangents_out: np.ndarray
            tangent output
        """
        # Hessian vector product
        return jacrev(lambda x_: _jvp_fun(x_, v))(x)

    @jit
    def _gvp(x, v):
        """
        Calculate J^TJ v, where J is the Jacobian of fun

         Parameters
        ----------
        x : np.ndarray
            primal
        v : np.ndarray
            tangent

        Returns
        -------
        tangents_out: np.ndarray
            tangent output
        """
        jac_v = _jvp_fun(x, v)
        jac_trans_w = _vjp_fun(x)
        return jac_trans_w(jac_v)[0]

    @jit
    def cond_r_greater_zero_true(operand):
        state, time, momentum, velocity, delta, b = operand
        state, momentum, velocity, time = __adapt(state, time, momentum, velocity,
                                                  beta_1, beta_2, delta, -b, cg_sigma,
                                                  cg_el_smoothness, epsilon)
        return state, momentum, velocity, time

    @jit
    def cond_r_greater_zero_false(operand):
        state, time, momentum, velocity, delta, b = operand
        return state, momentum, velocity, time

    if use_gnm:
        actual_loss = lambda x: 0.5 * squared_norm(fun(x))
        jac = jit(jacfwd(actual_loss))  # jacobian of the function
        mvp = _gvp
    else:
        actual_loss = fun
        jac = jit(jacfwd(fun))  # jacobian of the function
        mvp = _hvp

    dummy_inputs = np.empty(n_steps)  # dummy input

    @jit
    def __body_scan(carry_, inputs):
        state, momentum, second_moment, time, lamb = carry_
        b = -jac(state)
        loss_now = actual_loss(state)
        a_func = lambda v: mvp(state, v) + lamb * v

        # delta to be obtained by using conjugate gradient
        # todo: implement preconditioning here.
        delta = cg(a_func, b, maxiter=cg_max_iter, tol=cg_epsilon)[0]  # eq. 3.1.6

        # Levenberg Marquardt heuristics
        loss_next = actual_loss(state + delta)
        delta_f = loss_now - loss_next  # eq. 5.1.3
        delta_g = - (np.dot(-b, delta) + 0.5 * delta @ a_func(delta))  # eq. 5.1.4
        r = delta_f / delta_g  # eq. 5.1.5
        # compute next nu value based on r value, if else see eq. 5.2.7
        cond_operand = (r, lamb)
        lamb_next = lax.cond(cond_1_l_m_constant(cond_operand), cond_1_true_l_m_constant, cond_1_false_l_m_constant,
                             cond_operand)

        # decide state_next based on the value of r, if else see eq. 5.2.7
        operand = (state, time, momentum, second_moment, delta, b)
        state_next, momentum_next, velocity_next, time_next = lax.cond(cond_r_greater_zero(r), cond_r_greater_zero_true,
                                                                       cond_r_greater_zero_false, operand)

        new_carry = (state_next, momentum_next, velocity_next, time_next, lamb_next)
        return new_carry, (state, lamb, loss_now)

    initial_momentum = np.zeros_like(initial_state)
    initial_second_moment = np.zeros_like(initial_state)
    initial_time = 0

    carry = (initial_state, initial_momentum, initial_second_moment, initial_time, damping)
    carry, (state_hist, nu_hist, value_hist) = lax.scan(__body_scan, carry, dummy_inputs)
    return state_hist, nu_hist, value_hist


def __adapt(state: np.ndarray,
            time: int,
            momentum: np.ndarray,
            second_moment: np.ndarray,
            beta_1: float,
            beta_2: float,
            delta: float,
            jac_vector: np.ndarray,
            sigma: float,
            cg_L: float,
            epsilon: float) -> Tuple:
    """
    This is the adaptation algorithm from Algorithm 1 from
    "Adaptive Hessian-free optimization for training neural networks" 2020.
    See Also
    "ADAM: A METHOD FOR STOCHASTIC OPTIMIZATION"
    mu is replaced by epsilon
    Parameters
    ----------
    state : np.ndarray
        the n-dimensional variable of the loss function
    time: int
        time index
    momentum : np.ndarray
        the momentum
    second_moment : np.ndarray
        the second moment vector
    beta_1 : float
    beta_2 : float
    delta : np.ndarray
        result of the conjugate gradient from the main algorithm
    jac_vector : np.ndarray
        jacobian vector of the loss function at state
    sigma : float
    cg_L : float
    epsilon : float

    Returns
    -------
    output : Tuple
        The new state, momentum, velocity, time
    """
    time += 1
    momentum = beta_1 * momentum + (1 - beta_1) * delta
    second_moment = beta_2 * second_moment + (1 - beta_2) * delta * delta
    beta_1_t = np.power(beta_1, time)
    beta_2_t = np.power(beta_2, time)
    one_min_beta_1_t = (1 - beta_1_t)
    momentum_hat = momentum / one_min_beta_1_t
    second_moment_hat = second_moment / (1 - beta_2_t)
    norm_jac = np.linalg.norm(jac_vector)

    alpha = 4 * (norm_jac * norm_jac * (1 - beta_1) - norm_jac * (beta_1 - beta_1_t) * sigma) / (
            3 * cg_L * one_min_beta_1_t * one_min_beta_1_t * sigma)
    state = state + alpha * momentum_hat / np.sqrt(second_moment_hat + epsilon)
    return state, momentum, second_moment, time
