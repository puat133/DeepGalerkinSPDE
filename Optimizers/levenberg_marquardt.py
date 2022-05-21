import jax.numpy as np
from jax import jacfwd, jacrev, lax, jit, jvp
from jax.scipy.linalg import cho_solve, cholesky, cho_factor
from typing import Callable, Tuple


@jit
def cond_2_l_m_constant(operand: Tuple):
    r_test, nu_test = operand
    return r_test > 0.75


@jit
def cond_2_true_l_m_constant(operand: Tuple):
    r_test, nu_test = operand
    # return 0.5 * nu_test
    return nu_test * (2 / 3)


@jit
def cond_2_false_l_m_constant(operand: Tuple):
    r_test, nu_test = operand
    return nu_test


@jit
def cond_1_l_m_constant(operand: Tuple):
    r_test, nu_test = operand
    return r_test < 0.25


@jit
def cond_1_true_l_m_constant(operand: Tuple):
    r_test, nu_test = operand
    # return 4 * nu_test
    return nu_test * (3 / 2)


@jit
def cond_1_false_l_m_constant(operand: Tuple):
    r_test, nu_test = operand
    return lax.cond(cond_2_l_m_constant(operand), cond_2_true_l_m_constant, cond_2_false_l_m_constant,
                    (r_test, nu_test))


@jit
def __body_while_inner(nu_now):
    return 4 * nu_now


@jit
def cond_r_greater_zero(r_test):
    return r_test > 0.


@jit
def cond_r_greater_zero_true(operand):
    state_now, delta = operand
    return state_now + delta


@jit
def cond_r_greater_zero_false(operand):
    state_now, delta = operand
    return state_now


# @partial(jit, static_argnums=(0,))
def levenberg_marquardt(fun: Callable, initial_state: np.ndarray, nu_init: float,
                        n_steps: int, learning_rate: float = 1.) -> Tuple:
    """
    This function implements simple Levenberg - Marquardt optimization procedure.

    Parameters
    ----------
    fun: Callable
        The function to be optimized

    initial_state: np.ndarray
        Initial optimization state.

    nu_init: float
        Initial Levenberg-Marquardt constant.

    n_steps: int
        Number of steps.

    learning_rate:float
        Scaling factor to the step scaling.


    Returns
    -------
    out: Tuple
        History of states, values, and nu values.


    References
    ----------

    .. [1] Practical methods of optimization - Wiley 2000 Fletcher.
    """

    hessian = jit(jacfwd(jacrev(fun)))
    jac = jit(jacfwd(fun))
    identity_mat = np.eye(initial_state.shape[0])

    carry = (initial_state, nu_init)
    dummy_inputs = np.empty(n_steps)  # dummy input

    @jit
    def __body_scan(carry, inputs):
        state, nu = carry
        hess_now = hessian(state)

        fun_now = fun(state)

        # multiply lambda to 4 if G+lambda I is not positive definite
        condition = lambda nu_test: np.any(np.isnan(cholesky(hess_now + nu_test * identity_mat)))

        nu_passed = lax.while_loop(condition, __body_while_inner, nu)

        factors = cho_factor(hess_now + nu_passed * identity_mat)
        delta = learning_rate * cho_solve(factors, -jac(state))  # eq. 3.1.6

        fun_next = fun(state + delta)
        Delta_f = fun_now - fun_next  # eq. 5.1.3

        Delta_g = - (jvp(fun, (state,), (delta,))[1] + 0.5 * delta @ (hess_now @ delta))  # eq. 5.1.4 & 3.1.1

        r = Delta_f / Delta_g  # eq. 5.1.5

        # compute next nu value based on r value, if else see eq. 5.2.7
        cond_operand = (r, nu_passed)
        nu_next = lax.cond(cond_1_l_m_constant(cond_operand), cond_1_true_l_m_constant, cond_1_false_l_m_constant,
                           cond_operand)

        # decide state_next based on the value of r, if else see eq. 5.2.7
        operand = (state, delta)
        state_next = lax.cond(cond_r_greater_zero(r), cond_r_greater_zero_true,
                              cond_r_greater_zero_false, operand)

        new_carry = (state_next, nu_next)
        return new_carry, (state, nu, fun_now)

    carry, (state_hist, nu_hist, value_hist) = lax.scan(__body_scan, carry, dummy_inputs)

    state_last, nu_last = carry
    return state_hist, nu_hist, value_hist
