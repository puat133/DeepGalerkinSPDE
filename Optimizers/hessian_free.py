from typing import Callable, Tuple

import jax.numpy as np
from jax import jacfwd, jacrev, jit, jvp, lax, vjp, checkpoint
from jax.scipy.sparse.linalg import cg

from Optimizers.levenberg_marquardt import cond_1_false_l_m_constant, cond_1_l_m_constant, cond_1_true_l_m_constant, \
                                           cond_r_greater_zero, cond_r_greater_zero_false, cond_r_greater_zero_true
from utils import squared_norm


def hessian_free(fun: Callable,
                 initial_state: np.ndarray,
                 damping: float = 0.5,
                 cg_max_iter: int = 50,
                 n_steps: int = 50,
                 use_gnm: bool = True,
                 learning_rate: float = 1.) -> Tuple:
    """
    Optimize a function based on the Hessian free optimization method.

    ..
    Parameters
    ----------
    fun: Callable
        The function to be optimized.
    initial_state: np.ndarray
        Initial optimization state.
    damping: float, optional
        Initial value of the Tikhonov damping coefficient. (default: 0.5)
    cg_max_iter: int, optional
        Maximum number of Conjugate-Gradient iterations. (default: 50)
    n_steps: int, optional
        Iteration times. (default: 50)
    use_gnm: bool, optional
        Use the generalized Gauss-Newton matrix:probably solves the indefiniteness of the Hessian. (default: False)
    learning_rate: float, optional
        Learning rate. (default: 1)
    Returns
    -------
    out: Tuple

    Notes
    -----
    Motivated by https://github.com/fmeirinhos/pytorch-hessianfree/blob/master/hessianfree.py

    References
    ----------
    .. [1] Training Deep and Recurrent Networks with Hessian-Free Optimization:
        https://doi.org/10.1007/978-3-642-35289-8_27


    """

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

    if use_gnm:
        actual_loss = lambda x: 0.5*squared_norm(fun(x))
        jac = jit(jacfwd(actual_loss))  # jacobian of the function
        mvp = _gvp
    else:
        actual_loss = fun
        jac = jit(jacfwd(fun))  # jacobian of the function
        mvp = _hvp

    dummy_inputs = np.empty(n_steps)  # dummy input

    @jit
    @checkpoint
    def __body_scan(carry_, inputs):
        state, lamb = carry_
        b = -jac(state)
        loss_now = actual_loss(state)
        a_func = lambda v: mvp(state, v) + lamb * v

        # delta to be obtained by using conjugate gradient
        # todo: implement preconditioning here.
        delta = learning_rate * cg(a_func, b, maxiter=cg_max_iter)[0]  # eq. 3.1.6

        # Levenberg Marquardt heuristics by Flecther instead of what is mentioned in HF paper
        loss_next = actual_loss(state + delta)
        delta_f = loss_now - loss_next  # eq. 5.1.3
        delta_g = - (jvp(actual_loss, (state,), (delta,))[1] + 0.5 * delta @ a_func(delta))  # eq. 5.1.4
        r = delta_f / delta_g  # eq. 5.1.5
        # compute next nu value based on r value, if else see eq. 5.2.7
        cond_operand = (r, lamb)
        lamb_next = lax.cond(cond_1_l_m_constant(cond_operand),
                             cond_1_true_l_m_constant,
                             cond_1_false_l_m_constant,
                             cond_operand)

        # decide state_next based on the value of r, if else see eq. 5.2.7
        operand = (state, delta)
        state_next = lax.cond(cond_r_greater_zero(r), cond_r_greater_zero_true,
                              cond_r_greater_zero_false, operand)

        new_carry = (state_next, lamb_next)
        return new_carry, (state, lamb, loss_now)

    carry = (initial_state, damping)
    carry, (state_hist, nu_hist, value_hist) = lax.scan(__body_scan, carry, dummy_inputs)

    state_last, lamb_last = carry
    return state_hist, nu_hist, value_hist
