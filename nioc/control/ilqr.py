from typing import Any, Tuple
import jax.numpy as jnp
from jax import lax
import jaxopt

from nioc.control import lqr, make_lqg_approx
from nioc.control.lqr import Gains
from nioc import Env


def rollout(p: Env, X: jnp.array, U: jnp.array, gains: Gains, params: Any) -> Tuple[jnp.ndarray, jnp.ndarray]:
    x0 = X[0]

    def fwd(x, step):
        L, l, xbar, ubar = step

        u = L @ (x - xbar) + l + ubar

        x = p._dynamics(x, u, jnp.zeros(p.state_noise_shape), params)

        return x, (x, u)

    _, (X, U) = lax.scan(fwd, x0, (gains.L, gains.l, X[:-1], U))

    return jnp.vstack([x0, X]), U


def solve(p: Env,
          x0: jnp.array,
          U_init: jnp.array,
          params: Any,
          Sigma0=None,
          max_iter=10, tol=1e-6, lqr=lqr) -> Tuple[Gains, jnp.ndarray, jnp.ndarray]:
    T = U_init.shape[0]
    gains_init = Gains(L=jnp.zeros((T, p.action_shape[0], p.state_shape[0])),
                       l=jnp.zeros((T, p.action_shape[0])),
                       H=jnp.zeros((T, p.action_shape[0], p.action_shape[0])))

    X_init = jnp.vstack([x0, jnp.zeros((T, p.state_shape[0]))])

    def fixed_point_fun(carry, params):
        gains, X, U, cost = carry

        X, U = rollout(p, X, U, gains, params)
        cost = p.trajectory_cost(X, U, params)

        lqrspec = make_lqg_approx(p, params)(X, U)
        gains = lqr.backward(lqrspec)

        # TODO: try out jaxopt line search instead
        def cond_fn(eps):
            X_new, U_new = rollout(p, X, U, Gains(gains.L, eps * gains.l, gains.H), params)
            return (p.trajectory_cost(X_new, U_new, params) > cost) & (eps > tol)

        def body_fn(eps):
            return eps / 2.

        eps = lax.while_loop(cond_fn, body_fn, init_val=1.)

        gains = Gains(gains.L, eps * gains.l, gains.H)

        return (gains, X, U, cost)

    fpi = jaxopt.FixedPointIteration(fixed_point_fun=fixed_point_fun, maxiter=max_iter, implicit_diff=True)
    x_init = (gains_init, X_init, U_init, jnp.inf)
    res = fpi.run(x_init, params)
    gains, X, U, cost = res.params
    X, U = rollout(p, X, U, gains, params)

    # (gains, X, U, _), costs = lax.scan(ilqr_step, (gains_init, X, U, jnp.inf), jnp.arange(max_iter))

    return gains, X, U
