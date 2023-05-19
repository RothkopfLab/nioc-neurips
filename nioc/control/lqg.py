from jax import lax, random, numpy as jnp
from typing import Tuple

from nioc.control import lqr, LQGSpec
from nioc.belief import kf


def solve(spec: LQGSpec, Sigma0: jnp.ndarray, eps: float = 1e-8) -> Tuple[jnp.ndarray, lqr.Gains]:
    K = kf.forward(spec=spec, Sigma0=Sigma0)
    control_gains = lqr.backward(spec=lqr.LQRSpec(Q=spec.Q, q=spec.q, Qf=spec.Qf, qf=spec.qf,
                                                  P=spec.P, R=spec.R, r=spec.r,
                                                  A=spec.A, B=spec.B), eps=eps)

    return K, control_gains


def simulate(key: random.PRNGKey,
             spec: LQGSpec, x0: jnp.ndarray,
             filter: jnp.ndarray = None, gains: lqr.Gains = None,
             xhat0: jnp.ndarray = None, Sigma0: jnp.ndarray = None,
             eps: float = 1e-8) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """

    Args:
        key: jax.random.PRNGKey
        lqg: LQGSpec
        K: Kalman gain
        L: LQR control law
        x0: array of shape (xdim,) initial state
        xhat0: array of shape (xdim,) intial belief

    Returns:
        x: array of shape (T, xdim) trajectory with x[0] = p0
    """
    xhat0 = xhat0 if xhat0 is not None else x0
    Sigma0 = Sigma0 if Sigma0 is not None else spec.V[0] @ spec.V[0].T

    if filter is None or gains is None:
        filter, gains = solve(spec, eps=eps, Sigma0=Sigma0)

    T = spec.A.shape[0]
    xdim = spec.A.shape[1]
    ydim = spec.F.shape[1]

    key_system_noise, key_obs_noise = random.split(key, 2)
    state_noise = random.multivariate_normal(key_system_noise, jnp.zeros((xdim,)), jnp.eye(xdim),
                                             (T,))
    obs_noise = random.multivariate_normal(key_obs_noise, jnp.zeros((ydim,)), jnp.eye(ydim),
                                           (T,))

    def sim_iter(carry, step):
        x, xhat = carry

        A, B, F, V, W, K, gain, eps_x, eps_y = step

        y = F @ x + W @ eps_y

        xpred = A @ xhat + B @ (gain.L @ xhat + gain.l)
        xhat = xpred + K @ (y - F @ xhat)

        u = gain.L @ xhat + gain.l
        x = A @ x + B @ u + V @ eps_x

        return (x, xhat), (x, u)

    _, (X, U) = lax.scan(sim_iter, (x0, xhat0), (spec.A, spec.B, spec.F, spec.V, spec.W,
                                                 filter, gains, state_noise, obs_noise))

    return jnp.vstack([x0, X]), U
