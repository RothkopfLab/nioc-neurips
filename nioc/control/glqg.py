from jax import vmap, lax, random, jacobian, numpy as jnp
from typing import NamedTuple, Tuple, Any, Callable

from nioc.envs import Env
from nioc.control.lqr import Gains
from nioc.control import glqr
from nioc.belief import kf


class GLQGSpec(NamedTuple):
    Q: jnp.ndarray
    q: jnp.ndarray
    Qf: jnp.array
    qf: jnp.array
    P: jnp.ndarray
    R: jnp.ndarray
    r: jnp.ndarray
    A: jnp.ndarray
    B: jnp.ndarray
    Cx: jnp.ndarray
    Cu: jnp.ndarray
    V: jnp.ndarray
    H: jnp.ndarray
    W: jnp.ndarray


def make_approx(p: Env, params: Any) -> Callable:
    lqr_approx = glqr.make_approx(p, params)

    @vmap
    def approx_timestep(x, u):
        # approximately linear observation
        H = jacobian(p._observation, argnums=0)(x, jnp.zeros(p.obs_noise_shape), params)

        # observation noise
        W = jacobian(p._observation, argnums=1)(x, jnp.zeros(p.obs_noise_shape), params)

        return H, W

    def approx(X, U):
        assert X.shape[0] == (U.shape[0] + 1)
        H, W = approx_timestep(X[:-1], U)

        return GLQGSpec(*lqr_approx(X, U), H=H, W=W)

    return approx


def solve(spec: GLQGSpec, Sigma0: jnp.ndarray, eps: float = 1e-8) -> Tuple[jnp.ndarray, Gains]:
    # TODO: right now, this does not do the iterative procedure by Todorov (2005)
    #  and instead only computes the Kalman filter and the gLQR control gains
    #  but this should be way more efficient in the inverse optimal control procedure
    K = kf.forward(kf.KFSpec(A=spec.A, H=spec.H, V=spec.V, W=spec.W), Sigma0=Sigma0)
    control_gains = glqr.backward(glqr.GLQRSpec(Q=spec.Q, q=spec.q, Qf=spec.Qf, qf=spec.qf,
                                                P=spec.P, R=spec.R, r=spec.r,
                                                A=spec.A, B=spec.B, V=spec.V, Cu=spec.Cu, Cx=spec.Cx), eps=eps)

    return K, control_gains


def simulate(key: random.PRNGKey,
             spec: GLQGSpec, x0: jnp.ndarray,
             filter: jnp.ndarray = None, gains: Gains = None,
             xhat0: jnp.ndarray = None, eps: float = 1e-8) -> Tuple[jnp.ndarray, jnp.ndarray]:
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

    if filter is None or gains is None:
        filter, gains = solve(spec, eps=eps)

    T = spec.A.shape[0]

    key1, key2, key3, key4 = random.split(key, 4)
    noise_x = random.normal(key1, (T, spec.V.shape[-1],))
    noise_y = random.normal(key2, (T, spec.W.shape[-1],))
    noise_Cu = random.normal(key3, (T, spec.Cu.shape[-2],))
    noise_Cx = random.normal(key4, (T, spec.Cx.shape[-2],))

    def sim_iter(carry, step):
        x, xhat = carry

        A, B, H, V, W, Cx, Cu, K, gain, eps_x, eps_y, eps_Cx, eps_Cu = step

        y = H @ x + W @ eps_y

        xpred = A @ xhat + B @ (gain.L @ xhat + gain.l)
        xhat = xpred + K @ (y - H @ xpred)

        u = gain.L @ xhat + gain.l
        x = A @ x + B @ u + V @ eps_x + Cx @ x @ eps_Cx + Cu @ u @ eps_Cu

        return (x, xhat), (x, u)

    _, (X, U) = lax.scan(sim_iter, (x0, xhat0), (spec.A, spec.B, spec.H, spec.V, spec.W, spec.Cx, spec.Cu,
                                                 filter, gains, noise_x, noise_y, noise_Cx, noise_Cu))

    return jnp.vstack([x0, X]), U
