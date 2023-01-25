from typing import Callable, Tuple, NamedTuple, Any
import jax.numpy as jnp
from jax import vmap, jacfwd, grad, jacobian
from jax import lax

from nioc import Env


class Gains(NamedTuple):
    """LQR control gains"""

    L: jnp.ndarray
    l: jnp.ndarray
    D: jnp.ndarray = None


class LQRSpec(NamedTuple):
    """LQR specification"""

    Q: jnp.ndarray
    q: jnp.ndarray
    Qf: jnp.array
    qf: jnp.array
    P: jnp.ndarray
    R: jnp.ndarray
    r: jnp.ndarray
    A: jnp.ndarray
    B: jnp.ndarray


def make_approx(p: Env, params: Any) -> Callable:
    @vmap
    def approx_timestep(x, u):
        # quadratic approximation of cost function
        P = jacfwd(grad(p._cost, argnums=1), argnums=0)(x, u, params)
        Q = jacfwd(grad(p._cost, argnums=0), argnums=0)(x, u, params)
        R = jacfwd(grad(p._cost, argnums=1), argnums=1)(x, u, params)
        q, r = grad(p._cost, argnums=(0, 1))(x, u, params)

        # linear approximation of dynamics
        A, B = jacobian(p._dynamics, argnums=(0, 1))(x, u, jnp.zeros(p.state_noise_shape), params)
        return Q, q, P, R, r, A, B

    def approx(X, U):
        assert X.shape[0] == (U.shape[0] + 1)
        Q, q, P, R, r, A, B = approx_timestep(X[:-1], U)

        # quadratic approximation of the final time step costs
        Qf = jacfwd(grad(p._final_cost, argnums=0), argnums=0)(X[-1], params)
        qf = grad(p._final_cost, argnums=0)(X[-1], params)

        return LQRSpec(Q=Q, q=q, Qf=Qf, qf=qf, P=P, R=R, r=r, A=A, B=B)

    return approx


def backward(spec: LQRSpec, eps: float = 1e-8) -> Gains:
    def loop(carry, step):
        S, s = carry

        Q, q, P, R, r, A, B = step

        H = R + B.T @ S @ B
        G = P + B.T @ S @ A
        g = r + B.T @ s

        # Deal with negative eigenvals of H, see section 5.4.1 of Li's PhD thesis
        evals, _ = jnp.linalg.eigh(H)
        Ht = H + jnp.maximum(0., eps - evals[0]) * jnp.eye(H.shape[0])

        L = -jnp.linalg.solve(Ht, G)
        l = -jnp.linalg.solve(Ht, g)

        S = Q + A.T @ S @ A + L.T @ H @ L + L.T @ G + G.T @ L
        s = q + A.T @ s + G.T @ l + L.T @ H @ l + L.T @ g

        return (S, s), (L, l, Ht)

    _, (L, l, D) = lax.scan(loop, (spec.Qf, spec.qf),
                         (spec.Q, spec.q, spec.P, spec.R, spec.r, spec.A, spec.B),
                         reverse=True)

    return Gains(L=L, l=l, D=D)


def simulate(key,
             spec: LQRSpec, x0: jnp.ndarray, gains: Gains = None, eps: float = 1e-8) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Simulates noiseless forward dynamics"""

    if gains is None:
        gains = backward(spec, eps=eps)

    def dyn(x, inps):
        A, B, gain = inps
        u = gain.L @ x + gain.l
        nx = A @ x + B @ u
        return nx, (nx, u)

    _, (X, U) = lax.scan(dyn, x0, (spec.A, spec.B, gains))
    return jnp.vstack([x0, X]), U
