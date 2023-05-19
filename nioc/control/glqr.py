from jax import random, lax, numpy as jnp
from typing import Tuple

from nioc.control import lqr, LQGSpec
from nioc.utils import quadratic_form, bilinear_form


def backward(spec: LQGSpec, eps: float = 1e-8) -> lqr.Gains:
    def loop(carry, step):
        S, s = carry

        Q, q, P, R, r, A, B, V, Cx, Cu = step

        H = R + B.T @ S @ B + quadratic_form(Cu, S).sum(axis=0)
        G = P + B.T @ S @ A + bilinear_form(Cu, S, Cx).sum(axis=0)
        g = r + B.T @ s + bilinear_form(Cu, S, V).sum(axis=0)

        # Deal with negative eigenvals of H, see section 5.4.1 of Li's PhD thesis
        evals, _ = jnp.linalg.eigh(H)
        Ht = H + jnp.maximum(0., eps - evals[0]) * jnp.eye(H.shape[0])

        L = -jnp.linalg.solve(Ht, G)
        l = -jnp.linalg.solve(Ht, g)

        Sn = Q + A.T @ S @ A + L.T @ H @ L + L.T @ G + G.T @ L + quadratic_form(Cx, S).sum(axis=0)
        sn = q + A.T @ s + G.T @ l + L.T @ H @ l + L.T @ g + bilinear_form(Cx, S, V).sum(axis=0)

        return (Sn, sn), (L, l, Ht)

    _, (L, l, H) = lax.scan(loop, (spec.Qf, spec.qf),
                            (spec.Q, spec.q, spec.P, spec.R, spec.r, spec.A, spec.B, spec.V, spec.Cx, spec.Cu),
                            reverse=True)

    return lqr.Gains(L=L, l=l, H=H)


def simulate(key: random.PRNGKey,
             spec: LQGSpec, x0: jnp.ndarray,
             gains: lqr.Gains = None, eps: float = 1e-8) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Simulates noiseless forward dynamics"""

    T = spec.A.shape[0]

    if gains is None:
        gains = backward(spec, eps=eps)

    key1, key2, key3 = random.split(key, 3)
    noise_x = random.normal(key1, (T, spec.V.shape[-1],))
    noise_Cu = random.normal(key2, (T, spec.Cu.shape[-2],))
    noise_Cx = random.normal(key3, (T, spec.Cx.shape[-2],))

    def dyn(x, inps):
        A, B, V, Cx, Cu, gain, eps_x, eps_Cx, eps_Cu = inps
        u = gain.L @ x + gain.l
        nx = A @ x + B @ u + V @ eps_x + Cx @ x @ eps_Cx + Cu @ u @ eps_Cu
        return nx, (nx, u)

    _, (X, U) = lax.scan(dyn, x0, (spec.A, spec.B, spec.V, spec.Cx, spec.Cu, gains, noise_x, noise_Cx, noise_Cu))
    return jnp.vstack([x0, X]), U
