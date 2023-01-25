from typing import NamedTuple
from jax import lax, numpy as jnp


class KFSpec(NamedTuple):
    """KF specification"""

    A: jnp.ndarray
    H: jnp.ndarray
    V: jnp.ndarray
    W: jnp.ndarray


def forward(spec: KFSpec, Sigma0: jnp.ndarray) -> jnp.ndarray:
    def loop(P, step):
        A, H, V, W = step

        G = H @ P @ H.T + W @ W.T
        K = A @ P @ H.T @ jnp.linalg.inv(G)
        P = V @ V.T + (A - K @ H) @ P @ A.T

        return P, K

    _, K = lax.scan(loop, Sigma0,
                    (spec.A, spec.H, spec.V, spec.W))

    return K
