from typing import NamedTuple

import jax.numpy as jnp


class Belief(NamedTuple):
    xhat: jnp.ndarray
    Sigma: jnp.ndarray


def stack_belief(b: Belief) -> jnp.ndarray:
    xhat, Sigma = b
    S = jnp.linalg.cholesky(Sigma)
    return jnp.hstack((xhat, S[jnp.tril_indices_from(S)]))
