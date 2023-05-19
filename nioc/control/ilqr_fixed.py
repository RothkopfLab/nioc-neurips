from typing import Any, Tuple
import jax.numpy as jnp

from nioc import Env
from nioc.control import lqr, make_lqg_approx


def solve(p: Env,
          X: jnp.array,
          U: jnp.array,
          params: Any,
          Sigma0=None,
          lqr=lqr) -> Tuple[lqr.Gains, jnp.ndarray, jnp.ndarray]:
    lqrspec = make_lqg_approx(p, params)(X, U)
    gains = lqr.backward(lqrspec)

    return gains, X, U
