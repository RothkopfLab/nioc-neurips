from typing import Any, Tuple
import jax.numpy as jnp

from nioc import Env
from nioc.control import lqr


def solve(p: Env,
          X: jnp.array,
          U: jnp.array,
          params: Any,
          lqr=lqr) -> Tuple[lqr.Gains, jnp.ndarray, jnp.ndarray]:
    lqrspec = lqr.make_approx(p, params)(X, U)
    gains = lqr.backward(lqrspec)

    return gains, X, U
