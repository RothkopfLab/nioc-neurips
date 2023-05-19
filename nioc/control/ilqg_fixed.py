from typing import Any, Tuple
import jax.numpy as jnp

from nioc import Env
from nioc.control import lqg, make_lqg_approx
from nioc.control.lqr import Gains


def solve(p: Env,
          X: jnp.array,
          U: jnp.array,
          Sigma0: jnp.ndarray,
          params: Any, lqg_module=lqg) -> Tuple[jnp.ndarray, Gains, jnp.ndarray, jnp.ndarray]:
    lqgspec = make_lqg_approx(p, params)(X, U)
    K, gains = lqg_module.solve(lqgspec, Sigma0=Sigma0)

    return K, gains, X, U
