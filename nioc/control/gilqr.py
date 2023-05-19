from typing import Any, Tuple
from jax import numpy as jnp

from nioc import Env
from nioc.control import glqr, ilqr, lqr


def solve(p: Env,
          x0: jnp.array,
          U_init: jnp.array,
          params: Any,
          Sigma0=None,
          max_iter=10, tol=1e-6) -> Tuple[lqr.Gains, jnp.ndarray, jnp.ndarray]:
    return ilqr.solve(p, x0, U_init=U_init, params=params, max_iter=max_iter, tol=tol, lqr=glqr)
