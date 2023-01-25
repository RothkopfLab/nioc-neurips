from typing import Callable
import jax.numpy as jnp
from jax.numpy.linalg import cholesky, inv

from nioc.control.lqr import Gains


def create_lqr_policy(gains: Gains, xbar: jnp.ndarray, ubar: jnp.ndarray) -> Callable:
    return lambda t, x, noise=None: gains.L[t] @ (x - xbar[t]) + gains.l[t] + ubar[t]


def create_lqg_policy(gains: Gains, xbar: jnp.ndarray, ubar: jnp.ndarray) -> Callable:
    return lambda t, b, noise=None: gains.L[t] @ (b[0] - xbar[t]) + gains.l[t] + ubar[t]


def create_maxent_lqr_policy(gains: Gains, xbar: jnp.ndarray, ubar: jnp.ndarray, inv_temp=1e-6) -> Callable:
    return lambda t, x, noise=jnp.zeros(gains.D.shape[1]): gains.L[t] @ (x - xbar[t]) + gains.l[t] + ubar[t] \
                                                           + jnp.sqrt(inv_temp) * cholesky(inv(gains.D[t])) @ noise


def create_maxent_lqg_policy(gains: Gains, xbar: jnp.ndarray, ubar: jnp.ndarray, inv_temp=1e-6) -> Callable:
    return lambda t, b, noise=jnp.zeros(gains.D.shape[1]): gains.L[t] @ (b[0] - xbar[t]) + gains.l[t] + ubar[t] \
                                                           + jnp.sqrt(inv_temp) * cholesky(inv(gains.D[t])) @ noise


def create_zero_policy(control_dim):
    return lambda t, b, noise=None: jnp.zeros((control_dim,))
