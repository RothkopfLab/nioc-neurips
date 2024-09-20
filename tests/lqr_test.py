"""Test for LQR solver"""
import jax
import jax.numpy as jnp
from jax.test_util import check_grads

from nioc.control.spec import LQGSpec
from nioc import lqr, glqr


def init_stable(key, state_dim):
    """Initialize a stable matrix with dimensions `state_dim`"""
    R = jax.random.normal(key, (state_dim, state_dim))
    A, _ = jnp.linalg.qr(R)
    return 0.5 * A


def init_glqr(key, state_dim: int, control_dim: int, horizon: int) -> LQGSpec:
    """Initialize a random GLQR spec."""

    # Initialize a random LQR spec
    Q = jnp.stack(horizon * (jnp.eye(state_dim),))
    q = 0.2 * jnp.stack(horizon * (jnp.ones(state_dim),))
    Qf = jnp.eye(state_dim)
    qf = 0.2 * jnp.ones((state_dim,))
    R = 1e-4 * jnp.stack(horizon * (jnp.eye(control_dim),))
    r = 1e-4 * jnp.stack(horizon * (jnp.ones(control_dim),))
    P = 1e-4 * jnp.stack(horizon * (jnp.ones((control_dim, state_dim)),))
    key, subkey = jax.random.split(key)
    A = jnp.stack(horizon * (init_stable(subkey, state_dim),))
    key, subkey = jax.random.split(key)
    B = jnp.stack(horizon * (jax.random.normal(subkey, (state_dim, control_dim)),))

    V = 1e-2 * jnp.stack(horizon * (jnp.eye(state_dim, 1),))

    Cx = 1e-2 * jnp.stack(horizon * (jnp.eye(state_dim)[:, None],))
    Cu = 1e-2 * jnp.stack(horizon * (jnp.eye(state_dim, control_dim)[:, None],))

    return LQGSpec(Q=Q, q=q, Qf=Qf, qf=qf, R=R, r=r, P=P, A=A, B=B, V=V, Cx=Cx, Cu=Cu,
                   F=jnp.zeros((horizon, state_dim, state_dim)), W=jnp.zeros((horizon, state_dim, 1)), D=None)


def test_lqr():
    # problem dimensions
    state_dim, control_dim, T = 3, 2, 5
    # random key
    key = jax.random.PRNGKey(42)

    key, subkey = jax.random.split(key)
    glqr_spec = init_glqr(key, state_dim=state_dim, control_dim=control_dim, horizon=T)

    for lqr_module, spec in zip((lqr, glqr,), (glqr_spec, glqr_spec,)):
        key, subkey = jax.random.split(key)
        x0 = jax.random.normal(subkey, (state_dim,))

        key, subkey = jax.random.split(key)

        # check that the gradients match
        def loss(spec):
            gains = lqr_module.backward(spec)

            X, U = lqr_module.simulate(subkey, spec, x0=x0, gains=gains)
            return (
                    0.5 * jnp.sum(X ** 2)
                    + 0.5 * jnp.sum(U ** 2)
                    + jnp.sum(spec.A ** 2)
            )

        # check along one random direction
        check_grads(loss, (spec,), 1, modes=("rev",))
