from jax import vmap, numpy as jnp
import jax.random as jr
from typing import NamedTuple
from jax.test_util import check_grads

from nioc.control.policy import create_lqr_policy
from nioc.envs import NonlinearReaching, Pendulum, FullyObservedWrapper
from nioc.envs.env_utils import angle_normalize
from nioc.control import lqr, glqr, ilqr
from nioc.control import ilqr_fixed

from envs import ILQRProblem


def test_ilqr_nonlinear_reaching():
    target = jnp.array([0.05, 0.4])
    env = NonlinearReaching(target=target)
    params = env.get_params_type()(action_cost=1e-5, velocity_cost=1e-3)

    T = 50

    x0 = jnp.array([jnp.pi / 4, jnp.pi / 2, 0., 0.])

    for lqr_module in [lqr, glqr]:
        gains, X, U = ilqr.solve(env, U_init=jnp.zeros(shape=(T, 2)), x0=x0, params=params, lqr=lqr_module)

        pos = vmap(env.e)(X)

        # check that final position is reasonably close to target
        assert jnp.allclose(pos[-1], target, atol=1e-3, rtol=0.)


def test_ilqr_pendulum():
    model = FullyObservedWrapper(Pendulum)()

    T = 50
    params = model.get_params_type()(velocity_cost=1e-6, action_cost=1e-6)
    x0 = model._reset(None, params)

    for lqr_module in [lqr, glqr]:
        gains, xbar, ubar = ilqr.solve(model, x0=x0, U_init=jnp.zeros((T, 1)), params=params, lqr=lqr_module)

        policy = create_lqr_policy(gains, xbar, ubar)
        x, y, u, cost = model.simulate(jr.PRNGKey(0), trials=50, steps=T, params=params, policy=policy)

        # check that final angle is close to 0
        assert jnp.allclose(angle_normalize(x[:, -1, 0].mean()), jnp.array(0.), atol=1e-2, rtol=0.)


class Params(NamedTuple):
    Q: jnp.ndarray
    q: jnp.ndarray
    Qf: jnp.ndarray
    R: jnp.ndarray
    r: jnp.ndarray
    A: jnp.ndarray
    B: jnp.ndarray


def init_stable(key, state_dim):
    """Initialize a stable matrix with dimensions `state_dim`"""
    R = jr.normal(key, (state_dim, state_dim))
    A, _ = jnp.linalg.qr(R)
    return 0.5 * A


def init_theta(key, state_dim, control_dim) -> Params:
    Q = jnp.eye(state_dim)
    q = jnp.ones(state_dim) * 0.01
    Qf = jnp.eye(state_dim)
    R = jnp.eye(control_dim)
    r = jnp.ones(control_dim)
    key, subkey = jr.split(key)
    A = init_stable(subkey, state_dim)
    key, subkey = jr.split(key)
    B = jr.normal(subkey, (state_dim, control_dim))
    return Params(Q=Q, q=q, Qf=Qf, R=R, r=r, A=A, B=B)


def init_params(key, state_dim, control_dim):
    """Initialize random parameters."""
    key, subkey = jr.split(key)
    x0 = jr.normal(subkey, (state_dim,))
    key, subkey = jr.split(key)
    theta = init_theta(subkey, state_dim, control_dim)
    return x0, theta


def test_ilqr_custom_vjp():
    # problem dimensions
    state_dim, control_dim, T, maxiter = 10, 3, 30, 30
    # random key
    key = jr.PRNGKey(42)
    # initialize ilqr
    ilqr_problem = ILQRProblem(state_dim, control_dim)
    # initialize solvers

    # initialize parameters
    x0, params = init_params(key, state_dim, control_dim)
    # initialize state
    Uinit = jnp.zeros((T, control_dim))

    # simulate some average data
    gains, X_data, _ = ilqr.solve(ilqr_problem, x0, Uinit, params)

    # IOC MSE loss
    def loss(params):
        gains, X, U = ilqr.solve(ilqr_problem, x0, Uinit, params)

        # MSE between simulated and real data
        return jnp.mean((X - X_data) ** 2)

    # check along one random direction
    check_grads(loss, (params,), 1, modes=("rev",))


def test_ilqr_fixed():
    target = jnp.array([0.05, 0.4])
    env = NonlinearReaching(target=target)
    params = env.get_params_type()()

    T = 50

    b0 = (jnp.array([jnp.pi / 4, jnp.pi / 2, 0., 0.]), jnp.eye(4) * .1)

    gains, X, U = ilqr.solve(env, U_init=jnp.zeros(shape=(T, 2)), x0=b0[0], params=params)

    gains_fixed, X_new, U_new = ilqr_fixed.solve(env, X, U, params)

    # check that LQR gains are roughly the same
    assert jnp.allclose(gains.L, gains_fixed.L, atol=1e-3, rtol=0.)
