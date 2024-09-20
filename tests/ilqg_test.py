from jax import random, numpy as jnp

from nioc.control import ilqr
from nioc.control.policy import create_lqg_policy
from nioc.envs.navigation import Navigation, NavigationParams
from nioc.envs.wrappers import EKFWrapper


def test_ilqg_navigation():
    env = Navigation()
    params = NavigationParams(action_cost=1e-5, motor_noise=.25, velocity_cost=1., obs_noise=1e-1)

    T = 50

    key = random.PRNGKey(0)
    (x0, _), y0 = env.reset(key, params)
    gains, xbar, ubar = ilqr.solve(env, x0=x0, U_init=jnp.zeros((T, env.action_shape[0])), params=params)

    b0 = (x0, jnp.eye(x0.shape[0]) * 1.)
    ekf = EKFWrapper(Navigation)(b0=b0)
    lqg_policy = create_lqg_policy(gains, xbar, ubar)

    x, (xhat, P), u, cost = ekf.simulate(key, steps=T, trials=100, policy=lqg_policy, params=params)

    # test that the final position is reasonably close to the target on average
    assert jnp.allclose(x[:, -1, :2].mean(axis=0), env.target, atol=1e-2)
