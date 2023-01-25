from jax import random, vmap, numpy as jnp
import matplotlib.pyplot as plt

from nioc.envs import NonlinearReaching
from nioc.control import gilqr
from nioc.control.policy import create_lqg_policy
from nioc.envs.wrappers import EKFWrapper
from nioc.infer.inv_ilqg import FixedLinearizationInverseGILQG
from nioc.infer.utils import compute_mle

if __name__ == '__main__':
    env = NonlinearReaching()
    params = env.get_params_type()()
    x0 = env._reset(None, params)
    b0 = (x0, jnp.eye(x0.shape[0]))

    T = 50
    gains, xbar, ubar = gilqr.solve(p=env,
                                    x0=x0, U_init=jnp.zeros(shape=(T, env.action_shape[0])),
                                    params=params, max_iter=10)

    policy = create_lqg_policy(gains, xbar, ubar)

    ekf = EKFWrapper(NonlinearReaching)(b0=b0)
    xs, *_ = ekf.simulate(key=random.PRNGKey(0), steps=T, trials=20, policy=policy, params=params)
    pos = vmap(vmap(env.e))(xs)
    vel = vmap(vmap(env.edot))(xs)

    f, ax = plt.subplots()
    ax.plot(pos[..., 0].T, pos[..., 1].T, color="C0", alpha=0.8, linewidth=1)
    ax.scatter(env.target[0], env.target[1], color="k", marker="x", zorder=2, linewidth=1)
    f.show()

    ioc = FixedLinearizationInverseGILQG(env, b0=b0)

    result, result_params = compute_mle(xs, ioc, random.PRNGKey(0), restarts=10,
                                        bounds=env.get_params_bounds())
