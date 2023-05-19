from jax import jit, random, vmap, numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from nioc.envs import NonlinearReaching
from nioc.control import gilqr
from nioc.control.policy import create_lqg_policy
from nioc.envs.wrappers import EKFWrapper
from nioc.infer import FixedLinearizationInverseGILQG, FixedInverseMaxEntBaseline
from nioc.infer.utils import compute_mle


@jit
def simulate_trajectories(key, params):
    # solve control problem
    T = 50
    gains, xbar, ubar = gilqr.solve(p=env,
                                    x0=x0, U_init=jnp.zeros(shape=(T, env.action_shape[0])),
                                    params=params, max_iter=10)

    # create a policy and belief dynamics
    policy = create_lqg_policy(gains, xbar, ubar)
    ekf = EKFWrapper(NonlinearReaching)(b0=b0)

    # simulate some trajectories
    xs, *_ = ekf.simulate(key=key, steps=T, trials=20, policy=policy, params=params)

    # compute positions from angles
    pos = vmap(vmap(env.e))(xs)

    return xs, pos


if __name__ == '__main__':
    # get parameter type and setup default parameter values
    NonlinearReachingParams = NonlinearReaching.get_params_type()
    params = NonlinearReachingParams()
    print(f"Simulating data with: {params}")

    # create an environment, setup initial state and belief
    env = NonlinearReaching()
    x0 = env._reset(None, params)
    b0 = (x0, jnp.eye(x0.shape[0]))

    # setup random seed
    key = random.PRNGKey(1)

    # simulate some trajectories given ground truth parameters
    key, subkey = random.split(key)
    xs, pos = simulate_trajectories(subkey, params)

    # visualize trajectories
    f, ax = plt.subplots()
    ax.plot(pos[..., 0].T, pos[..., 1].T, color="C0", alpha=0.8, linewidth=1,
            label="Ground truth")
    ax.scatter(env.target[0], env.target[1], color="k", marker="x", zorder=2, linewidth=1)

    # setup inverse optimal control
    ioc = FixedLinearizationInverseGILQG(env, b0=b0)

    # run maximization of the IOC likelihood
    print("Running inverse ILQG...")
    key, subkey = random.split(key)
    result = compute_mle(xs, ioc, subkey, restarts=10,
                         bounds=env.get_params_bounds(), optim="L-BFGS-B")
    print(f"Estimated with inverse ILQG: {result.params}")

    # simulate some trajectories given the maximum likelihood estimate parameters
    key, subkey = random.split(key)
    xs_sim, pos_sim = simulate_trajectories(subkey, result.params)

    # plot trajectories simulated using MLE parameters
    ax.plot(pos_sim[..., 0].T, pos_sim[..., 1].T, color="C1", alpha=0.8, linewidth=1,
            label="Ours")

    # setup baseline method
    baseline = FixedInverseMaxEntBaseline(env)

    # run maxent baseline
    print("Running MaxEnt baseline...")
    key, subkey = random.split(key)
    baseline_result = compute_mle(xs, baseline, subkey, restarts=10,
                                  bounds=env.get_params_bounds(), optim="L-BFGS-B")
    print(f"Estimated with MaxEnt baseline: {baseline_result.params}")

    # simulate some trajectories given the parameters estimated using the maxent baseline
    key, subkey = random.split(key)
    xs_baseline, pos_baseline = simulate_trajectories(subkey, baseline_result.params)

    # plot trajectories simulated using MLE parameters
    ax.plot(pos_baseline[..., 0].T, pos_baseline[..., 1].T, color="C2", alpha=0.8, linewidth=1,
            label="Baseline")

    # make the plot pretty (get unique labels, remove spines, set labels etc.)
    handles, labels = ax.get_legend_handles_labels()
    labels, ids = np.unique(labels, return_index=True)
    handles = [handles[i] for i in ids]
    ax.legend(handles, labels, frameon=False)
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    f.suptitle("Reaching trajectories")
    f.show()
