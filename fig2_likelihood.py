from jax import jit, random, vmap, numpy as jnp
import matplotlib.pyplot as plt

from nioc.envs.wrappers import EKFWrapper
from nioc.envs import NonlinearReaching
from nioc.control import gilqr
from nioc.control.policy import create_lqg_policy
from nioc.infer import FixedLinearizationInverseGILQG

from plot_utils import update_rcparams

update_rcparams()

x0 = jnp.array([jnp.pi / 4, jnp.pi / 2, 0., 0.])
b0 = (x0, jnp.eye(x0.shape[0]))


@jit
def simulate_trajectories(key, params, target=jnp.array([0.05, 0.5])):
    env = NonlinearReaching(target=target)

    gains, xbar, ubar = gilqr.solve(env, x0=x0,
                                    U_init=jnp.zeros((T, env.action_shape[0])),
                                    params=params)
    policy = create_lqg_policy(gains, xbar, ubar)

    ekf = EKFWrapper(NonlinearReaching)(b0=b0)

    xs, *_ = ekf.simulate(key=key, steps=T, trials=n, policy=policy, params=params)

    pos = vmap(vmap(env.e))(xs)
    return xs, pos


if __name__ == '__main__':
    # setup default parameters
    NonlinearReachingParams = NonlinearReaching.get_params_type()

    T = 50  # time steps
    n = 50  # trials

    # number of directions around the circle
    n_angles = 8
    # radius of the circle
    radius = 0.1
    # starting position in Cartesian coordinates
    starting_position = NonlinearReaching().e(x0)

    # setup random seed
    key = random.PRNGKey(1)

    # initialize plotting
    f, ax = plt.subplots(3, 2, sharex="row", sharey="row",
                         figsize=(3.25063, 3.25063 / 2 * 3))

    # for two different sets of parameters
    for j, params in enumerate([NonlinearReachingParams(action_cost=5e-5, motor_noise=.02,
                                                        velocity_cost=1e-1, obs_noise=1.),
                                NonlinearReachingParams(action_cost=5e-4, motor_noise=.1,
                                                        velocity_cost=1e-1, obs_noise=1.)]):

        # simulate data given true parameters, for all angles around the circle
        for i, angle in enumerate(jnp.arange(n_angles) / n_angles * 2 * jnp.pi):
            # get target position
            target = starting_position + jnp.array([jnp.cos(angle) * radius, jnp.sin(angle) * radius])

            # simulate trajectories
            key, subkey = random.split(key)
            xs, pos = simulate_trajectories(subkey, params, target)

            # visualize trajectories and target
            ax[0, j].plot(pos[..., 0].T, pos[..., 1].T, color=f"C{i}", alpha=0.8, linewidth=1)
            ax[0, j].scatter(target[0], target[1], color="k", marker="x", zorder=2, linewidth=1)

        ax[0, j].set_xlabel("x [m]")
        ax[0, j].set_xticks([-.12, -0.02, 0.08])
        ax[0, j].set_yticks([0.35, 0.45, 0.55])
        ax[0, j].set_xlim(-.14, .1)
        ax[0, j].set_ylim(.32, .57)

        # setup parameter grid to evaluate log likelihood (for two params: action cost and motor noise)
        grid_size = 25  # numer of points for log likelihood grid (increase for finer grid)
        action_costs, motor_noises = jnp.meshgrid(jnp.linspace(-5, -2, grid_size),
                                                  jnp.linspace(-2, -0.5, grid_size))

        # simulate trajectories (for last angle
        xs, pos = simulate_trajectories(key, params)
        ioc = FixedLinearizationInverseGILQG(NonlinearReaching(), b0=b0)

        # function that evaluates log likelihood at two of the params, leaves others fixed at true values
        ll_two_param = lambda action_cost, motor_noise: ioc.loglikelihood(xs,
                                                                          NonlinearReachingParams(
                                                                              action_cost=10 ** action_cost,
                                                                              velocity_cost=params.velocity_cost,
                                                                              motor_noise=10 ** motor_noise,
                                                                              obs_noise=params.obs_noise))
        # compute log likelihood for the whole grid
        lls = vmap(vmap(ll_two_param))(action_costs, motor_noises)

        # normalize log likelihood result (for plotting)
        norm_lls = lls / lls.sum()

        # get result and convert back to original parameter space
        max_ll_idx = jnp.unravel_index(lls.argmax(), lls.shape)
        action_cost_hat = 10 ** action_costs[max_ll_idx]
        motor_noise_hat = 10 ** motor_noises[max_ll_idx]

        # plot log likelihood
        vmin, vmax = jnp.percentile(lls, jnp.array([80, 100]))
        ax[1, j].pcolormesh(10 ** action_costs, 10 ** motor_noises, lls,
                            vmin=vmin, vmax=vmax,
                            rasterized=True, cmap="jet")
        ax[1, j].scatter(params.action_cost, params.motor_noise,
                         marker="o", color="black", label="true", s=10)
        ax[1, j].scatter(action_cost_hat, motor_noise_hat,
                         marker="x", color="magenta", label="MLE", s=10)
        ax[1, j].set_yscale("log")
        ax[1, j].set_xscale("log")
        ax[1, j].set_aspect('equal', 'box')
        ax[1, j].set_title("Log likelihood")
        ax[1, j].set_xlabel("$c_a$")
        ax[1, j].set_xlim(10 ** -5, 10 ** -3)

        # setup parameters (MLE results for action_cost and motor_noise, default values for others)
        mle_params = NonlinearReachingParams(action_cost=action_cost_hat,
                                             motor_noise=motor_noise_hat,
                                             velocity_cost=params.velocity_cost, obs_noise=params.obs_noise)

        # simulate data given inferred parameters, for all angles around the circle
        for i, angle in enumerate(jnp.arange(n_angles) / n_angles * 2 * jnp.pi):
            target = starting_position + jnp.array([jnp.cos(angle) * radius, jnp.sin(angle) * radius])

            # simulate trajectories
            key, subkey = random.split(key)
            xs, pos = simulate_trajectories(subkey, mle_params, target)

            # visualize trajectories and target
            ax[2, j].plot(pos[..., 0].T, pos[..., 1].T, color=f"C{i}", alpha=0.8, linewidth=1)
            ax[2, j].scatter(target[0], target[1], color="k", marker="x", zorder=2, linewidth=1)

        ax[2, j].set_xlabel("x [m]")
        ax[2, j].set_xticks([-.12, -0.02, 0.08])
        ax[2, j].set_yticks([0.35, 0.45, 0.55])
        ax[2, j].set_xlim(-.14, .1)
        ax[2, j].set_ylim(.32, .57)

        if j == 0:
            ax[0, j].set_ylabel("y [m]")
            ax[2, j].set_ylabel("y [m]")
            ax[1, j].set_ylabel("$\sigma_m$")
            ax[1, j].legend(fontsize=6)

            ax[0, j].set_title("Low cost, low noise", fontsize=6)

        else:
            ax[0, j].set_title("High cost, high noise", fontsize=6)

    f.tight_layout()
    f.show()
