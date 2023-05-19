from matplotlib import pyplot as plt
from jax.config import config

config.update("jax_enable_x64", True)

from jax import random, vmap, numpy as jnp

from nioc.control import gilqg, gilqr
from nioc.control.policy import create_lqg_policy
from nioc.envs import LightDark
from nioc.envs.wrappers import EKFWrapper
from nioc.infer import InverseILQG, compute_mle, FixedInverseMaxEntBaseline

M = 100  # trials for true simulation
T = 50  # time steps
sigma = 0.2  # perceptual uncertainty standard deviation
init_var = 3.  # initial uncertainty variance

max_iter = 50  # iLQG iterations
optim = "L-BFGS-B"
params_to_infer = ["preferred_pos_cost", "sigma", "exp_target_pos"]


def plot_results(params_true, result_ilqg, result_baseline, cost_cmap="jet", obs_cmap="Greys"):
    f, ax = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(5.50107, 5.50107 / 1.33))

    final_cost_vmin = final_cost_map(params_true).min()
    final_cost_vmax = final_cost_map(params_true).max()
    cost_vmin = running_cost_map(result_baseline.params).min()
    cost_vmax = running_cost_map(result_baseline.params).max()
    obs_vmin = observation_map(params_true).min()
    obs_vmax = observation_map(params_true).max()

    # cost maps (true)
    ax[0, 0].pcolormesh(xv, yv, final_cost_map(params_true), cmap=cost_cmap, vmin=final_cost_vmin, vmax=final_cost_vmax,
                        rasterized=True)
    ax[0, 0].plot(xbar_true[:, 0], xbar_true[:, 1], color="C1", label="state")

    ax[0, 1].pcolormesh(xv, yv, running_cost_map(params_true), cmap=cost_cmap, vmin=cost_vmin, vmax=cost_vmax,
                        rasterized=True)
    ax[0, 1].plot(xbar_true[:, 0], xbar_true[:, 1], color="C1", label="state")

    # perceptual uncertainty map (true)
    ax[0, 2].pcolormesh(xv, yv, observation_map(params_true), cmap=obs_cmap, vmin=obs_vmin, vmax=obs_vmax,
                        rasterized=True)
    ax[0, 2].plot(xbar_true[:, 0], xbar_true[:, 1], color="C1", label="state")

    # cost maps (ours)
    ax[1, 0].pcolormesh(xv, yv, final_cost_map(result_ilqg.params), cmap=cost_cmap, vmin=final_cost_vmin,
                        vmax=final_cost_vmax, rasterized=True)
    ax[1, 0].plot(xbar_ilqg[:, 0], xbar_ilqg[:, 1], color="C1", label="state")

    ax[1, 1].pcolormesh(xv, yv, running_cost_map(result_ilqg.params), cmap=cost_cmap, vmin=cost_vmin, vmax=cost_vmax,
                        rasterized=True)
    ax[1, 1].plot(xbar_ilqg[:, 0], xbar_ilqg[:, 1], color="C1", label="state")

    # perceptual uncertainty map (ours)
    ax[1, 2].pcolormesh(xv, yv, observation_map(result_ilqg.params), cmap=obs_cmap, vmin=obs_vmin, vmax=obs_vmax,
                        rasterized=True)
    ax[1, 2].plot(xbar_ilqg[:, 0], xbar_ilqg[:, 1], color="C1", label="state")

    # cost maps (baseline)
    ax[2, 0].pcolormesh(xv, yv, final_cost_map(result_baseline.params), cmap=cost_cmap,
                        vmin=final_cost_vmin, vmax=final_cost_vmax, rasterized=True)
    ax[2, 0].plot(xbar_gilqr[:, 0], xbar_gilqr[:, 1], color="C1", label="state")

    ax[2, 1].pcolormesh(xv, yv, running_cost_map(result_baseline.params), cmap=cost_cmap,
                        vmin=cost_vmin, vmax=cost_vmax, rasterized=True)
    ax[2, 1].plot(xbar_gilqr[:, 0], xbar_gilqr[:, 1], color="C1", label="state")

    # perceptual uncertainty map (baseline)
    ax[2, 2].pcolormesh(xv, yv, observation_map(result_baseline.params), cmap=obs_cmap, vmin=obs_vmin,
                        vmax=obs_vmax, rasterized=True)
    ax[2, 2].plot(xbar_gilqr[:, 0], xbar_gilqr[:, 1], color="C1", label="state")

    # make x and y axis equal
    for axi in ax.flat:
        axi.set_aspect("equal", "box")

    # row titles
    ax[0, 0].set_title("Final\ncost")
    ax[0, 1].set_title("Running\ncost")
    ax[0, 2].set_title("Perceptual\nuncertainty")

    # column titles
    ax[0, 0].set_ylabel(f"ground truth")
    ax[1, 0].set_ylabel(f"ours")
    ax[2, 0].set_ylabel(f"baseline")

    f.tight_layout()
    f.show()


if __name__ == '__main__':
    key = random.PRNGKey(100)

    # setup environment and params
    env = LightDark()
    LightDarkParams = LightDark.get_params_type()
    params_true = LightDarkParams(sigma=sigma)
    x0 = jnp.array([2., 2.])
    Sigma0 = jnp.eye(2) * init_var
    b0 = (x0, Sigma0)

    # x and y coordinates of LightDark domain
    xv, yv = jnp.meshgrid(jnp.linspace(-1, 7), jnp.linspace(-2, 4))


    def final_cost_map(params):
        """ 2d map of cost at final time step """
        return vmap(vmap(lambda x: env._final_cost(x, params)))(jnp.stack((xv, yv), axis=-1))


    def running_cost_map(params):
        """ 2d map of cost during movement """
        # env._final_cost(x, params) + T *
        return vmap(vmap(lambda x: env._cost(x, jnp.zeros(env.action_shape), params)))(jnp.stack((xv, yv), axis=-1))


    def observation_map(params):
        """ 2d map of observation standard deviation in light-dark domain """
        return env._observation_std(jnp.stack((xv, yv)), params=params)


    # solve with belief-space iLQG
    gains_true, xbar_true, ubar_true = gilqg.solve(env, x0=x0, Sigma0=Sigma0,
                                                   U_init=jnp.zeros((T, 2)),
                                                   max_iter=max_iter,
                                                   params=params_true)
    policy_true = create_lqg_policy(gains_true, xbar_true, ubar_true)

    # environment for simulation
    ekf = EKFWrapper(LightDark)(b0=b0)

    # simulate some data using the belief-space iLQG policy and EKF
    x, (xhat, P), u, cost = ekf.simulate(key, steps=T, trials=M, policy=policy_true, params=params_true)

    # setup inference method
    ioc_ilqg = InverseILQG(env=env, b0=b0, solve=gilqg.solve, kf=gilqg, max_iter=max_iter)

    # compute maximum likelihood estimate using inverse ILQG
    result_ilqg = compute_mle(x, ioc_ilqg, key, restarts=25,
                              params=params_to_infer,
                              bounds=LightDark.get_params_bounds(),
                              optim=optim)

    gains_ilqg, xbar_ilqg, ubar_ilqg = gilqg.solve(env, x0=x0, Sigma0=Sigma0,
                                                   U_init=jnp.zeros((T, 2)),
                                                   max_iter=max_iter,
                                                   params=result_ilqg.params)

    # setup inference method
    ioc_baseline = FixedInverseMaxEntBaseline(env=env, maxent_temp=1e-6)

    # compute maximum likelihood estimate
    result_baseline = compute_mle(x, ioc_baseline, key, restarts=25,
                                  params=params_to_infer,
                                  bounds=LightDark.get_params_bounds(),
                                  optim=optim)

    gains_gilqr, xbar_gilqr, ubar_gilqr = gilqr.solve(env, x0=x0, Sigma0=Sigma0,
                                                      U_init=jnp.zeros((T, 2)),
                                                      max_iter=max_iter,
                                                      params=result_baseline.params)

    # plotting
    plot_results(params_true, result_ilqg, result_baseline)
