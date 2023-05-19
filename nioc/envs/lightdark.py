from typing import NamedTuple

from jax import numpy as jnp

from nioc.envs.base import Env


class LightDarkParams(NamedTuple):
    exp_target_pos: float = 10 ** 0.  # this parameter is in exp space because our optimization runs in log space
    sigma: float = 0.1
    preferred_pos_cost: float = 0.


class LightDark(Env):

    @staticmethod
    def get_params_type():
        return LightDarkParams

    @staticmethod
    def get_params_bounds():
        return (LightDarkParams(exp_target_pos=10 ** -1.,
                                sigma=1e-2,
                                preferred_pos_cost=1e-3),
                LightDarkParams(exp_target_pos=10 ** 1.,
                                sigma=1.,
                                preferred_pos_cost=1.))

    def __init__(self, state_costs=0., action_cost=.5, final_state_costs=200., const=1e-6, motor_noise=.1):
        self.dt = 1.
        self.const = const
        self.state_costs = state_costs
        self.action_cost = action_cost
        self.final_state_costs = final_state_costs
        self.motor_noise = motor_noise

        super().__init__(state_shape=(2,), action_shape=(2,), observation_shape=(2,))

    def _dynamics(self, state, action, noise, params):
        return state + self.dt * action + self.motor_noise * jnp.diag(action) @ noise

    def _observation(self, state, noise, params):
        return state + self._observation_std(state, params) * noise

    def _observation_std(self, x, params):
        return params.sigma * jnp.sqrt(.5 * (x[0] - 5.) ** 2 + self.const)

    def _cost(self, state, action, params):
        return (self.action_cost * jnp.sum(action ** 2)
                + self.state_costs * jnp.sum((state - jnp.array([jnp.log10(params.exp_target_pos), 0.])) ** 2)
                + 0.01 * params.preferred_pos_cost * (state[0] - 5.) ** 2)

    def _final_cost(self, state, params):
        return self.final_state_costs * jnp.sum((state - jnp.array([jnp.log10(params.exp_target_pos), 0.])) ** 2)

    def _reset(self, noise, params):
        x0 = jnp.array([2., 2.])
        return x0


if __name__ == '__main__':

    from jax import random, vmap
    import matplotlib.pyplot as plt

    from nioc.envs.wrappers import EKFWrapper
    from nioc.control import gilqg, gilqr
    from nioc.control.policy import create_lqg_policy

    env = LightDark()

    x0 = jnp.array([2., 2.])
    Sigma0 = jnp.eye(2) * 3.
    b0 = (x0, Sigma0)

    for sigma in [1.]:
        T = 50
        params = LightDarkParams(sigma=sigma, preferred_pos_cost=1e-2)

        # solve with iLQG, partially observed version (Li & Todorov, 2007)
        gains_gilqg, xbar, ubar = gilqg.solve(env, x0=x0, Sigma0=Sigma0,
                                              U_init=jnp.zeros((T, 2)),
                                              max_iter=10,
                                              params=params)
        gilqg_policy = create_lqg_policy(gains_gilqg, xbar, ubar)

        # solve with iLQG, fully observed  version (Todorov & Li, 2005)
        gains_gilqr, xbar, ubar = gilqr.solve(env, x0=x0,
                                             U_init=jnp.zeros((T, 2)),
                                             max_iter=100,
                                             params=LightDarkParams(sigma=sigma,
                                                                    preferred_pos_cost=0.))
        gilqr_policy = create_lqg_policy(gains_gilqr, xbar, ubar)

        # plot results
        f, ax = plt.subplots(1, 2, figsize=(0.5 * 5.50107 * 1.61803398875, 0.5 * 5.50107), sharex=True, sharey=True)

        # plot trajectories for both policies
        for i, (name, policy) in enumerate({
                                               "Fully observed \n iLQG": gilqr_policy,
                                               "Partially observed \n iLQG": gilqg_policy,
                                           }.items()):
            # environment for simulation
            ekf = EKFWrapper(LightDark)(b0=b0)
            key = random.PRNGKey(10)

            xx = jnp.linspace(-1, 7)[None]
            std = ekf._observation_std(xx, params)

            x, (xhat, P), u, cost = ekf.simulate(key, steps=T, trials=50, policy=policy, params=params)

            ax[i].imshow(jnp.tile(std[None], (T, 1)), cmap="Greys",
                         extent=[xx.min(), xx.max(), -2, 4], )

            ax[i].plot(x[..., 0].T, x[..., 1].T, color="C1", label="state")
            ax[i].grid()
            ax[i].set_title(f"{name}")
            ax[i].set_aspect("equal", "box")
            ax[i].set_xticks([0, 2, 4, 6])
            ax[i].set_yticks([-2, 0, 2, 4])
        f.show()
