from typing import NamedTuple
import jax.numpy as jnp

from nioc.envs import Env


class NavigationParams(NamedTuple):
    action_cost: float = 1e-3
    velocity_cost: float = 1e-3
    obs_noise: float = .1
    motor_noise: float = 0.5


class Navigation(Env):

    def __init__(self, dt=1. / 10.):
        self.dt = dt

        self.target = jnp.array([1., 1.])

        super().__init__(state_shape=(4,), action_shape=(2,), state_noise_shape=(2,),
                         observation_shape=(3,))

    def _dynamics(self, state, action, noise, params):
        u = action + params.motor_noise * action * noise
        return state + self.dt * jnp.array((jnp.cos(state[2]) * state[3],
                                            jnp.sin(state[2]) * state[3],
                                            u[0],
                                            u[1]))

    def _observation(self, state, noise, params):
        heading = state[2]

        bearing = jnp.arctan2(self.target[1] - state[1], self.target[0] - state[0])

        distance = jnp.sqrt((self.target[1] - state[1]) ** 2 + (self.target[0] - state[0]) ** 2)

        speed = state[3]

        y = jnp.array([heading - bearing, distance, speed])
        return y + params.obs_noise * noise

    def _cost(self, state, action, params):
        return params.action_cost * jnp.sum(action ** 2)

    def _final_cost(self, state, params):
        return jnp.sum((state[0:2] - self.target) ** 2) + params.velocity_cost * state[3] ** 2

    def _reset(self, noise, params):
        return jnp.array([0., 0., jnp.pi / 2, 0.])

    @staticmethod
    def get_params_type():
        return NavigationParams

    @staticmethod
    def get_params_bounds():
        lo = NavigationParams(action_cost=1e-4, velocity_cost=1e-4, motor_noise=1e-1, obs_noise=1e-2)
        hi = NavigationParams(action_cost=1., velocity_cost=10., motor_noise=1., obs_noise=1.)

        return lo, hi
