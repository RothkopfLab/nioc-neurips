from typing import NamedTuple
import jax.numpy as jnp

from nioc.envs.base import Env
from nioc.envs.env_utils import angle_normalize


class PendulumParams(NamedTuple):
    action_cost: float = 1e-3
    velocity_cost: float = 1e-2
    motor_noise: float = 1e-1
    obs_noise: float = 1.


class Pendulum(Env):
    """ Inverted pendulum adapted from gym Pendulum-v1
        (github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py)
        (https://www.gymlibrary.dev/environments/classic_control/pendulum/)

    with the following differences:
    - because we are considering the finite-horizon case, we only apply the state costs at the final time step
    - to make it a stochastic system, we have added signal-dependent motor noise and observation noise

    """

    @staticmethod
    def get_params_type():
        return PendulumParams

    @staticmethod
    def get_params_bounds():
        lo = PendulumParams(action_cost=1e-3, velocity_cost=1e-3, motor_noise=1e-1, obs_noise=1e-2)
        hi = PendulumParams(action_cost=1e-1, velocity_cost=1e-1, motor_noise=2., obs_noise=10.)
        return lo, hi

    def __init__(self):
        self.m = 1.0
        self.l = 1.0
        self.g = 9.8

        self.dt = 0.01

        super().__init__(state_shape=(2,), action_shape=(1,), observation_shape=(3,),
                         state_noise_shape=(1,))

    def _dynamics(self, state, action, noise, params):
        b = 3 / (self.m * self.l ** 2)

        dx = jnp.array([state[1],
                        3 * self.g / (2 * self.l) * jnp.sin(state[0]) + b * action[0]])

        return state + self.dt * dx + self.dt * params.motor_noise * jnp.array([[0.01 * b], [b]]) @ (action * noise)

    def _observation(self, state, noise, params):
        return jnp.array([jnp.cos(state[0]),
                          jnp.sin(state[0]),
                          state[1],
                          ]) + params.obs_noise * noise

    def _final_cost(self, state, params):
        return angle_normalize(state[0]) ** 2 + params.velocity_cost * state[1] ** 2

    def _cost(self, state, action, params):
        return self.dt * params.action_cost * action[0] ** 2

    def _reset(self, noise, params):
        return jnp.array([jnp.pi, 0.])  # Initial condition
