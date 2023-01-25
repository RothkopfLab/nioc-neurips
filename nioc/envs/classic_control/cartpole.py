from typing import NamedTuple
import jax.numpy as jnp

from nioc.envs import Env


class CartPoleParams(NamedTuple):
    action_cost: float = 1e-3
    velocity_cost: float = 1e-2
    motor_noise: float = 1e-1
    obs_noise: float = 1.


class CartPole(Env):
    """ Classic cart-pole system adapted from gym Cartpole v1
        (https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py)

        with the following differences:
        - continuous instead of discrete controls
        - a bit of noise on the dynamics and observations
    """

    @staticmethod
    def get_params_type():
        return CartPoleParams

    @staticmethod
    def get_params_bounds():
        lo = CartPoleParams(action_cost=5e-4, velocity_cost=1e-2, motor_noise=0.2, obs_noise=1e-2)
        hi = CartPoleParams(action_cost=1., velocity_cost=10., motor_noise=1.5, obs_noise=10.)
        return lo, hi

    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates

        super().__init__(state_shape=(4,), action_shape=(1,), observation_shape=(4,),
                         state_noise_shape=(1,))

    def _dynamics(self, state, action, noise, params):
        x, theta, x_dot, theta_dot = state
        force = action[0] + params.motor_noise * action[0] * noise[0]

        costheta = jnp.cos(theta)
        sintheta = jnp.sin(theta)

        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
                self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        # standard euler integration
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        return jnp.array((x, theta, x_dot, theta_dot))

    def _observation(self, state, noise, params):
        return state + params.obs_noise * jnp.array([1., 1., 0.1, 0.1]) * noise

    def _cost(self, state, action, params):
        # Code 8.4
        # return (state[0] - 1.) ** 2 + (state[2] - jnp.pi) ** 2 + 1e-4 * jnp.sum(action ** 2)
        #
        # return jnp.sum((state - jnp.array([1., 0., jnp.pi, 0.])) ** 2) + params.action_cost * jnp.sum(action ** 2)
        # pos_cost = (state[0] - 1.) ** 2
        # vel_cost = state[1] ** 2
        # ang_cost = (angle_normalize(state[2]) - angle_normalize(jnp.pi)) ** 2
        # ang_vel_cost = state[3] ** 2

        return params.action_cost * jnp.sum(action ** 2)

    def _final_cost(self, state, params):
        # pos_cost = (state[0] - 1.) ** 2
        # vel_cost = state[1] ** 2
        # ang_cost = (angle_normalize(state[2]) - angle_normalize(jnp.pi)) ** 2
        # ang_vel_cost = state[3] ** 2
        return state @ jnp.diag(jnp.array([1., 1., params.velocity_cost, params.velocity_cost])) @ state.T

    def _reset(self, noise, params):
        # Code 8.5
        return jnp.array([-1., 0., 0., 0.])
