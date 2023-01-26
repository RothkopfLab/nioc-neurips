from typing import NamedTuple

import jax.numpy as jnp

from nioc.envs.base import Env


class NonlinearReachingParams(NamedTuple):
    action_cost: float = 1e-4
    velocity_cost: float = 1e-2
    motor_noise: float = 1e-1
    obs_noise: float = 1.


class NonlinearReaching(Env):
    def __init__(self, dt=0.01, target=jnp.array([0.05, 0.5]), x0=jnp.array([jnp.pi / 4, jnp.pi / 2, 0., 0.]),
                 indep_noise=0., upper_arm_length=0.3, forearm_length=0.33, I1=0.025, I2=0.045):
        """ Non-linear reaching task from Weiwei Li's PhD thesis

        Args:
            dt (float): time step duration
            target (jnp.array): target position (x, y)
            x0 (jnp.array): initial state (theta1, theta2, theta1_dot, theta2_dot)
            indep_noise (float): state-independent noise on dynamics
            upper_arm_length (float): upper arm length in meters
            forearm_length (float): forearm length in meters
            I1, I2 (float): moments of inertia of the joints (kg / m**2)
        """
        self.dt = dt

        self.target = target
        self.x0 = x0

        self.l1 = upper_arm_length
        self.l2 = forearm_length

        # setting the centers of mass of the arm links
        # based on the original values in Li (2006)
        # s1 = 0.11 / 0.3 * self.l1
        self.s2 = 0.16 / 0.33 * self.l2

        # scale the masses of the arm segments (assumed cylinders)
        # according to their lengths
        # self.m1 = 1.4 / 0.3 * self.l1
        self.m2 = 1.1 / 0.33 * self.l2

        # eqn 3.5
        self.d1 = I1 + I2 + self.m2 * self.l1 ** 2
        self.d2 = self.m2 * self.l1 * self.s2
        self.d3 = I2

        self.bii = 0.05
        self.bij = 0.025

        self.v = indep_noise
        super().__init__(state_shape=(4,), action_shape=(2,), observation_shape=(4,))

    def _dynamics(self, state, action, noise, params):
        det = self.d1 * self.d3 - self.d3 ** 2 - (self.d2 * jnp.cos(state[1])) ** 2
        dx = jnp.array([state[2],
                        state[3],
                        1 / det * (-self.d2 * self.d3 * (state[2] + state[3]) ** 2 * jnp.sin(state[1]) - self.d2 ** 2 *
                                   state[2] ** 2 * jnp.sin(
                                    state[1]) * jnp.cos(state[1]) - self.d2 * (
                                           self.bij * state[2] + self.bii * state[3]) * jnp.cos(state[1]) + (
                                           self.d3 * self.bii - self.d3 * self.bij) * state[2] + (
                                           self.d3 * self.bij - self.d3 * self.bii) * state[3]),
                        1 / det * (self.d2 * self.d3 * state[3] * (2 * state[2] + state[3]) * jnp.sin(
                            state[1]) + self.d1 * self.d2 * state[
                                       2] ** 2 * jnp.sin(state[1]) + self.d2 ** 2 * (
                                           state[2] + state[3]) ** 2 * jnp.sin(state[1]) * jnp.cos(
                            state[1]) + self.d2 * (
                                           (2 * self.bij - self.bii) * state[2] + (2 * self.bii - self.bij) * state[
                                       3]) * jnp.cos(state[1]) + (
                                           self.d1 * self.bij - self.d3 * self.bii) * state[2] + (
                                           self.d1 * self.bii - self.d3 * self.bij) * state[3])])
        G = 1 / det * jnp.array([[0., 0.],
                                 [0., 0.],
                                 [self.d3, -(self.d3 + self.d2 * jnp.cos(state[1]))],
                                 [-(self.d3 + self.d2 * jnp.cos(state[1])), self.d1 + 2 * self.d2 * jnp.cos(state[1])]])

        H = jnp.array([[0., 0.], [0., 0.], [1., 0.], [0., 1.]])

        du = G @ action
        f = self.dt * (dx + du)
        w = jnp.sqrt(self.dt) * (G @ (params.motor_noise * jnp.diag(action) @ noise[2:]) + H @ (self.v * noise[:2]))
        return state + f + w

    def _observation(self, state, noise, params):
        return state + self.dt * params.obs_noise * jnp.eye(self.observation_shape[0]) @ noise

    def _cost(self, state, action, params):
        return 0.5 * params.action_cost * jnp.sum(action ** 2)

    def _final_cost(self, state, params):
        # 1e4 *
        return jnp.sum((self.e(state) - self.target) ** 2) + params.velocity_cost * jnp.sum(self.edot(state) ** 2)

    def _reset(self, noise, params):
        x0 = self.x0
        return x0

    def e(self, x):
        return jnp.array([self.l1 * jnp.cos(x[0]) + self.l2 * jnp.cos(x[0] + x[1]),
                          self.l1 * jnp.sin(x[0]) + self.l2 * jnp.sin(x[0] + x[1])])

    def gamma(self, x):
        return jnp.array([[-self.l1 * jnp.sin(x[0]) - self.l2 * jnp.sin(x[0] + x[1]), -self.l2 * jnp.sin(x[0] + x[1])],
                          [self.l1 * jnp.cos(x[0]) + self.l2 * jnp.cos(x[0] + x[1]), self.l2 * jnp.cos(x[0] + x[1])]])

    def edot(self, x):
        return self.gamma(x) @ x[2:]

    @staticmethod
    def get_params_type():
        return NonlinearReachingParams

    @staticmethod
    def get_params_bounds():
        lo = NonlinearReachingParams(action_cost=1e-5, velocity_cost=1e-3, motor_noise=1e-2, obs_noise=1e-1)
        hi = NonlinearReachingParams(action_cost=1e-1, velocity_cost=1e-1, motor_noise=1., obs_noise=100.)
        return lo, hi
