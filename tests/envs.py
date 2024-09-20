from typing import NamedTuple
import jax
from jax import random, numpy as jnp

from nioc.envs import Env

phi = lambda x: jax.nn.relu(x)

class RandomLinearProblemParams(NamedTuple):
    pass

class RandomLinearProblem(Env):
    def __init__(self, key, obs_noise=1.):
        key, subkey = random.split(key)
        self.A = random.normal(subkey, shape=(5, 5))

        key, subkey = random.split(key)
        self.B = random.normal(subkey, shape=(5, 2))

        key, subkey = random.split(key)
        self.H = random.normal(subkey, shape=(3, 5))

        key, subkey = random.split(key)
        self.V = random.normal(subkey, shape=(5, 1))

        key, subkey = random.split(key)
        self.W = random.normal(subkey, shape=(3, 3)) * obs_noise

        super().__init__(state_shape=(5,), action_shape=(2,), observation_shape=(3,),
                         state_noise_shape=(1,))

    def _dynamics(self, state, action, noise, params):
        return self.A @ state + self.B @ action + self.V @ noise

    def _observation(self, state, noise, params):
        return self.H @ state + self.W @ noise

    def _cost(self, state, action, params):
        return 0.

    def _final_cost(self, state, params):
        return 0.

    def _reset(self, noise, params):
        return jnp.zeros((5,))


    @staticmethod
    def get_params_type():
        return RandomLinearProblemParams

    @staticmethod
    def get_params_bounds():
        return RandomLinearProblemParams(), RandomLinearProblemParams()

class ILQRProblem(Env):

    def __init__(self, state_dim: int, control_dim: int):
        super().__init__(state_shape=(state_dim,), action_shape=(control_dim,), observation_shape=(state_dim,))

    def _dynamics(self, state, action, noise, params):
        return params.A @ phi(state) + params.B @ action + 0.5

    def _cost(self, x, u, theta):
        n = x.shape[-1]
        m = u.shape[-1]
        lQ = 0.5 * jnp.dot(jnp.dot(theta.Q, x), x)
        lq = jnp.dot(theta.q, x)
        lR = 1e-4 * jnp.dot(jnp.dot(theta.R, u), u)
        lM = -1e-4 * jnp.dot(jnp.dot(jnp.ones((n, m)), u), x)
        lr = 1e-4 * jnp.dot(theta.r, u)
        return lQ + lq + lR + lM + lr

    def _final_cost(self, xf, theta):
        return 0.5 * jnp.dot(jnp.dot(theta.Qf, xf), xf)

    def _reset(self, noise, params):
        # x0, theta = init_params(jr.PRNGKey(0), self.state_shape[0], self.action_shape[0])
        return noise

    def _observation(self, state, noise, params):
        return state
