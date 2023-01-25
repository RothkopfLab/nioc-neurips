from functools import partial
from typing import Type, TypeVar

from jax import lax, numpy as jnp, jacobian, jit, random

from nioc.envs.base import Env

T = TypeVar("T", bound=Env)


def FullyObservedWrapper(cls: Type[T]) -> Type[T]:
    """ Creates a fully observed version of an environment
    by redefining env._observation() to return the state
    """

    class FullyObserved(cls):
        def __init__(self, *args, **kwargs):
            super(FullyObserved, self).__init__(*args, **kwargs)

        def _observation(self, state, noise, params):
            return state

    return FullyObserved


def EKFWrapper(cls: Type[T]) -> Type[T]:
    class EKF(cls):
        def __init__(self, b0, *args, **kwargs):
            super().__init__(*args, **kwargs)

            self.b0 = b0

            self.dfdx = jacobian(self._dynamics, argnums=0)  # df / dx
            self.dfdv = jacobian(self._dynamics, argnums=2)  # df / dv

            self.dgdx = jacobian(self._observation, argnums=0)  # dg / dx
            self.dgdw = jacobian(self._observation, argnums=1)  # dg / dw

        def _reset(self, noise, params):
            return super()._reset(noise, params), self._reset_belief()

        def _reset_belief(self):
            return self.b0

        @partial(jit, static_argnums=(0,))
        def filter_step(self, b, P, action, y, params):
            A = self.dfdx(b, action, jnp.zeros(self.state_noise_shape), params)
            V = self.dfdv(b, action, jnp.zeros(self.state_noise_shape), params)

            H = self.dgdx(b, jnp.zeros(self.obs_noise_shape), params)
            W = self.dgdw(b, jnp.zeros(self.obs_noise_shape), params)

            # use observation at time t instead of t+1
            K = A @ P @ H.T @ jnp.linalg.inv(H @ P @ H.T + W @ W.T)

            P = V @ V.T + (A - K @ H) @ P @ A.T

            b_pred = self._dynamics(b, action, jnp.zeros(self.state_noise_shape), params)

            b = b_pred + K @ (y - self._observation(b, jnp.zeros(self.obs_noise_shape), params))

            return b, P

        @partial(jit, static_argnums=(0,))
        def step(self, env_state, belief, action, params):
            state, key = env_state  # unpack state
            b, P = belief

            cost = self._cost(state, action, params)

            # sample noises
            pro_noise_key, obs_noise_key, new_key = random.split(key, 3)
            pro_noise = random.normal(pro_noise_key, shape=self.state_noise_shape)
            obs_noise = random.normal(obs_noise_key, shape=self.obs_noise_shape)

            # get observation
            obsv = self._observation(state, obs_noise, params)

            # apply belief update
            belief = self.filter_step(b, P, action, obsv, params)

            # apply dynamics
            new_state = self._dynamics(state, action, pro_noise, params)

            env_state = new_state, new_key

            return env_state, belief, cost

        @partial(jit, static_argnums=(0,))
        def reset(self, key, params):
            pro_noise_key, obs_noise_key, new_key = random.split(key, 3)
            new_state, new_belief = self._reset(random.normal(pro_noise_key, shape=self.state_noise_shape), params)
            env_state = new_state, new_key
            return env_state, new_belief

        def rollout(self, key, steps, policy, params):
            initial_state, initial_belief = self.reset(key, params)

            def scan_body(carry, t):
                (state, key), belief = carry

                # sample action
                action_noise_key, new_key = random.split(key, 2)
                acion_noise = random.normal(action_noise_key, shape=self.action_shape)
                action = policy(t, belief, acion_noise)

                env_state, belief, cost = self.step((state, new_key), belief, action, params)
                return (env_state, belief), (env_state, belief, action, cost)

            (final_state, final_belief), (env_state, belief, actions, cost) = lax.scan(scan_body,
                                                                                       (initial_state, initial_belief),
                                                                                       jnp.arange(steps))

            states = jnp.vstack([initial_state[0], env_state[0]])
            xhat = jnp.vstack([initial_belief[0], belief[0]])
            covs = jnp.vstack([initial_belief[1][None], belief[1]])

            costs = jnp.hstack([cost, jnp.array([self._final_cost(final_state[0], params)])])
            return states, (xhat, covs), actions, costs

    return EKF
