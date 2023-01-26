from typing import Callable
import jax.numpy as jnp
import jax.scipy as js
from jax import vmap, jacobian

from nioc.envs import Env
from nioc.control import glqr, gilqr, ilqr, ilqr_fixed
from nioc.control.policy import create_lqr_policy, create_maxent_lqr_policy
from nioc.infer.utils import estimate_controls


class InverseILQR:
    def __init__(self, env: Env, solve: Callable = ilqr.solve, maxent_temp: float = 0.):
        self.env = env
        self.solve = solve
        if maxent_temp > 0.:
            self.create_policy = lambda gains, xbar, ubar: create_maxent_lqr_policy(gains, xbar, ubar,
                                                                                    inv_temp=maxent_temp)
        else:
            self.create_policy = create_lqr_policy

    def moments(self, x, policy, params):
        def step(xt, t):
            # apply policy
            policy_noise_zero = jnp.zeros(self.env.action_shape)
            ut = policy(t, xt, policy_noise_zero)

            # propagation of mean
            mu = self.env._dynamics(xt, ut, jnp.zeros(self.env.state_noise_shape), params)

            # approximate propagation of covariance
            V = jacobian(self.env._dynamics, argnums=2)(xt, ut,
                                                        jnp.zeros(self.env.state_noise_shape),
                                                        params)

            # add noise due to policy
            dyn_policy_noise = lambda noise: self.env._dynamics(xt, policy(t, xt, noise),
                                                                jnp.zeros(self.env.state_noise_shape), params)
            W = jacobian(dyn_policy_noise)(policy_noise_zero)

            Sigma = V @ V.T + W @ W.T

            return mu, Sigma

        mu, Sigma = vmap(step)(x[:-1], jnp.arange(x.shape[0] - 1))

        return mu, Sigma

    def apply_solver(self, x, params):
        T = x.shape[1] - 1

        gains, xbar, ubar = self.solve(self.env, x0=x[:, 0].mean(axis=0),
                                       U_init=jnp.zeros(shape=(T, self.env.action_shape[0])),
                                       params=params, max_iter=5)
        policy = self.create_policy(gains, xbar, ubar)

        return policy

    def loglikelihood(self, x, params):
        # get policy for current params
        policy = self.apply_solver(x, params)

        mu, Sigma = vmap(lambda xi: self.moments(xi, policy, params))(x)

        return jnp.sum(js.stats.multivariate_normal.logpdf(x[:, 1:],
                                                           mu,
                                                           Sigma + jnp.eye(self.env.state_shape[0]) * 1e-6))


class InverseGILQR(InverseILQR):
    def __init__(self, env: Env, maxent_temp: float = 0.):
        super().__init__(env, solve=gilqr.solve, maxent_temp=maxent_temp)


class FixedLinearizationInverseILQR(InverseILQR):
    def __init__(self, env: Env, maxent_temp: float = 0.):
        super().__init__(env, solve=ilqr_fixed.solve, maxent_temp=maxent_temp)

    def apply_solver(self, x, params):
        # get policy for current params
        gains, xbar, ubar = self.solve(self.env, X=x, U=estimate_controls(x, self.env, params), params=params)
        policy = self.create_policy(gains, xbar, ubar)

        return policy

    def loglikelihood(self, x, params):
        mu, Sigma = vmap(lambda xi: self.moments(xi, self.apply_solver(xi, params), params))(x)

        return jnp.sum(js.stats.multivariate_normal.logpdf(x[:, 1:],
                                                           mu,
                                                           Sigma + jnp.eye(self.env.state_shape[0]) * 1e-6))


class FixedLinearizationInverseGILQR(FixedLinearizationInverseILQR):
    def __init__(self, env: Env, maxent_temp: float = 0.):
        super().__init__(env, maxent_temp=maxent_temp)

    def apply_solver(self, x, params):
        # get policy for current params
        gains, xbar, ubar = self.solve(self.env, X=x, U=estimate_controls(x, self.env, params), params=params, lqr=glqr)
        policy = self.create_policy(gains, xbar, ubar)

        return policy
