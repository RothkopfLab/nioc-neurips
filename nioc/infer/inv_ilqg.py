from typing import Callable, Any, Tuple
import jax.numpy as jnp
from jax import vmap, jacobian, lax, scipy as js

from nioc.envs import Env
from nioc.belief import kf, Belief
from nioc.control import ilqr, gilqr, ilqg_fixed, glqg, lqg
from nioc.control.policy import create_lqr_policy, create_maxent_lqr_policy
from nioc.infer.utils import estimate_controls


class InverseILQG:
    def __init__(self, env: Env, b0: Belief, solve: Callable = ilqr.solve, maxent_temp: float = 0.):
        self.env = env
        self.solve = solve

        self.b0, self.Sigma0 = b0

        self.xdim = self.b0.shape[0]
        self.bdim = self.xdim

        if maxent_temp > 0:
            self.create_policy = lambda gains, xbar, ubar: create_maxent_lqr_policy(gains, xbar, ubar, maxent_temp)
        else:
            self.create_policy = create_lqr_policy

    def moments(self, x: jnp.ndarray, joint_dynamics: Callable,
                policy: Callable, params: Any) -> Tuple[jnp.ndarray, jnp.ndarray]:
        d = self.xdim

        def step(carry, t):
            mu, Sigma = carry

            policy_noise_zero = jnp.zeros(self.env.action_shape)

            # propagation of mean
            mu_z = joint_dynamics(t, x[t], mu,
                                  jnp.zeros(self.env.state_noise_shape),
                                  jnp.zeros(self.env.obs_noise_shape),
                                  policy_noise_zero,
                                  policy, params)

            # approximate propagation of covariance
            gb, gm, gn, go = jacobian(joint_dynamics, argnums=(2, 3, 4, 5))(t, x[t], mu,
                                                                            jnp.zeros(self.env.state_noise_shape),
                                                                            jnp.zeros(self.env.obs_noise_shape),
                                                                            policy_noise_zero,
                                                                            policy, params)
            Sigma_z = gb @ Sigma @ gb.T + gm @ gm.T + gn @ gn.T + go @ go.T

            # condition on next observation (multivariate Gaussian conditioning equations)
            mu = mu_z[d:] + Sigma_z[d:, :d] @ jnp.linalg.solve(Sigma_z[:d, :d] + jnp.eye(d) * 1e-6,
                                                               x[t + 1] - mu_z[:d])
            Sigma = Sigma_z[d:, d:] - Sigma_z[d:, :d] @ jnp.linalg.solve(Sigma_z[:d, :d] + jnp.eye(d) * 1e-6,
                                                                         Sigma_z[:d, d:])

            return (mu, Sigma), (mu_z, Sigma_z)

        _, (mu, Sigma) = lax.scan(step, (self.b0, jnp.eye(self.bdim)), jnp.arange(x.shape[0] - 1))

        return mu, Sigma

    def loglikelihood(self, x: jnp.ndarray, params: Any):
        # get policy for current params
        policy, joint_dynamics = self.apply_solver(x, params)

        mu, Sigma = vmap(lambda xi: self.moments(xi, joint_dynamics, policy, params))(x)

        d = x.shape[-1]

        return jnp.sum(js.stats.multivariate_normal.logpdf(x[:, 1:],
                                                           mu[:, :, :d],
                                                           Sigma[:, :, :d, :d] + jnp.eye(d) * 1e-6))

    def apply_solver(self, x: jnp.ndarray, params: Any) -> Tuple[Callable, Callable]:
        T = x.shape[1] - 1

        # TODO: x0 could be different for different trials. solver depends on x0. how do we want to deal with this?
        #  right now, I am using the mean of the initial belief
        gains, xbar, ubar = self.solve(p=self.env, x0=self.b0,
                                       U_init=jnp.zeros(shape=(T, self.env.action_shape[0])),
                                       params=params, max_iter=25)
        policy = self.create_policy(gains, xbar, ubar)

        lqgspec = lqg.make_approx(p=self.env, params=params)(xbar, ubar)
        K = kf.forward(kf.KFSpec(A=lqgspec.A, H=lqgspec.H, V=lqgspec.V, W=lqgspec.W), Sigma0=self.Sigma0)

        joint_dynamics = create_joint_dynamics(self.env, K)

        return policy, joint_dynamics


class InverseGILQG(InverseILQG):
    def __init__(self, env: Env, b0: Belief, maxent_temp: float = 0.):
        super().__init__(env, b0, solve=gilqr.solve, maxent_temp=maxent_temp)


def create_joint_dynamics(p: Env, K: jnp.ndarray) -> Callable:
    f = p._dynamics
    h = p._observation

    def joint_dynamics(t, x, xhat, state_noise, obs_noise, policy_noise, policy, params):
        u = policy(t, xhat, policy_noise)
        x_next = f(x, u, state_noise, params)
        xhat_next = f(xhat, u, jnp.zeros_like(state_noise), params) + K[t] @ (
                h(x, obs_noise, params) - h(xhat, jnp.zeros_like(obs_noise), params))

        return jnp.hstack((x_next, xhat_next))

    return joint_dynamics


class FixedLinearizationInverseILQG(InverseILQG):
    def __init__(self, env: Env, b0: Belief, maxent_temp: float = 0.):
        super().__init__(env, b0, solve=ilqg_fixed.solve, maxent_temp=0.)

        if maxent_temp > 0:
            self.create_policy = lambda gains, xbar, ubar: create_maxent_lqr_policy(gains, xbar, ubar, maxent_temp)
        else:
            self.create_policy = create_lqr_policy

    def apply_solver(self, x: jnp.ndarray, params: Any) -> Tuple[Callable, Callable]:
        # get policy for current params
        K, gains, xbar, ubar = self.solve(self.env, X=x, U=estimate_controls(x, self.env, params),
                                          Sigma0=self.Sigma0, params=params)
        policy = self.create_policy(gains, xbar, ubar)
        joint_dynamics = create_joint_dynamics(self.env, K)

        return joint_dynamics, policy

    def loglikelihood(self, x: jnp.ndarray, params: Any):
        mu, Sigma = vmap(lambda xi: self.moments(xi, *self.apply_solver(xi, params), params))(x)

        d = self.xdim
        return jnp.sum(js.stats.multivariate_normal.logpdf(x[:, 1:],
                                                           mu[:, :, :d],
                                                           Sigma[:, :, :d, :d] + jnp.eye(d) * 1e-6))


class FixedLinearizationInverseGILQG(FixedLinearizationInverseILQG):
    def __init__(self, env: Env, b0: Belief, maxent_temp: float = 0.):
        super().__init__(env, b0, maxent_temp=maxent_temp)

    def apply_solver(self, x: jnp.ndarray, params: Any) -> Tuple[Callable, Callable]:
        # get policy for current params
        K, gains, xbar, ubar = self.solve(self.env, X=x, U=estimate_controls(x, self.env, params), params=params,
                                          Sigma0=self.Sigma0,
                                          lqg_module=glqg)
        policy = self.create_policy(gains, xbar, ubar)
        joint_dynamics = create_joint_dynamics(self.env, K)

        return joint_dynamics, policy