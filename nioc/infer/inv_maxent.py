import jax.numpy as jnp
import jax.scipy.stats as jstats
from jax import vmap

from nioc.envs import Env
from nioc.control import gilqr
from nioc.control.policy import create_lqr_policy
from nioc.infer.utils import estimate_controls


class InverseMaxEntBaseline:
    def __init__(self, env: Env, maxent_temp: float = 1e-6, *args, **kwargs):
        self.env = env
        self.solve = gilqr.solve
        self.maxent_temp = maxent_temp

    def apply_solver(self, x, u, params):
        T = x.shape[1] - 1

        gains, xbar, ubar = self.solve(self.env, x0=x[:, 0].mean(axis=0),
                                       U_init=jnp.zeros(shape=(T, self.env.action_shape[0])),
                                       params=params)
        policy = create_lqr_policy(gains, xbar, ubar)

        return policy, gains

    def loglikelihood(self, x, params, u=None):
        T = x.shape[1] - 1

        if u is None:
            u = vmap(lambda xi: estimate_controls(xi, self.env, params))(x)

        policy, gains = self.apply_solver(x, u, params)

        # compute log likelihood of generated samples under the used controller
        def eval_llh(x, u):
            def eval_llh_t(t, x, u):
                u_policy = policy(t, x)
                llh = jstats.multivariate_normal.logpdf(u, u_policy, self.maxent_temp * jnp.linalg.inv(gains.D[t]))
                return llh

            llh = vmap(eval_llh_t)(jnp.arange(T), x[:-1], u)
            return llh

        llh = jnp.mean(vmap(eval_llh)(x, u))
        return llh
