import jax
from jax import vmap, numpy as jnp, random
import jaxopt
from nioc.envs import Env
from nioc.infer.optim import run_with_restarts


def estimate_controls(x: jnp.array, env: Env, params: dict, maxiter: int = 100) -> jnp.ndarray:
    def estimate_control(x0, x1):
        def objective(control, x0, x1, theta):
            x1e = env._dynamics(x0, control, jnp.zeros(env.state_noise_shape), theta)
            return (x1e - x1)

        u_dim = env.action_shape
        solver = jaxopt.GaussNewton(residual_fun=objective, maxiter=maxiter)
        res = solver.run(jnp.zeros(u_dim), x0, x1, params)
        u_opt, state = res
        return u_opt

    u = vmap(estimate_control)(x[:-1], x[1:])
    return u


def draw_random_uniform_in_log_space(key, lo, hi, params_type):
    return jax.tree_map(lambda key, lo, hi: random.uniform(key, minval=jnp.log10(lo), maxval=jnp.log10(hi)),
                        params_type(*random.split(key, len(params_type._fields))), lo, hi)

def compute_mle(xs, ioc, key, restarts, bounds, optim="L-BFGS-B",
                params=None,
                likelihood_params=None):
    if likelihood_params is None:
        likelihood_params = {}

    ParamsType = ioc.env.get_params_type()

    # if we do not tell the function which parameters to infer, we infer all of them
    if not params:
        params = ParamsType._fields

    # minimizer for negative log likelihood, do optimization in log space
    # nll = lambda params: -ioc.loglikelihood(xs, {name: 10. ** val for name, val in params._asdict().items()})
    nll = lambda params: -ioc.loglikelihood(xs, jax.tree_map(lambda x: 10 ** x,
                                                             ParamsType(**params)), **likelihood_params)

    # define the optimizer from jaxopt
    optim = jaxopt.ScipyBoundedMinimize(fun=nll, method=optim)

    # draw random uniform parameters in log space
    key, subkey = random.split(key)
    init_params = [draw_random_uniform_in_log_space(key, *bounds, ParamsType) for key in random.split(subkey, restarts)]
    # only use initial params of those parameters that we want to infer
    init_params = [
        {param_name: param_value for param_name, param_value in p._asdict().items() if param_name in params} for p
        in init_params]
    # only use the bounds of those parameters that we want to infer
    bounds = tuple({param_name: getattr(bound, param_name) for param_name in params} for bound in bounds)

    # run inference
    result = run_with_restarts(optim, init_params=init_params, bounds=jax.tree_map(jnp.log10, bounds))

    # convert back to original space
    result_params = {name: (10. ** value)
                     for name, value in result.params.items()}

    return result, result_params
