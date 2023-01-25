from jaxopt._src.base import Solver


def run_with_restarts(optim: Solver, init_params, *args, **kwargs):
    results = []
    for i, x0 in enumerate(init_params):
        result = optim.run(init_params=x0, *args, **kwargs)

        results.append(result)

    # find the result with the best log likelihood
    min_idx = min(enumerate(results), key=lambda x: x[1].state.fun_val)[0]
    result = results[min_idx]

    return result
