import jax.numpy as jnp
from jax import random, vmap

from nioc.control.spec import LQGSpec
from nioc.control import lqg


def init_linear_reaching_task():
    state_dim = 6
    control_dim = 1

    # params
    dt = 0.01
    r = 1e-5
    tau = 0.04

    A = jnp.array([[1., dt, 0, 0., 0., 0.],
                   [0., 1., dt, 0., 0., 0.],
                   [0., 0., 1 - dt / tau, dt / tau, 0., 0.],
                   [0., 0., 0., 1 - dt / tau, 0., 0.],
                   [0., 0., 0., 0., 1., 0.],
                   [0., 0., 0., 0., 0., 1.]])
    B = jnp.array([[0.], [0.], [0.], [dt / tau], [0.], [0.]])

    d = jnp.diag(jnp.array([1., 0., jnp.sqrt(0.002), jnp.sqrt(0.0002)])) @ jnp.array([[1., 0., 0., 0., -1., 0.],
                                                                                      [1., 0., 0., 0., 0., -1],
                                                                                      [0., 1., 0., 0., 0., 0.],
                                                                                      [0., 0., 1., 0., 0., 0.]])

    Qf = d.T @ d
    R = jnp.eye(control_dim) * r

    F = jnp.eye(3, 6)
    V = jnp.diag(jnp.array([1., 1., 1., 1., 0., 0.])) * 1e-2
    W = jnp.eye(3) * 1e-2

    T = 100

    problem = LQGSpec(A=jnp.stack([A] * T), B=jnp.stack([B] * T), V=jnp.stack(T * (V,)),
                      F=jnp.stack(T * (F,)), W=jnp.stack(T * (W,)),
                      Q=jnp.zeros((T, state_dim, state_dim)),
                      R=jnp.stack([R] * T),
                      P=jnp.zeros((T, control_dim, state_dim)),
                      Qf=Qf,
                      q=jnp.zeros((T, state_dim)), qf=jnp.zeros((state_dim,)),
                      r=jnp.zeros((T, control_dim)), Cu=None, Cx=None, D=None
                      )

    return problem


def test_lqg():
    problem = init_linear_reaching_task()

    x0 = jnp.array([0., 0., 0., 0., 1., -1.])
    X, U = vmap(lambda key: lqg.simulate(key, problem, x0))(random.split(random.PRNGKey(0), 50))

    # check that there are no NaNs
    assert not jnp.isnan(X).any()
