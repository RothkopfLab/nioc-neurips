from jax import random, numpy as jnp

from nioc.control.spec import make_lqr_approx, make_lqg_approx
from envs import RandomLinearProblem


def test_linearizer_random():
    """ Make sure that the linearizer applied to a linear system actually returns the original system
    """
    rp = RandomLinearProblem(random.PRNGKey(0))
    linearizer = make_lqr_approx(rp, None)

    T = 31

    p = linearizer(jnp.zeros((T, 5,)), jnp.zeros((T - 1, 2,)))

    assert (p.A == rp.A).all() and (p.B == rp.B).all()


def test_glqr_linearizer():
    """ Make sure that the linearized system with signal-dependent noise returns correct shapes for the reaching problem
    """
    rp = RandomLinearProblem(random.PRNGKey(0))
    linearizer = make_lqg_approx(rp, None)

    T = 31

    p = linearizer(jnp.zeros((T, rp.state_shape[0],)), jnp.zeros((T - 1, rp.action_shape[0],)))

    assert (p.V.shape == (T - 1, rp.state_shape[0], rp.state_noise_shape[0]))
    assert (p.Cu.shape == (T - 1, rp.state_shape[0], rp.state_noise_shape[0], rp.action_shape[0]))
    assert (p.Cx.shape == (T - 1, rp.state_shape[0], rp.state_noise_shape[0], rp.state_shape[0]))
