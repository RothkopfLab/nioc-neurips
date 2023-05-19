from typing import Any, Tuple
from jax import lax, numpy as jnp
import jaxopt

from nioc import Env
from nioc.control import lqr
from nioc.control.lqr import Gains
from nioc.control.spec import LQGSpec, make_lqg_approx
from nioc.utils import quadratic_form, bilinear_form, bilinear_form_t, quadratic_form_t


def backward(spec: LQGSpec, K: jnp.ndarray, eps: float = 1e-8) -> lqr.Gains:
    def loop(carry, step):
        Sx, Sb, Sxb, sx, sb = carry

        Q, q, P, R, r, A, B, V, Cx, Cu, F, W, D, K = step

        # # eqn 5.23
        H = R + B.T @ (Sx + Sb + 2. * Sxb) @ B + quadratic_form(Cu, Sx).sum(axis=0)
        # # eqn 5.24
        g = r + B.T @ (sx + sb) + bilinear_form(Cu, Sx, V).sum(axis=0)
        # # eqn 5.25
        Gx = P + B.T @ (Sx + Sxb) @ A + B.T @ (Sb + Sxb) @ K @ F + bilinear_form(Cu, Sx, Cx).sum(axis=0)
        # # eqn 5.26
        Gb = B.T @ (Sb + Sxb) @ (A - K @ F)


        G = Gx + Gb

        # Deal with negative eigenvals of H, see section 5.4.1 of Li's PhD thesis
        evals, _ = jnp.linalg.eigh(H)
        Ht = H + jnp.maximum(0., eps - evals[0]) * jnp.eye(H.shape[0])
        # Ht = evecs @ jnp.diag(evals + eps) @ evecs.T

        L = -jnp.linalg.solve(Ht, G)
        l = -jnp.linalg.solve(Ht, g)

        # Sxn = Q + A.T @ Sx @ A + L.T @ H @ L + L.T @ G + G.T @ L + quadratic_form(Cx, Sx).sum(axis=0)
        # sxn = q + A.T @ sx + G.T @ l + L.T @ H @ l + L.T @ g + bilinear_form(Cx, Sx, V).sum(axis=0)

        # eqn 5.17
        Sxn = Q + A.T @ Sx @ A + F.T @ K.T @ Sb @ K @ F + 2. * A.T @ Sxb @ K @ F
        Sxn += quadratic_form(Cx, Sx).sum(axis=0) + quadratic_form(D, K.T @ Sb @ K).sum(axis=0)

        # eqn 5.18
        Sbn = (A - K @ F).T @ Sb @ (A - K @ F) + L.T @ H @ L + L.T @ Gb + Gb.T @ L

        # eqn 5.19
        Sxbn = F.T @ K.T @ Sb @ (A - K @ F) + A.T @ Sxb @ (A - K @ F) + Gx.T @ L

        # eqn 5.20
        sxn = q + A.T @ sx + F.T @ K.T @ sb + Gx.T @ l
        sxn += bilinear_form(Cx, Sx, V).sum(axis=0) + bilinear_form(D, K.T @ Sb @ K, W).sum(axis=0)

        # eqn 5.21
        sbn = (A - K @ F).T @ sb + L.T @ H @ l + L.T @ g + Gb.T @ l

        return (Sxn, Sbn, Sxbn, sxn, sbn), (L, l, Ht)

    _, (L, l, H) = lax.scan(loop,
                            (spec.Qf, jnp.zeros_like(spec.Qf), jnp.zeros_like(spec.Qf),  # final cost
                             spec.qf, jnp.zeros_like(spec.qf)),
                            (spec.Q, spec.q, spec.P, spec.R, spec.r,  # cost function
                             spec.A, spec.B, spec.V, spec.Cx, spec.Cu,  # dynamics model
                             spec.F, spec.W, spec.D, K),  # observation model
                            reverse=True)

    return lqr.Gains(L=L, l=l, H=H)


def rollout(p: Env, X: jnp.array, U: jnp.array, gains: Gains, params: Any) -> Tuple[jnp.ndarray, jnp.ndarray]:
    x0 = X[0]

    def fwd(x, step):
        L, l, xbar, ubar = step

        u = L @ (x - xbar) + l + ubar

        x = p._dynamics(x, u, jnp.zeros(p.state_noise_shape), params)

        return x, (x, u)

    _, (X, U) = lax.scan(fwd, x0, (gains.L, gains.l, X[:-1], U))

    return jnp.vstack([x0, X]), U


def forward(spec: LQGSpec, gains: Gains, xhat0: jnp.ndarray, Sigma0: jnp.ndarray) -> jnp.ndarray:
    def loop(carry, step):
        mb, me, Sigma_b, Sigma_e, Sigma_be = carry
        A, B, V, Cx, Cu, F, W, D, L, l = step

        # make the vectors shape (d, 1) instead of (d,)
        d = W[..., None]
        c = V[..., None]
        li = l[:, None]

        # eqn 5.61 (skipping the Du terms because we don't have control-dependent observation noise)
        P = bilinear_form_t(d, (mb + me)[:, None].T, D).sum(axis=0)
        P += bilinear_form_t(D, (mb + me)[:, None], d).sum(axis=0)
        P += quadratic_form_t(d, jnp.eye(1)).sum(axis=0)
        P += quadratic_form_t(D, Sigma_b + Sigma_be + Sigma_e).sum(axis=0)

        # eqn. 5.62
        M = bilinear_form_t(c, (mb + me)[:, None].T, Cx).sum(axis=0)
        M += bilinear_form_t(Cx, (mb + me)[:, None], c).sum(axis=0)
        M += bilinear_form_t(c, (l + L @ mb)[:, None].T, Cu).sum(axis=0)
        M += bilinear_form_t(Cu, (l + L @ mb)[:, None], c).sum(axis=0)
        M += quadratic_form_t(c, jnp.eye(1)).sum(axis=0)
        M += bilinear_form_t(Cx, ((mb + me)[:, None] @ li.T + (Sigma_b + Sigma_be.T) @ L.T), Cu).sum(axis=0)
        M += bilinear_form_t(Cu, (li @ (mb + me)[:, None].T + L @ (Sigma_b + Sigma_be)), Cx).sum(axis=0)
        M += quadratic_form_t(Cx, Sigma_b + Sigma_be + Sigma_be.T + Sigma_e).sum(axis=0)
        M += quadratic_form_t(Cu,
                              li @ li.T + li @ mb[:, None].T @ L.T + L @ mb[:, None] @ li.T + L @ Sigma_b @ L.T).sum(
            axis=0)

        # eqn 5.55 (filter gain)
        K = A @ Sigma_e @ F.T @ jnp.linalg.inv(F @ Sigma_e @ F.T + P)

        # update means
        # eqn 5.56
        mbn = (A + B @ L) @ mb + K @ F @ me + B @ l
        # eqn 5.57
        men = (A - K @ F) @ me

        # update covariances
        # eqn 5.58
        Sigma_bn = (A + B @ L) @ Sigma_b @ (A @ B @ L).T + K @ F @ Sigma_e @ A.T
        Sigma_bn += (A + B @ L) @ Sigma_be @ F.T @ K.T + K @ F @ Sigma_be.T @ (A + B @ L).T
        Sigma_bn += ((A + B @ L) @ mb + K @ F @ me)[:, None] @ li.T @ B.T
        Sigma_bn += B @ li @ ((A + B @ L) @ mb + K @ F @ me)[:, None].T
        Sigma_bn += B @ li @ li.T @ B.T
        # eqn 5.59
        Sigma_en = (A - K @ F) @ Sigma_e @ A.T + M
        # eqn 5.60
        Sigma_ben = (A + B @ L) @ Sigma_be @ (A - K @ F).T + B @ li @ me[:, None].T @ (A - K @ F).T

        return (mbn, men, Sigma_bn, Sigma_en, Sigma_ben), K

    _, K = lax.scan(loop,
                    # initialization
                    (xhat0, jnp.zeros_like(xhat0), xhat0[:, None] @ xhat0[:, None].T, Sigma0, jnp.zeros_like(Sigma0)),
                    (spec.A, spec.B, spec.V, spec.Cx, spec.Cu,  # dynamics model
                     spec.F, spec.W, spec.D,  # observation model
                     gains.L, gains.l))  # current control law

    return K


def solve(p: Env,
          x0: jnp.array,
          U_init: jnp.array,
          params: Any,
          Sigma0: jnp.array = None,
          max_iter=10, tol=1e-6) -> Tuple[Gains, jnp.ndarray, jnp.ndarray]:
    if Sigma0 is None:
        Sigma0 = jnp.eye(x0.shape[0])

    T = U_init.shape[0]
    gains_init = Gains(L=jnp.zeros((T, p.action_shape[0], p.state_shape[0])),
                       l=jnp.zeros((T, p.action_shape[0])),
                       H=jnp.zeros((T, p.action_shape[0], p.action_shape[0])))

    X_init = jnp.vstack([x0, jnp.zeros((T, p.state_shape[0]))])

    def fixed_point_fun(carry, params):
        gains, X, U, cost = carry

        X, U = rollout(p, X, U, gains, params)
        cost = p.trajectory_cost(X, U, params)

        spec = make_lqg_approx(p, params)(X, U)
        K = forward(spec, gains=gains, xhat0=x0, Sigma0=Sigma0)
        gains = backward(spec, K=K)

        # TODO: try out jaxopt line search instead
        def cond_fn(eps):
            X_new, U_new = rollout(p, X, U, Gains(gains.L, eps * gains.l, gains.H), params)
            return (p.trajectory_cost(X_new, U_new, params) > cost) & (eps > tol)

        def body_fn(eps):
            return eps / 2.

        eps = lax.while_loop(cond_fn, body_fn, init_val=1.)

        gains = Gains(L=gains.L, l=eps * gains.l, H=gains.H)

        return (gains, X, U, cost)

    fpi = jaxopt.FixedPointIteration(fixed_point_fun=fixed_point_fun, maxiter=max_iter, implicit_diff=True)
    x_init = (gains_init, X_init, U_init, jnp.inf)
    res = fpi.run(x_init, params)
    gains, X, U, cost = res.params
    X, U = rollout(p, X, U, gains, params)

    return gains, X, U
