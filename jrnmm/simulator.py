import chex
import jax

jax.config.update("jax_enable_x64", True)

import jax.lax
from einops import rearrange
from jax import random as jr, numpy as jnp
from jax.scipy.linalg import cholesky


def simulate(
    rng_key,
    dt,
    T_end,
    initial_states,
    Cs,
    mus,
    sigmas,
    gains,
    sigma_4=0.01,
    sigma_6=1.0,
    A=3.25,
    B=22,
    a=100,
    b=50,
    v0=6,
    vmax=5.0,
    r=0.56,
):
    chex.assert_equal_shape([Cs, mus, sigmas, gains])
    n_iter = len(jnp.arange(0, T_end, dt))
    dm = exp_mat(dt, a, b)
    cms = cov_mats(dt, sigma_4, sigmas, sigma_6, a, b)
    C1 = Cs
    C2 = 0.8 * C1
    C3 = 0.25 * C1
    C4 = C3
    Aa = A * a
    BbC = B * b * C4

    @jax.jit
    def _step(states, rng_key):
        noises = jr.normal(rng_key, (states.shape[0], 6))
        new_states = states
        new_states = ode(new_states, dt / 2, Aa, mus, BbC, C1, C2, C3, vmax, v0, r)
        new_states = sde(new_states, dm, cms, noises)
        new_states = ode(new_states, dt / 2, Aa, mus, BbC, C1, C2, C3, vmax, v0, r)
        return new_states, new_states

    sampling_keys = jr.split(rng_key, n_iter)
    if initial_states.ndim == 1:
        initial_states = jnp.tile(initial_states, [len(Cs), 1])
    chex.assert_shape(initial_states, (len(Cs), 6))
    _, states = jax.lax.scan(_step, initial_states, sampling_keys)

    ret = states[..., [1]] - states[..., [2]]
    ret = jnp.power(10, gains.reshape(1, len(gains), 1) / 10) * ret

    return rearrange(ret, "t b l -> b t l")


def ode(states, dt, Aa, mu, BbC, C1, C2, C3, vmax, v0, r):
    rows = jnp.zeros_like(states)
    rows = rows.at[:, 3].set(Aa * sigmoid(states[:, 1] - states[:, 2], vmax, v0, r))
    rows = rows.at[:, 4].set(Aa * (mu + C2 * sigmoid(C1 * states[:, 0], vmax, v0, r)))
    rows = rows.at[:, 5].set(BbC * sigmoid(C3 * states[:, 0], vmax, v0, r))
    ret = states + dt * rows
    return ret


def sde(states, dm, cms, noises):
    return mult_dm(dm, states) + mult_cm(cms, noises)


def mult_dm(mat, vecs):
    idx012345 = jnp.arange(6)
    idx345012 = jnp.roll(jnp.arange(6), 3)
    ret = vecs * jnp.diag(mat) + vecs[:, idx345012] * mat[idx012345, idx345012]
    return ret


def mult_cm(mats, vecs):
    def map(mat, vec):
        ret = vec * jnp.diag(mat)
        ret = ret.at[3].set(ret[3] + mat[3, 0] * vec[0])
        ret = ret.at[4].set(ret[4] + mat[4, 1] * vec[1])
        ret = ret.at[5].set(ret[5] + mat[5, 2] * vec[2])
        return ret

    ret = jax.vmap(map)(mats, vecs)
    return ret


def exp_mat(t, a, b):
    eat = jnp.exp(-a * t)
    eatt = jnp.exp(-a * t) * t
    ebt = jnp.exp(-b * t)
    ebtt = jnp.exp(-b * t) * t
    ret = jnp.diag(
        jnp.array(
            [
                eat + a * eatt,
                eat + a * eatt,
                ebt + b * ebtt,
                eat - a * eatt,
                eat - a * eatt,
                ebt - b * ebtt,
            ]
        )
    )

    ret = ret.at[0, 3].set(eatt)
    ret = ret.at[1, 4].set(eatt)
    ret = ret.at[2, 5].set(ebtt)

    ret = ret.at[3, 0].set(-(a**2) * eatt)
    ret = ret.at[4, 1].set(-(a**2) * eatt)
    ret = ret.at[5, 2].set(-(b**2) * ebtt)
    return ret


def cov_mats(t, sigma_4, sigmas, sigma_6, a, b):
    em2at = jnp.exp(-2 * a * t)
    em2bt = jnp.exp(-2 * b * t)
    e2at = jnp.exp(2 * a * t)
    e2bt = jnp.exp(2 * b * t)

    def cov(sigma):
        sigma = jnp.array([0.0, 0.0, 0.0, sigma_4, sigma, sigma_6])
        sigma = sigma**2
        ret = jnp.diag(
            jnp.array(
                [
                    em2at * (e2at - 1 - 2 * a * t * (1 + a * t)) * sigma[3] / (4 * a**3),
                    em2at * (e2at - 1 - 2 * a * t * (1 + a * t)) * sigma[4] / (4 * a**3),
                    em2bt * (e2bt - 1 - 2 * b * t * (1 + b * t)) * sigma[5] / (4 * b**3),
                    em2at * (e2at - 1 - 2 * a * t * (a * t - 1)) * sigma[3] / (4 * a),
                    em2at * (e2at - 1 - 2 * a * t * (a * t - 1)) * sigma[4] / (4 * a),
                    em2bt * (e2bt - 1 - 2 * b * t * (b * t - 1)) * sigma[5] / (4 * b),
                ]
            )
        )
        ret = ret.at[0, 3].set(em2at * t**2 * sigma[3] / 2)
        ret = ret.at[1, 4].set(em2at * t**2 * sigma[4] / 2)
        ret = ret.at[2, 5].set(em2bt * t**2 * sigma[5] / 2)
        ret = ret.at[3, 0].set(em2at * t**2 * sigma[3] / 2)
        ret = ret.at[4, 1].set(em2at * t**2 * sigma[4] / 2)
        ret = ret.at[5, 2].set(em2bt * t**2 * sigma[5] / 2)
        return ret

    ret = jax.vmap(cov)(sigmas)
    ret = jax.vmap(lambda x: cholesky(x))(ret)
    ret = jax.vmap(jnp.transpose)(ret)
    return ret


def sigmoid(x, vmax, v0, r):
    return vmax / (1 + jnp.exp(r * (v0 - x)))
