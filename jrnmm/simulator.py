import chex
import jax
from einops import rearrange
from jax import numpy as jnp
from jax import random as jr
from jax.scipy.linalg import cholesky

jax.config.update("jax_enable_x64", True)


def simulate(
    rng_key: jr.PRNGKey,
    dt: float,
    t_end: float,
    initial_states: jax.Array,
    Cs: float | jax.Array = 135,
    mus: float | jax.Array = 220,
    sigmas: float | jax.Array = 2000,
    gains: float | jax.Array = 0.0,
    sigma_4: float = 0.01,
    sigma_6: float = 1.0,
    A: float = 3.25,
    B: float = 22,
    a: float = 100,
    b: float = 50,
    v0: float = 6,
    vmax: float = 5.0,
    r: float = 0.56,
) -> jax.Array:
    """Jansen-Rit neural mass model simulator.

    Simulate realizations of the stochastic Jansen-Rit model [1-3].
    The return value os the difference Y(t)^2 - Y(t)^3 in decibels.

    Args:
        rng_key: a JAX random key
        dt: resolution at which time points are saved. This is the 'saveat'
            argument in typical ODE/SDE solvers. However, for the
            Strang splitting implementation, it is both the resolution of the
            solver as well as the resolution at which time points are saved.
            The lower the better.
        t_end: end time point
        initial_states: vector or matrix of initial conditions

    Examples:
        >>> from jax import numpy as jnp
        >>> from jax import random as jr
        >>> from jrnmm import simulate
        >>>
        >>> # simulate one trajectory
        >>> initial_states = jr.normal(jr.PRNGKey(1), (6,))
        >>> y = simulate(jr.PRNGKey(2), dt=0.1, t_end=10, initial_states=initial_states)
        >>>
        >>> # simulate 10 trajectories with the same initial condition and
        >>> # different parameter values
        >>> initial_states = jr.normal(jr.PRNGKey(1), (6,))
        >>> y = simulate(
        ...     jr.PRNGKey(2),
        ...     dt=0.1,
        ...     t_end=10,
        ...     initial_states=initial_states,
        ...     Cs=jr.uniform(jr.PRNGKey(3), (10,), minval=10, maxval=250),
        ...     mus=jr.uniform(jr.PRNGKey(4), (10,), minval=50, maxval=500),
        ...     sigmas=jr.uniform(jr.PRNGKey(5), (10,), minval=100, maxval=5000),
        ...     gains=jr.uniform(jr.PRNGKey(6), (10,), minval=-20, maxval=20)
        ... )
        >>>
        >>> # simulate 10 trajectories with different initial conditions and
        >>> # different parameter values
        >>> initial_states = jr.normal(jr.PRNGKey(1), (10, 6))
        >>> y = simulate(
        ...     jr.PRNGKey(2),
        ...     dt=0.1,
        ...     t_end=10,
        ...     initial_states=initial_states,
        ...     Cs=jr.uniform(jr.PRNGKey(3), (10,), minval=10, maxval=250),
        ...     mus=jr.uniform(jr.PRNGKey(4), (10,), minval=50, maxval=500),
        ...     sigmas=jr.uniform(jr.PRNGKey(5), (10,), minval=100, maxval=5000),
        ...     gains=jr.uniform(jr.PRNGKey(6), (10,), minval=-20, maxval=20)
        ... )

    References:
       .. [1] Rodrigues, Pedro, et al., "HNPE: Leveraging global parameters for
           neural posterior estimation.",
           Advances in Neural Information Processing Systems 34, 2021
       .. [2] Ableidinger, Markus, Evelyn Buckwar, and Harald Hinterleitner.
           "A stochastic version of the Jansen and Rit neural mass model:
           Analysis and numerics."
           The Journal of Mathematical Neuroscience, 2017
       .. [3] Buckwar, Evelyn, Massimiliano Tamborrino, and Irene Tubikanec.
           "Spectral density-based and measure-preserving ABC for partially
           observed diffusion processes. An illustration on Hamiltonian SDEs."
           Statistics and Computing, 2020
    """
    initial_states, Cs, mus, sigmas, gains = _preprocess(initial_states, Cs, mus, sigmas, gains)

    n_iter = len(jnp.arange(0, t_end, dt))
    dm, cms = exp_mat(dt, a, b), cov_mats(dt, sigma_4, sigmas, sigma_6, a, b)
    C1, C2, C3, C4 = Cs, 0.8 * Cs, 0.25 * Cs, 0.25 * Cs
    Aa, BbC = A * a, B * b * C4

    @jax.jit
    def _step(states, rng_key):
        noises = jr.normal(rng_key, (states.shape[0], 6))
        new_states = states
        new_states = ode(new_states, dt / 2, Aa, mus, BbC, C1, C2, C3, vmax, v0, r)
        new_states = sde(new_states, dm, cms, noises)
        new_states = ode(new_states, dt / 2, Aa, mus, BbC, C1, C2, C3, vmax, v0, r)
        return new_states, new_states

    sampling_keys = jr.split(rng_key, n_iter)
    _, states = jax.lax.scan(_step, initial_states, sampling_keys)

    ret = states[..., [1]] - states[..., [2]]
    ret = jnp.power(10, gains.reshape(1, len(gains), 1) / 10) * ret

    return rearrange(ret, "t b l -> b t l")


def _preprocess(initial_states, *args):
    Cs, mus, sigmas, gains = (jnp.atleast_1d(arg) for arg in args)
    chex.assert_equal_shape([Cs, mus, sigmas, gains])

    initial_states = jnp.atleast_1d(initial_states)
    chex.assert_equal(initial_states.shape[-1], 6)
    initial_states = jnp.atleast_2d(initial_states).reshape(-1, 6)
    if initial_states.ndim == 1 or initial_states.shape == (1, 6):
        initial_states = jnp.tile(jnp.squeeze(initial_states), [len(Cs), 1])
        chex.assert_shape(initial_states, (len(Cs), 6))
    elif initial_states.ndim == 2 and len(Cs) == 1:
        Cs = jnp.tile(Cs, [len(Cs), 1])
        mus = jnp.tile(mus, [len(mus), 1])
        sigmas = jnp.tile(Cs, [len(sigmas), 1])
        gains = jnp.tile(Cs, [len(gains), 1])
    elif initial_states.ndim == 2 and len(Cs) > 1:
        chex.assert_equal(initial_states.shape[0], len(Cs))
    else:
        raise ValueError(
            "something is wrong with the dimensionalities. " "please, check your inputs or file a issue at GitHub."
        )

    return initial_states, Cs, mus, sigmas, gains


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
