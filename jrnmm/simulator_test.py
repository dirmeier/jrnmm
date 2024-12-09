# pylint: skip-file

import chex
import pytest
import numpy as np
import rpy2.robjects as robjects
from jax import numpy as jnp, random as jr
from jrnmm.simulator import cov_mats, exp_mat, mult_cm, mult_dm, ode, sde
from rpy2.robjects.packages import importr

sdbmp = importr("sdbmsABC")

rset_seed = robjects.r["set.seed"]
rchol = robjects.r["chol"]
rt = robjects.r["t"]


@pytest.mark.parametrize(
    "dt, sigma4, sigma, sigma6",
    [
        (0.01, 0.01, 10, 1.0),
        (0.01, 0.001, 10, 10.0),
        (0.1, 0.01, 10, 1.0),
        (0.1, 0.001, 10, 10.0),
        (1.0, 0.01, 10, 1.0),
        (1.0, 0.001, 10, 10.0),
        (0.01, 0.01, 2000, 1.0),
        (0.1, 0.01, 2000, 1.0),
    ]
)
def test_cov(dt, sigma4, sigma, sigma6):
    jac_cov = np.array(
        cov_mats(dt, sigma4, jnp.array([sigma]), sigma6, a=100, b=50),
        copy=True)
    r_cov = rt(
        rchol(sdbmp.cov_matJR(dt, robjects.FloatVector(
            [0, 0, 0, sigma4, sigma, sigma6]), a=100, b=50))
    )
    chex.assert_trees_all_close(jac_cov[0], r_cov, atol=1e-3)


@pytest.mark.parametrize("dt", [0.01, 0.02, 0.1, 0.001])
def test_exp(dt):
    jac_exp = np.array(exp_mat(dt, a=100, b=50), copy=True)
    r_exp = sdbmp.exp_matJR(dt, a=100, b=50)
    chex.assert_trees_all_close(jac_exp, r_exp, atol=1e-3)


@pytest.mark.parametrize("key", [1, 2, 3, 4, 5])
def test_mult_cm(key):
    mats = np.array(jr.normal(jr.PRNGKey(key), (1, 6, 6)), copy=True)
    vecs = np.array(jr.normal(jr.PRNGKey(key + 5), (1, 6)), copy=True)

    jax_cm = mult_cm(mats, vecs)
    vec = robjects.vectors.FloatVector(vecs[0])
    mat = robjects.r.matrix(
        robjects.vectors.FloatVector([el for row in mats[0] for el in row]),
        nrow=6, byrow=True
    )
    r_cm = sdbmp.mv_multcm_(mat, vec)

    chex.assert_trees_all_close(jax_cm[0], r_cm, atol=1e-3)


@pytest.mark.parametrize("key", [1, 2, 3, 4, 5])
def test_mult_dm(key):
    mat = np.array(jr.normal(jr.PRNGKey(key), (6, 6)), copy=True)
    vecs = np.array(jr.normal(jr.PRNGKey(key + 5), (1, 6)), copy=True)

    jax_dm = mult_dm(mat, vecs)
    vec = robjects.vectors.FloatVector(vecs[0])
    mat = robjects.r.matrix(
        robjects.vectors.FloatVector([el for row in mat for el in row]), nrow=6,
        byrow=True
    )
    r_dm = sdbmp.mv_multdm_(mat, vec)

    chex.assert_trees_all_close(jax_dm[0], r_dm, atol=1e-3)


@pytest.mark.parametrize(
    "key, dt",
    [
        (1, 0.1),
        (2, 0.1),
        (3, 0.1),
        (1, 1.0),
        (2, 1.0),
        (3, 1.0),
    ]
)
def test_ode(key, dt):
    states = np.array(jr.normal(jr.PRNGKey(key), (2, 6)), copy=True)
    state = robjects.vectors.FloatVector(states[0])
    jax_ode = ode(
        states,
        dt,
        Aa=3.25,
        mu=22,
        BbC=100,
        C1=1,
        C2=0.8,
        C3=0.25,
        vmax=5.0,
        v0=6,
        r=0.56
    )
    r_ode = sdbmp.ODE_Cpp_(
        state,
        dt,
        3.25,
        22,
        100,
        1,
        0.8,
        0.25,
        5.0,
        6,
        0.56
    )
    chex.assert_trees_all_close(jax_ode[0], r_ode, atol=1e-3)


@pytest.mark.parametrize(
    "key, dt",
    [
        (1, 0.1),
        (2, 0.1),
        (3, 0.1),
        (1, 1.0),
        (2, 1.0),
        (3, 1.0),
    ]
)
def test_sde(key, dt):
    states = np.array(jr.normal(jr.PRNGKey(key), (1, 6)), copy=True)
    dm = np.array(jr.normal(jr.PRNGKey(key + 1), (6, 6)), copy=True)
    cms = np.array(jr.normal(jr.PRNGKey(key + 2), (1, 6, 6)), copy=True)
    noises = jr.normal(jr.PRNGKey(key + 3), states.shape)
    jax_sde = sde(states, dm, cms, noises)

    state = robjects.vectors.FloatVector(states[0])
    dm = robjects.r.matrix(
        robjects.vectors.FloatVector([el for row in dm for el in row]), nrow=6,
        byrow=True
    )
    cm = robjects.r.matrix(
        robjects.vectors.FloatVector([el for row in cms[0] for el in row]), nrow=6,
        byrow=True
    )
    noise = robjects.vectors.FloatVector(noises[0])
    r_sde = sdbmp.SDE_Cpp_(state, dm, cm, noise)
    chex.assert_trees_all_close(jax_sde[0], r_sde, atol=1e-3)