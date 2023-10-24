from state_space_models.lti import SpringMassDamper, LTISystem, DCMotor
import numpy as np
import pytest


def test_lti_AB(lti_AB):
    lti_AB


def test_lti_simulate(lti_AB):
    num_step = 50
    dim = 2
    U = np.random.randn(num_step, dim)
    t, y, x = lti_AB.simulate(U)

    assert x.shape == (num_step, dim)
    assert y.shape == (num_step, lti_AB.C.shape[0])


@pytest.mark.parametrize("triangular", [True, False])
def test_lti_controllable_system(triangular):
    lti = LTISystem.controllable_system(2, 2, triangular=triangular)


@pytest.mark.parametrize("cls", [SpringMassDamper, DCMotor])
def test_custom_lti_obsv_ctrb(cls):
    s = cls.from_params()

    assert s.A.shape[0] == s.A.shape[1]
    state_dim = s.A.shape[0]

    assert np.linalg.matrix_rank(s.controllability) == state_dim
    assert np.linalg.matrix_rank(s.observability) == state_dim
