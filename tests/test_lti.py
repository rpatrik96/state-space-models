from state_space_models.lti import SpringMassDamper, LTISystem
import numpy as np


def test_lti_AB(lti_AB):
    lti_AB


def test_lti_simulate(lti_AB):
    num_step = 50
    dim = 2
    U = np.random.randn(num_step, dim)
    t, y, x = lti_AB.simulate(U, dt=0.1)

    assert x.shape == (num_step, dim)
    assert y.shape == (num_step, lti_AB.C.shape[0])


def test_spring_mass():
    s = SpringMassDamper.from_params()
    m = s.controllability
