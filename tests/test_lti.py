from state_space_models.lti import SpringMassDamper, LTISystem
import numpy as np


def test_lti_AB():
    dim = 2
    A = np.random.randn(dim, dim)
    B = np.random.randn(dim, dim)
    lti = LTISystem(A=A, B=B)


def test_spring_mass():
    s = SpringMassDamper.from_params()
    m = s.controllability
