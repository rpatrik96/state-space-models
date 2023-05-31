import pytest
import numpy as np
from state_space_models.lti import LTISystem


@pytest.fixture()
def lti_AB():
    dim = 2
    np.random.seed(42)
    A = np.random.randn(dim, dim)
    B = np.random.randn(dim, dim)
    dt = 0.01
    lti = LTISystem(A=A, B=B, dt=dt)

    return lti
