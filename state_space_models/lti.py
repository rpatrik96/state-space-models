from scipy.signal import StateSpace
import numpy as np


class LTISystem(object):
    def __int__(self, A, B, C=0, D=None):
        super().__init__()

        if A.shape[0] != A.shape[1]:
            raise ValueError(
                f"the sytem matrix (A) should be square, got shape {A.shape}"
            )
        if A.shape[0] != B.shape[0]:
            raise ValueError(
                f"the control matrix (B) should have {A.shape[0]} rows, got {B.shape[0]}"
            )

        self.A = A
        self.B = B
        self.C = C if C is not None else np.eye(A.shape[0])
        self.D = D if D is not None else 0.0
        self.ss = StateSpace(A, B, C, D)


class SpringMassDamper(LTISystem):
    def __init__(self, A, B, C, D):
        super().__init__(A, B, C, D)

    @classmethod
    def from_params(cls, damping=4, spring_stiffness=2, mass=20):
        # System matrices
        A = [[0, 1], [-spring_stiffness / mass, -damping / mass]]
        B = [[0], [1 / mass]]
        C = [[1, 0]]
        D = 0
        return cls(A, B, C, D)
