from scipy.signal import StateSpace, lsim
import numpy as np


class LTISystem(object):
    def __int__(self, A, B, C=0, D=None, dt=None):
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
        self.dt = dt
        self.ss = StateSpace(A, B, C, D, dt)

    def simulate(self, U, initial_state=None, dt=None):
        if dt is None and self.dt is None:
            raise ValueError(f"Simulation requires time step, None supplied")

        if initial_state is not None and initial_state.shape != (self.A.shape[0], 1):
            raise ValueError(
                f"Initial state should have dimensions {(self.A.shape[0],1)}, got {initial_state.shape}!"
            )

        t = np.arange(0, len(U) * dt, dt)

        t, y, x = lsim(self.ss, U, t, initial_state)

        return t, y, x


class SpringMassDamper(LTISystem):
    def __init__(self, A, B, C, D):
        super().__init__(A, B, C, D, None)

    @classmethod
    def from_params(cls, damping=4, spring_stiffness=2, mass=20):
        # System matrices
        A = [[0, 1], [-spring_stiffness / mass, -damping / mass]]
        B = [[0], [1 / mass]]
        C = [[1, 0]]
        D = 0
        return cls(A, B, C, D)
