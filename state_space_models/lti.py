from __future__ import annotations
from scipy.signal import StateSpace, lsim
import numpy as np
from typing import Optional
from control import ctrb, obsv


class LTISystem(object):
    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        C: Optional[np.ndarray] = None,
        D: Optional[np.ndarray] = None,
        dt: Optional[float] = None,
    ):
        super().__init__()

        if A.shape[0] != A.shape[1]:
            raise ValueError(
                f"the sytem matrix (A) should be square, got shape {A.shape}"
            )
        if A.shape[0] != B.shape[0]:
            raise ValueError(
                f"the control matrix (B) should have {A.shape[0]} rows, got {B.shape[0]}"
            )

        self.A: np.ndarray = A
        self.B: np.ndarray = B
        self.C: np.ndarray = C if C is not None else np.eye(A.shape[0])
        self.D: np.ndarray = D if D is not None else np.zeros_like(B)
        self.dt: Optional[float] = dt
        self.ss: StateSpace = StateSpace(self.A, self.B, self.C, self.D)

    def simulate(
        self,
        U: np.ndarray,
        initial_state: Optional[np.ndarray] = None,
        dt: Optional[float] = None,
    ):
        if dt is None:
            if self.dt is None:
                raise ValueError(f"Simulation requires time step, None supplied")
            else:
                dt = self.dt

        if initial_state is not None and initial_state.shape != (self.A.shape[0], 1):
            raise ValueError(
                f"Initial state should have dimensions {(self.A.shape[0],1)}, got {initial_state.shape}!"
            )

        t = np.arange(0, len(U) * dt, dt)

        t, y, x = lsim(self.ss, U, t, initial_state)

        return t, y, x

    @property
    def controllability(self) -> np.ndarray:
        return ctrb(self.A, self.B)

    @property
    def observability(self) -> np.ndarray:
        return obsv(self.A, self.C)

    @classmethod
    def controllable_system(cls, state_dim, control_dim, triangular=False):
        num_attempt = 0

        while True:
            A = np.random.randn(state_dim, state_dim)
            B = np.random.randn(state_dim, control_dim)
            if np.linalg.matrix_rank(ctrb(A, B)) == state_dim:
                break

            num_attempt += 1
            if num_attempt > 100:
                raise ValueError("Could not find controllable system!")

        C = None
        D = None

        return cls(A, B, C, D)


class SpringMassDamper(LTISystem):
    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        C: Optional[np.ndarray] = None,
        D: Optional[np.ndarray] = None,
    ):
        super().__init__(A, B, C, D, None)

    @classmethod
    def from_params(cls, damping=4, spring_stiffness=2, mass=20) -> SpringMassDamper:
        A: np.ndarray = np.array([[0, 1], [-spring_stiffness / mass, -damping / mass]])
        B: np.ndarray = np.array([[0], [1 / mass]])
        C: np.ndarray = np.array([[1, 0]])
        D: np.ndarray = np.array(0)
        return cls(A, B, C, D)
