from __future__ import annotations

from typing import Optional

import numpy as np
from control import ctrb, obsv
from scipy.signal import StateSpace, dlsim, cont2discrete


class LTISystem(object):
    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        C: Optional[np.ndarray] = None,
        D: Optional[np.ndarray] = None,
        dt: float = 1e-3,
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
        self.dt: float = dt
        self.ss: StateSpace = StateSpace(self.A, self.B, self.C, self.D, dt=self.dt)

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
        else:
            if dt != self.dt:
                raise ValueError(
                    f"Simulation requires time step {self.dt}, got {dt} instead"
                )

        if initial_state is not None and initial_state.shape != (self.A.shape[0], 1):
            raise ValueError(
                f"Initial state should have dimensions {(self.A.shape[0], 1)}, got {initial_state.shape}!"
            )

        t = np.arange(0, len(U) * dt, dt)

        t, y, x = dlsim(self.ss, U, t, initial_state)

        return t, y, x

    @property
    def controllability(self) -> np.ndarray:
        return ctrb(self.A, self.B)

    @property
    def observability(self) -> np.ndarray:
        return obsv(self.A, self.C)

    @classmethod
    def controllable_system(
        cls, state_dim, control_dim, triangular=False, dt=0.01, use_B=True, use_C=False
    ):
        num_attempt = 0

        while True:
            A = np.random.randn(state_dim, state_dim)

            if triangular is True:
                A = np.tril(A)

            # rescale A such that all eigenvalues are < 1
            A = A / np.max(np.abs(np.linalg.eigvals(A))) * 0.9

            print(np.abs(np.linalg.eigvals(A)))

            if use_B is True:
                B = np.random.randn(state_dim, control_dim)
            else:
                B = np.eye(state_dim)

            if np.linalg.matrix_rank(ctrb(A, B)) == state_dim:
                break

            num_attempt += 1
            if num_attempt > 100:
                raise ValueError("Could not find a controllable system!")

        if use_C is True:
            while True:
                C = np.random.randn(state_dim, state_dim)

                if np.linalg.matrix_rank(obsv(A, C)) == state_dim:
                    break

                num_attempt += 1
                if num_attempt > 100:
                    raise ValueError("Could not find an observable system!")
        else:
            C = None

        D = None

        return cls(A, B, C, D, dt=dt)


class SpringMassDamper(LTISystem):
    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        C: Optional[np.ndarray] = None,
        D: Optional[np.ndarray] = None,
        dt: float = 1e-3,
    ):
        super().__init__(A, B, C, D, dt)

    @classmethod
    def from_params(
        cls, damping=4, spring_stiffness=2, mass=20, dt=1e-3
    ) -> SpringMassDamper:
        A: np.ndarray = np.array([[0, 1], [-spring_stiffness / mass, -damping / mass]])
        B: np.ndarray = np.array([[0], [1 / mass]])
        C: np.ndarray = np.array([[1, 0]])
        D: np.ndarray = np.array(0)
        discrete_system = cont2discrete((A, B, C, D), dt=dt)
        return cls(*discrete_system)


class DCMotor(LTISystem):
    """Source: https://ctms.engin.umich.edu/CTMS/index.php?example=MotorSpeed&section=SystemModeling
    The C matrix is modified to be the identity matrix, so that the output is the state itself.
    """

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        C: Optional[np.ndarray] = None,
        D: Optional[np.ndarray] = None,
        dt: float = 1e-3,
    ):
        super().__init__(A, B, C, D, dt)

    @classmethod
    def from_params(
        cls,
        armature_resistance=1,
        armature_inductance=0.5,
        electromotive_force_constant=0.01,
        rotor_inertia=0.01,
        rotor_damping=0.1,
        dt=1e-3,
    ) -> DCMotor:
        A: np.ndarray = np.array(
            [
                [
                    -armature_resistance / armature_inductance,
                    electromotive_force_constant / armature_inductance,
                ],
                [
                    -electromotive_force_constant / rotor_inertia,
                    -rotor_damping / rotor_inertia,
                ],
            ]
        )
        B: np.ndarray = np.array([[1 / armature_inductance], [0]])
        C: np.ndarray = np.eye(2)
        D: np.ndarray = np.zeros_like(B)

        discrete_system = cont2discrete((A, B, C, D), dt=dt)
        return cls(*discrete_system)
