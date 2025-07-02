from typing import List, Optional, Tuple

import cvxpy as cvx
import numpy as np

from SCvx.global_parameters import K

from .base_model import BaseModel


class SingleIntegratorModel(BaseModel):
    """
    Single-integrator dynamics with obstacle constraints for 3D trajectory planning.

    State: x = [x, y, z]^T
    Input: u = [v_x, v_y, v_z]^T (velocity commands)

    Obstacles: list of (center, radius) tuples
    """

    n_x = 3
    n_u = 3

    def __init__(
        self,
        r_init: np.ndarray = np.array([-8.0, -8.0, -8.0]),
        r_final: np.ndarray = np.array([8.0, 8.0, 8.0]),
        v_max: float = 1.0,
        bounds: Tuple[float, float] = (-10.0, 10.0),
        robot_radius: float = 0.5,
        obstacles: Optional[List[Tuple[List[float], float]]] = None,
    ):
        super().__init__()
        # Initial and final states (3D positions)
        self.x_init = r_init.reshape(-1)
        self.x_final = r_final.reshape(-1)

        # Velocity limit and workspace bounds
        self.v_max = v_max
        self.lower_bound, self.upper_bound = bounds
        self.robot_radius = robot_radius

        # Obstacles: list of (center: List[float], radius: float)
        self.obstacles = (
            obstacles
            if obstacles is not None
            else [([-5.0, -4.0, -5.0], 2.0), ([0.0, 0.0, 4.0], 2.0)]
        )
        # Slack variables for linearized obstacle avoidance
        self.s_prime = [cvx.Variable((K, 1), nonneg=True) for _ in self.obstacles]

        # Dynamics for single integrator: f(x,u) = u
        self.f = lambda x, u: u
        # Jacobians: A = 0, B = I
        self.A = lambda x, u: np.zeros((self.n_x, self.n_x))
        self.B = lambda x, u: np.eye(self.n_x)

    def get_equations(self) -> Tuple:
        """
        Return the dynamics function and its Jacobians.
        """
        return self.f, self.A, self.B

    def initialize_trajectory(self, X: np.ndarray, U: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initialize positions along a straight line in 3D and zero velocities.
        """
        K_local = X.shape[1]
        for k in range(K_local):
            alpha1 = (K_local - 1 - k) / (K_local - 1)
            alpha2 = k / (K_local - 1)
            # Interpolate linearly in 3D
            X[:, k] = alpha1 * self.x_init + alpha2 * self.x_final
        # Zero initial controls
        U[:] = 0
        return X, U

    def get_constraints(
        self,
        X: cvx.Variable,
        U: cvx.Variable,
        X_ref: cvx.Parameter,
        U_ref: cvx.Parameter,  # noqa: ARG002
    ) -> List[cvx.Constraint]:
        """
        Construct model-specific constraints:
        - boundary conditions
        - velocity bounds
        - workspace bounds
        - linearized obstacle avoidance with slack
        """
        constraints = []
        # Boundary conditions: start and end positions, zero start/end velocities
        constraints += [
            X[:, 0] == self.x_init,
            X[:, -1] == self.x_final,
            U[:, 0] == 0,
            U[:, -1] == 0,
        ]

        # Velocity bounds (Euclidean norm per timestep)
        for k in range(K):
            constraints.append(cvx.norm(U[:, k], 2) <= self.v_max)

        # Workspace bounds (positions)
        constraints += [
            X[0:3, :] <= self.upper_bound - self.robot_radius,
            X[0:3, :] >= self.lower_bound + self.robot_radius,
        ]

        # Linearized spherical obstacle constraints
        for j, (p, r) in enumerate(self.obstacles):
            p_vec = np.array(p).reshape(
                3,
            )
            r_total = r + self.robot_radius
            for k in range(K):
                x_ref_k = X_ref[0:3, k]
                diff = x_ref_k - p_vec
                norm_ref = cvx.norm(diff, 2) + 1e-6
                a_k = diff / norm_ref
                lhs = a_k.T @ (X[0:3, k] - p_vec)
                constraints.append(lhs >= r_total - self.s_prime[j][k, 0])

        return constraints

    def get_objective(
        self,
        X: cvx.Variable,  # noqa: ARG002
        U: cvx.Variable,  # noqa: ARG002
        X_ref: cvx.Parameter,  # noqa: ARG002
        U_ref: cvx.Parameter,  # noqa: ARG002
    ):
        """
        Model-specific objective: minimize obstacle slack.
        """
        slack_sum = sum(cvx.sum(self.s_prime[j]) for j in range(len(self.obstacles)))
        return cvx.Minimize(1e5 * slack_sum)
