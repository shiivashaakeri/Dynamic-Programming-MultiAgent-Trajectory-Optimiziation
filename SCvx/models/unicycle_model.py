from typing import List, Optional, Tuple

import cvxpy as cvx
import numpy as np
import sympy as sp

from SCvx.global_parameters import K

from .base_model import BaseModel


class UnicycleModel(BaseModel):
    """
    Unicycle dynamics with obstacle constraints for path planning.

    State: x = [x, y, theta]^T
    Input: u = [v, w]^T

    Obstacles: list of (center, radius) tuples
    """

    n_x = 3
    n_u = 2

    def __init__(
        self,
        r_init: np.ndarray = np.array([-8.0, -8.0, 0.0]),
        r_final: np.ndarray = np.array([8.0, 8.0, 0.0]),
        v_max: float = 1.0,
        w_max: float = np.pi / 6,
        bounds: Tuple[float, float] = (-10.0, 10.0),
        robot_radius: float = 0.5,
        obstacles: Optional[List[Tuple[List[float], float]]] = None,
    ):
        super().__init__()
        # Initial and final states
        self.x_init = r_init.reshape(-1)
        self.x_final = r_final.reshape(-1)

        # Input limits and spatial bounds
        self.v_max = v_max
        self.w_max = w_max
        self.lower_bound, self.upper_bound = bounds
        self.robot_radius = robot_radius

        # Obstacles: list of (center, radius)
        self.obstacles = (
            obstacles if obstacles is not None else [([5.0, 4.0], 3.0), ([-5.0, -4.0], 3.0), ([0.0, 0.0], 2.0)]
        )
        # Slack variables for linearized obstacle constraints
        self.s_prime = [cvx.Variable((K, 1), nonneg=True) for _ in self.obstacles]

        # Symbolic definitions for dynamics
        x_sym = sp.Matrix(sp.symbols("x y theta", real=True))
        u_sym = sp.Matrix(sp.symbols("v w", real=True))
        f_expr = sp.Matrix([u_sym[0] * sp.cos(x_sym[2]), u_sym[0] * sp.sin(x_sym[2]), u_sym[1]])
        A_expr = f_expr.jacobian(x_sym)
        B_expr = f_expr.jacobian(u_sym)

        # Lambdify dynamics and Jacobians
        self.f = sp.lambdify((x_sym, u_sym), f_expr, "numpy")
        self.A = sp.lambdify((x_sym, u_sym), A_expr, "numpy")
        self.B = sp.lambdify((x_sym, u_sym), B_expr, "numpy")

    def get_equations(self) -> Tuple:
        """
        Return the dynamics function and its Jacobians.
        Returns:
            f: callable, A: callable, B: callable
        """
        return self.f, self.A, self.B

    def initialize_trajectory(self, X: np.ndarray, U: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initialize states along a straight line and zero controls.
        """
        K_local = X.shape[1]
        for k in range(K_local):
            alpha1 = (K_local - 1 - k) / (K_local - 1)
            alpha2 = k / (K_local - 1)
            X[:, k] = alpha1 * self.x_init + alpha2 * self.x_final
        U[:] = 0
        return X, U

    def get_constraints(
        self, X: cvx.Variable, U: cvx.Variable, X_ref: cvx.Parameter, U_ref: cvx.Parameter  # noqa: ARG002
    ) -> List[cvx.Constraint]:
        """
        Construct model-specific constraints, including bounds and
        linearized obstacle avoidance with slack.
        """
        constraints = []
        # Boundary conditions
        constraints += [X[:, 0] == self.x_init, X[:, -1] == self.x_final, U[:, 0] == 0, U[:, -1] == 0]
        # Input bounds
        constraints += [U[0, :] >= 0, U[0, :] <= self.v_max, cvx.abs(U[1, :]) <= self.w_max]
        # State bounds (x and y)
        constraints += [
            X[0:2, :] <= self.upper_bound - self.robot_radius,
            X[0:2, :] >= self.lower_bound + self.robot_radius,
        ]
        # Linearized obstacle constraints
        for j, (p, r) in enumerate(self.obstacles):
            p_vec = np.array(p).reshape(
                2,
            )
            r_total = r + self.robot_radius
            for k in range(K):
                x_ref_k = X_ref[0:2, k]
                diff = x_ref_k - p_vec
                norm_ref = cvx.norm(diff, 2) + 1e-6
                a_k = diff / norm_ref
                lhs = a_k.T @ (X[0:2, k] - p_vec)
                constraints.append(lhs >= r_total - self.s_prime[j][k, 0])
        return constraints

    def get_objective(self, X: cvx.Variable, U: cvx.Variable, X_ref: cvx.Parameter, U_ref: cvx.Parameter):  # noqa: ARG002
        """
        Model-specific objective: minimize obstacle slack.
        """
        slack_sum = sum(cvx.sum(self.s_prime[j]) for j in range(len(self.obstacles)))
        return cvx.Minimize(1e5 * slack_sum)
