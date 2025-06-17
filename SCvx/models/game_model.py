from typing import List, Optional

import cvxpy as cvx
import numpy as np

from SCvx.global_parameters import K

from .unicycle_model import UnicycleModel


class GameUnicycleModel(UnicycleModel):
    """Unicycle model with per-agent cost parameters for Nash games."""

    def __init__(
        self,
        *,
        r_init: np.ndarray,
        r_final: np.ndarray,
        obstacles: Optional[List] = None,
        # ---- COST WEIGHTS ------------------------------------------
        control_weight: float = 1.0,
        collision_weight: float = 10.0,  # for soft costs, not used in hard constraint version
        collision_radius: float = 0.50,
        control_rate_weight: float = 10.0,
        curvature_weight: float = 10.0,
        inertia_weight: float = 0.0,
        **kwargs,
    ):
        # Strip cost-related kwargs before passing to superclass
        kwargs.pop("collision_weight", None)
        kwargs.pop("collision_radius", None)
        kwargs.pop("control_weight", None)
        kwargs.pop("control_rate_weight", None)
        kwargs.pop("curvature_weight", None)

        super().__init__(
            r_init=r_init,
            r_final=r_final,
            obstacles=obstacles,
            **kwargs,
        )

        # Store agent-specific cost weights
        self.control_weight = control_weight
        self.collision_weight = collision_weight
        self.collision_radius = collision_radius
        self.control_rate_weight = control_rate_weight
        self.curvature_weight = curvature_weight
        self.inertia_weight = inertia_weight

        # Will be filled by cost function
        self.extra_constraints: List[cvx.Constraint] = []

    # ------------------------------------------------------------------
    def get_cost_function(
        self,
        X_v: cvx.Variable,  # shape (3, K)
        U_v: cvx.Variable,  # shape (2, K)
        neighbour_pos: List[cvx.Parameter],  # each (2, K)
        X_prev: cvx.Parameter,  # shape (3, K)
        neighbour_prev_pos: List[np.ndarray],  # each (2, K), numpy arrays
    ) -> cvx.Expression:
        """Convex cost expression with linearized hard collision constraints."""

        self.extra_constraints.clear()
        cost = 0

        # 1. Control effort
        if self.control_weight > 0:
            cost += self.control_weight * cvx.sum_squares(U_v)

        # 2. Smoothness: penalize control rate changes
        if self.control_rate_weight > 0:
            dU = U_v[:, 1:] - U_v[:, :-1]
            cost += self.control_rate_weight * cvx.sum_squares(dU)

        # 3. Smoothness: penalize heading curvature
        if self.curvature_weight > 0:
            dtheta = X_v[2, 1:] - X_v[2, :-1]
            cost += self.curvature_weight * cvx.sum_squares(dtheta)

        # 4. Linearized hard collision constraints
        p_i = X_v[0:2, :]  # current agent's position: (2, K)
        p_i_prev = X_prev.value[0:2, :]  # previous positions (as np array)

        for p_j_param, p_j_prev in zip(neighbour_pos, neighbour_prev_pos):
            for k in range(K):
                d = p_i_prev[:, k] - p_j_prev[:, k]
                norm_d = np.linalg.norm(d)
                if norm_d < 1e-6:
                    continue  # skip degenerate case
                n_hat = d / norm_d  # (2,)
                relative_displacement = p_i[:, k] - p_j_param[:, k]
                constraint_expr = n_hat.T @ relative_displacement
                self.extra_constraints.append(constraint_expr >= self.collision_radius)

        # 5. Inertia: penalize deviation from previous trajectory
        if self.inertia_weight > 0:
            cost += self.inertia_weight * cvx.sum_squares(X_v - X_prev)

        return cost
