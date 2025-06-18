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
        control_weight: float = 1.0,
        collision_weight: float = 10.0,
        collision_radius: float = 0.50,
        control_rate_weight: float = 10.0,
        curvature_weight: float = 10.0,
        inertia_weight: float = 0.0,
        **kwargs,
    ):
        # strip cost-kwargs before calling super
        for key in (
            "control_weight",
            "collision_weight",
            "collision_radius",
            "control_rate_weight",
            "curvature_weight",
            "inertia_weight",
        ):
            kwargs.pop(key, None)
        super().__init__(r_init=r_init, r_final=r_final, obstacles=obstacles, **kwargs)

        self.control_weight = control_weight
        self.collision_weight = collision_weight
        self.collision_radius = collision_radius
        self.control_rate_weight = control_rate_weight
        self.curvature_weight = curvature_weight
        self.inertia_weight = inertia_weight

        # will be populated in get_cost_function
        self.extra_constraints: List[cvx.Constraint] = []

    def get_cost_function(
        self,
        X_v: cvx.Variable,  # (3, K)
        U_v: cvx.Variable,  # (2, K)
        neighbour_pos: List[cvx.Parameter],
        X_prev: cvx.Parameter,  # (3, K)
        neighbour_prev_pos: List[np.ndarray],
    ) -> cvx.Expression:
        """
        Build per-agent cost = control effort + smoothing + inertia,
                        with hard linearized collision constraints.
        """
        self.extra_constraints.clear()
        cost = 0

        # 1) control effort
        cost += self.control_weight * cvx.sum_squares(U_v)

        # 2) smoothing on control-rate & curvature
        if self.control_rate_weight > 0:
            dU = U_v[:, 1:] - U_v[:, :-1]
            cost += self.control_rate_weight * cvx.sum_squares(dU)
        if self.curvature_weight > 0:
            dtheta = X_v[2, 1:] - X_v[2, :-1]
            cost += self.curvature_weight * cvx.sum_squares(dtheta)

        # 3) inertia regularization
        if self.inertia_weight > 0:
            cost += self.inertia_weight * cvx.sum_squares(X_v - X_prev)

        # 4) linearized hard-collision constraints
        p_i = X_v[0:2, :]
        p_i_prev = X_prev.value[0:2, :]
        for P_j, P_j_prev in zip(neighbour_pos, neighbour_prev_pos):
            for k in range(K):
                d = p_i_prev[:, k] - P_j_prev[:, k]
                norm_d = np.linalg.norm(d)
                if norm_d < 1e-6:
                    continue
                n_hat = d / norm_d
                self.extra_constraints.append(n_hat.T @ (p_i[:, k] - P_j[:, k]) >= self.collision_radius)

        return cost
