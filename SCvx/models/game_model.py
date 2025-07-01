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
        control_rate_weight: float = 5.0,
        curvature_weight: float = 100.0,
        inertia_weight: float = 0.0,
        path_weight: float = 0.0,
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
            "path_weight",
        ):
            kwargs.pop(key, None)
        super().__init__(r_init=r_init, r_final=r_final, obstacles=obstacles, **kwargs)

        self.control_weight = control_weight
        self.collision_weight = collision_weight
        self.collision_radius = collision_radius
        self.control_rate_weight = control_rate_weight
        self.curvature_weight = curvature_weight
        self.inertia_weight = inertia_weight
        self.path_weight = path_weight

        # will be populated in get_cost_function
        self.extra_constraints: List[cvx.Constraint] = []
        self.z_params: List[List[cvx.Parameter]] = []

    def update_slabs(self, p_i: np.ndarray, neighbour_prev_pos: List[np.ndarray]):
        """
        Compute each z* = argmax_{||z||<=1} z^T (p_i - P_j)
        => z* = (d / ||d||) clipped to unit ball.
        """
        for j, P_j_prev in enumerate(neighbour_prev_pos):
            for k in range(K):
                d = p_i[:, k] - P_j_prev[:, k]
                norm_d = np.linalg.norm(d)
                # unit-vector in direction of d
                z_star = np.zeros(2) if norm_d < 1e-6 else d / norm_d
                # assign into the primal's parameter
                self.z_params[j][k].value = z_star

    def get_cost_function(
        self,
        X_v: cvx.Variable,  # shape = (3, K)
        U_v: cvx.Variable,  # shape = (2, K)
        neighbour_pos: List[cvx.Parameter],
        X_prev: cvx.Parameter,  # shape = (3, K)
        neighbour_prev_pos: List[np.ndarray],  # noqa: ARG002
    ) -> cvx.Expression:
        """
        Build per-agent cost = control effort
                        + control-rate smoothing
                        + curvature smoothing
                        + inertia regularization
                        + path-length penalty
        and collect primal-dual (slab) collision constraints.
        """
        self.extra_constraints.clear()
        cost = 0

        # 1) control effort
        cost += self.control_weight * cvx.sum_squares(U_v)

        # 2) smoothing
        if self.control_rate_weight > 0:
            dU = U_v[:, 1:] - U_v[:, :-1]
            cost += self.control_rate_weight * cvx.sum_squares(dU)
        if self.curvature_weight > 0:
            dtheta = X_v[2, 1:] - X_v[2, :-1]
            cost += self.curvature_weight * cvx.sum_squares(dtheta)

        # 3) inertia
        if self.inertia_weight > 0:
            cost += self.inertia_weight * cvx.sum_squares(X_v - X_prev)

        # 4) path-length
        if getattr(self, "path_weight", 0.0) > 0:
            p = X_v[0:2, :]
            dp = p[:, 1:] - p[:, :-1]
            cost += self.path_weight * cvx.sum(cvx.norm(dp, axis=0))

        p_i = X_v[0:2, :]

        # lazy init of z-parameters
        if not self.z_params or len(self.z_params) != len(neighbour_pos):
            self.z_params = [
                [cvx.Parameter((2,), name=f"slab_z_{j}_{k}") for k in range(K)] for j in range(len(neighbour_pos))
            ]
            # set all to zero initially
            for j in range(len(neighbour_pos)):
                for k in range(K):
                    self.z_params[j][k].value = np.zeros(2)

        # only the affine “slab” constraint in the primal
        for j, P_j in enumerate(neighbour_pos):
            for k in range(K):
                z = self.z_params[j][k]
                self.extra_constraints.append(z @ (p_i[:, k] - P_j[:, k]) >= self.collision_radius)

        return cost
