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
        control_weight:      float = 1.0,
        collision_weight:    float = 10.0,     # <<< keep, but *internal* only
        collision_radius:    float = 0.50,
        control_rate_weight: float = 10.0,
        curvature_weight:    float = 10.0,
        **kwargs,
    ):
        # >>> Strip cost params from kwargs so they don't reach UnicycleModel
        kwargs.pop("collision_weight", None)
        kwargs.pop("collision_radius", None)
        kwargs.pop("control_weight", None)
        kwargs.pop("control_rate_weight", None)
        kwargs.pop("curvature_weight", None)

        super().__init__(
            r_init=r_init,
            r_final=r_final,
            obstacles=obstacles,
            **kwargs,                 # now safe: nothing unexpected inside
        )

        # store weights
        self.control_weight      = control_weight
        self.collision_weight    = collision_weight
        self.collision_radius    = collision_radius
        self.control_rate_weight = control_rate_weight
        self.curvature_weight    = curvature_weight

        self.extra_constraints: List[cvx.Constraint] = []
    # ------------------------------------------------------------------
    # Cost builder -----------------------------------------------------
    # ------------------------------------------------------------------
    def get_cost_function(
        self,
        X_v: cvx.Variable,             # (3, K)   states
        U_v: cvx.Variable,             # (2, K)   inputs
        neighbour_pos: List[cvx.Parameter],   # list of (2,K)
    ) -> cvx.Expression:
        """
        Return convex cost expression and populate self.extra_constraints
        with *hard* pair-wise distance inequalities.

        Caller (AgentBestResponse) must add those constraints to the SCProblem.
        """
        # -------- 1) effort ------------------------------------------
        cost = self.control_weight * cvx.sum_squares(U_v)

        # -------- 2) smoothness --------------------------------------
        if self.control_rate_weight > 0:
            dU = U_v[:, 1:] - U_v[:, :-1]
            cost += self.control_rate_weight * cvx.sum_squares(dU)

        if self.curvature_weight > 0:
            dtheta = X_v[2, 1:] - X_v[2, :-1]
            cost += self.curvature_weight * cvx.sum_squares(dtheta)

        # -------- 3) hard collision buffer ---------------------------
        p_i = X_v[0:2, :]                         # 2K
        self.extra_constraints.clear()

        if neighbour_pos:
            for p_j in neighbour_pos:
                s = cvx.Variable(K, nonneg=True)
                self.extra_constraints += [
                    cvx.norm(p_i - p_j, 2, axis=0) <= self.collision_radius + s
                ]
                cost += self.collision_weight * cvx.norm(s, 1)   # L¹, not square

        return cost
