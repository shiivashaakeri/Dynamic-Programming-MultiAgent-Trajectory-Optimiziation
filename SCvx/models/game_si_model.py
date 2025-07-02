import cvxpy as cvx
import numpy as np

from SCvx.global_parameters import K
from SCvx.models.single_integrator_model import SingleIntegratorModel


class GameSIModel(SingleIntegratorModel):
    """
    Single-integrator model with per-agent cost parameters for Nash games.
    State  : x = [x, y, z]
    Control: u = [vx, vy, vz]
    """

    from typing import ClassVar

    _COST_KEYS: ClassVar[set[str]] = {
        "control_weight",
        "collision_weight",
        "collision_radius",
        "control_rate_weight",
        "curvature_weight",
        "inertia_weight",
        "path_weight",
    }

    def __init__(self, *, r_init, r_final, obstacles=None, **kwargs):
        # ----- store weights (with defaults) -----
        self.control_weight = kwargs.pop("control_weight", 1.0)
        self.collision_weight = kwargs.pop("collision_weight", 80.0)
        self.collision_radius = kwargs.pop("collision_radius", 0.5)
        self.control_rate_weight = kwargs.pop("control_rate_weight", 5.0)
        self.curvature_weight = kwargs.pop("curvature_weight", 0.0)
        self.inertia_weight = kwargs.pop("inertia_weight", 0.0)
        self.path_weight = kwargs.pop("path_weight", 0.0)

        # strip any stray cost keys
        for k in list(kwargs):
            if k in self._COST_KEYS:
                kwargs.pop(k)

        super().__init__(r_init=r_init, r_final=r_final, obstacles=obstacles, **kwargs)

        # Placeholders for constraints and parameters
        self.extra_constraints: list[cvx.Constraint] = []
        self.z_params = []
        self.coll_slacks = []  # <<< NEW: For soft collision constraints

    def update_slabs(self, p_i: np.ndarray, neighbour_prev_pos):
        """
        z* = argmax_{‖z‖≤1} zᵀ(p_i - P_j).
        """
        for j, P_prev in enumerate(neighbour_prev_pos):
            for k in range(K):
                d = p_i[:, k] - P_prev[:, k]
                norm = np.linalg.norm(d)
                z_star = np.zeros(3) if norm < 1e-6 else d / norm
                self.z_params[j][k].value = z_star

    def get_cost_function(
        self,
        X_v: cvx.Variable,  # (3,K)
        U_v: cvx.Variable,  # (3,K)
        neighbour_pos,
        X_prev: cvx.Parameter,  # (3,K)
        neighbour_prev_pos,  # noqa: ARG002
    ):
        self.extra_constraints.clear()
        cost = 0

        # 1) Control effort
        cost += self.control_weight * cvx.sum_squares(U_v)

        # 2) Control-rate smoothing
        if self.control_rate_weight > 0:
            cost += self.control_rate_weight * cvx.sum_squares(U_v[:, 1:] - U_v[:, :-1])

        # 3) Inertia (regularization)
        if self.inertia_weight > 0:
            cost += self.inertia_weight * cvx.sum_squares(X_v - X_prev)

        # 4) Path-length
        if self.path_weight > 0:
            cost += self.path_weight * cvx.sum(cvx.norm(X_v[:, 1:] - X_v[:, :-1], axis=0))

        # --- Soft Inter-Agent Collision Avoidance ---
        p_i = X_v  # positions are the state for a single integrator

        # Re-initialize slack variables for this solve
        self.coll_slacks = [cvx.Variable(K, name=f"s_coll_{j}", nonneg=True) for j in range(len(neighbour_pos))]

        # 5) Add slack penalty to the cost
        for s_var in self.coll_slacks:
            cost += self.collision_weight * cvx.sum(s_var)

        # Initialize z-parameters for slab constraints if they don't exist
        if not self.z_params:
            self.z_params = [
                [cvx.Parameter((3,), name=f"z_{j}_{k}") for k in range(K)] for j in range(len(neighbour_pos))
            ]

        # Build the slab constraints, now softened by the slack variables
        for j, P_j in enumerate(neighbour_pos):
            s_j = self.coll_slacks[j]  # Get the slack variable for this neighbor
            for k in range(K):
                z = self.z_params[j][k]
                constraint = z @ (p_i[:, k] - P_j[:, k]) >= self.collision_radius - s_j[k]
                self.extra_constraints.append(constraint)

        return cost
