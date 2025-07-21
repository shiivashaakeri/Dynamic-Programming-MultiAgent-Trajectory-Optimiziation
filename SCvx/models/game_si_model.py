import cvxpy as cvx
import numpy as np

from SCvx.global_parameters import K as GLOBAL_K
from SCvx.models.single_integrator_model import SingleIntegratorModel
from SCvx.utils.intersample_collision import (
    find_critical_times,
    linearize_h,
    make_segment_f,
)


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
        # ----- NEW for inter-sample collision slacks -----
        # Will hold CVXPY Variables for each inter-sample minimum
        self.inter_slacks: list[cvx.Variable] = []
        # Weight to penalize those slacks in the cost
        self.inter_slack_weight: float = kwargs.get("inter_slack_weight", 1e6)
        # Placeholders for constraints and parameters
        self.extra_constraints: list[cvx.Constraint] = []
        # Reset the list of inter-sample slacks each iteration
        self.inter_slacks = []
        self.z_params = []
        self.coll_slacks = []  # <<< NEW: For soft collision constraints

    def update_slabs(self, p_i: np.ndarray, neighbour_prev_pos):
        """
        z* = argmax_{‖z‖≤1} zᵀ(p_i - P_j).
        """
        for j, P_prev in enumerate(neighbour_prev_pos):
            for k in range(GLOBAL_K):
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
        self.coll_slacks = [cvx.Variable(GLOBAL_K, name=f"s_coll_{j}", nonneg=True) for j in range(len(neighbour_pos))]

        # 5) Add slack penalty to the cost
        for s_var in self.coll_slacks:
            cost += self.collision_weight * cvx.sum(s_var)
# --- Soft Intersample Collision Avoidance ---
        # Penalize each inter-sample slack we created earlier
        for s in self.inter_slacks:
            cost += self.inter_slack_weight * s

        # Initialize z-parameters for slab constraints if they don't exist
        if not self.z_params:
            self.z_params = [
                [cvx.Parameter((3,), name=f"z_{j}_{k}") for k in range(GLOBAL_K)] for j in range(len(neighbour_pos))
            ]

        # Build the slab constraints, now softened by the slack variables
        for j, P_j in enumerate(neighbour_pos):
            s_j = self.coll_slacks[j]  # Get the slack variable for this neighbor
            for k in range(GLOBAL_K):
                z = self.z_params[j][k]
                constraint = z @ (p_i[:, k] - P_j[:, k]) >= self.collision_radius - s_j[k]
                self.extra_constraints.append(constraint)

        return cost

    def update_intersample_constraints(
        self,
        X_v: cvx.Variable,  # CVXPY state variable (n_x×K)
        U_v: cvx.Variable,  # CVXPY control variable (n_u×K)
        X_nom: np.ndarray,  # current nominal states (n_x×K)
        U_nom: np.ndarray,  # current nominal controls (n_u×K)
        foh,  # FirstOrderHold instance
        sigma_ref: float,
    ):
        """
        Builds and appends linearized inter‐sample obstacle constraints
        into self.extra_constraints, with debug prints.
        """
        # reset or initialize extra_constraints list
        self.extra_constraints = []

        # Loop over each time‐step segment
        for k in range(GLOBAL_K - 1):
            # Build the per‐segment transition function f_seg(t)
            f_seg, _ = make_segment_f(foh, U_nom[:, k], U_nom[:, k + 1], sigma=1.0)

            # For each obstacle, find interior minima and linearize
            for obs_idx, obstacle in enumerate(self.obstacles):
                # 1) find all t* in (0,1)
                t_stars = find_critical_times(
                    xk=X_nom[:, k],
                    uk=U_nom[:, k],  # ignored by f_seg
                    f=f_seg,
                    T=np.eye(self.n_x),  # sphere test
                    obstacle=obstacle,
                    dt=1.0,  # since f_seg uses t∈[0,1]
                )

                # 2) For each t*, compute h0 and gradients, then append affine constraint
                for t_star in t_stars:
                    # compute margin & gradients
                    h0, grad_x, grad_u = linearize_h(
                        xk       = X_nom[:, k],
                        uk       = U_nom[:, k],
                        t_star   = t_star,
                        f        = f_seg,
                        T        = np.eye(self.n_x),
                        obstacle = obstacle,
                    )

                    # 1) create a non-neg slack for this minima
                    s = cvx.Variable(
                        name=f"s_intersample_k{k}_obs{obs_idx}_t{int(t_star*1e3)}",
                        nonneg=True,
                    )
                    self.inter_slacks.append(s)
                    # 2) build the affine constraint + slack ≥ 0
                    expr = (
                       h0
                        + grad_x @ (X_v[:, k] - X_nom[:, k])
                        + grad_u @ (U_v[:, k] - U_nom[:, k])
                        + s
                    )
                    self.extra_constraints.append(expr >= 0)
