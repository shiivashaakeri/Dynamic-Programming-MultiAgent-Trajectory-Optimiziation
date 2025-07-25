from typing import ClassVar, List, Tuple, Optional
import cvxpy as cvx
import numpy as np

from SCvx.config.SI_default_game import AGT_COLL_RAD as GLOBAL_COLL_RAD
from SCvx.global_parameters import K as GLOBAL_K
from .single_integrator_model import SingleIntegratorModel
from SCvx.utils.intersample_collision import find_critical_times, linearize_h, make_segment_f


class GameSIModel(SingleIntegratorModel):
    """
    Single-integrator model with per-agent cost parameters for Nash games.
    State:   x = [x, y, z]
    Control: u = [vx, vy, vz]
    """

    _COST_KEYS: ClassVar[set[str]] = {
        "control_weight",
        "collision_weight",
        "control_rate_weight",
        "curvature_weight",
        "inertia_weight",
        "path_weight",
    }

    def __init__(
        self,
        *,
        r_init: np.ndarray,
        r_final: np.ndarray,
        robot_radius: float = 0.5,
        collision_radius: Optional[float] = None,
        obstacles: Optional[List[Tuple[List[float], float]]] = None,
        **kwargs
    ):
        # --- agent geometry ---
        self.robot_radius = robot_radius
        self.agent_coll_rad = (
            collision_radius if collision_radius is not None else GLOBAL_COLL_RAD
        )

        # --- cost weights ---
        self.control_weight      = kwargs.pop("control_weight", 1.0)
        self.collision_weight    = kwargs.pop("collision_weight", 80.0)
        self.control_rate_weight = kwargs.pop("control_rate_weight", 5.0)
        self.curvature_weight    = kwargs.pop("curvature_weight", 0.0)
        self.inertia_weight      = kwargs.pop("inertia_weight", 0.0)
        self.path_weight         = kwargs.pop("path_weight", 0.0)
        # strip out any stray cost keys
        for key in list(kwargs):
            if key in self._COST_KEYS:
                kwargs.pop(key)

        # --- initialize base dynamics + obstacle model ---
        super().__init__(
            r_init       = r_init,
            r_final      = r_final,
            robot_radius = self.robot_radius,
            obstacles    = obstacles,
        )

        # --- placeholders for slabs & inter-sample constraints ---
        self.coll_slacks:       List[cvx.Variable]      = []  # per-neighbor
        self.z_params:          List[List[cvx.Parameter]] = []  # normals
        self.inter_slacks:      List[cvx.Variable]      = []  # per-minima
        self.extra_constraints: List[cvx.Constraint]    = []  # appended later

    def update_slabs(
        self,
        p_i: np.ndarray,                    # (3, K)
        neighbour_prev_pos: List[np.ndarray]  # each (3, K)
    ) -> None:
        """
        Compute and store z* = (x_i - x_j)/||x_i - x_j|| for each neighbor & knot.
        """
        if not self.z_params:
            self.z_params = [
                [cvx.Parameter((self.n_x,), name=f"z_{j}_{k}")
                 for k in range(GLOBAL_K)]
                for j in range(len(neighbour_prev_pos))
            ]
        for j, P_prev in enumerate(neighbour_prev_pos):
            for k in range(GLOBAL_K):
                d = p_i[:, k] - P_prev[:, k]
                norm = np.linalg.norm(d)
                z_star = np.zeros(self.n_x) if norm < 1e-6 else d / norm
                self.z_params[j][k].value = z_star

    def get_cost_function(
        self,
        X_v: cvx.Variable,                 # (3, K)
        U_v: cvx.Variable,                 # (3, K)
        neighbour_pos: List[cvx.Parameter],  # each (3, K)
        X_prev: cvx.Parameter,             # (3, K)
        neighbour_prev_pos: List[np.ndarray] # each (3, K)
    ) -> cvx.Expression:
        """
        Build and return the per-agent cost (control + smoothing + slabs),
        while populating self.extra_constraints with soft inter-agent slabs.
        """
        self.extra_constraints.clear()
        cost = 0

        # 1) control effort
        cost += self.control_weight * cvx.sum_squares(U_v)
        # 2) control-rate smoothing
        cost += self.control_rate_weight * cvx.sum_squares(U_v[:,1:] - U_v[:,:-1])
        # 3) inertia regularization
        cost += self.inertia_weight * cvx.sum_squares(X_v - X_prev)
        # 4) path-length penalty
        cost += self.path_weight * cvx.sum(
            cvx.norm(X_v[:,1:] - X_v[:,:-1], axis=0)
        )

        # --- soft inter-agent collision avoidance ---
        Nnbr = len(neighbour_pos)
        self.coll_slacks = [
            cvx.Variable(GLOBAL_K, name=f"s_coll_{j}", nonneg=True)
            for j in range(Nnbr)
        ]
        for s in self.coll_slacks:
            cost += self.collision_weight * cvx.sum(s)

        # compute normals from last iterate
        self.update_slabs(X_prev.value, neighbour_prev_pos)

        # add one slab constraint per neighbor per knot
        for j, P in enumerate(neighbour_pos):
            for k in range(GLOBAL_K):
                z = self.z_params[j][k]
                expr = z @ (X_v[:,k] - P[:,k])
                self.extra_constraints.append(
                    expr >= self.agent_coll_rad
                )

        return cost

    def update_intersample_constraints(
        self,
        X_v: cvx.Variable,       # (3, K)
        U_v: cvx.Variable,       # (3, K)
        X_nom: np.ndarray,       # (3, K)
        U_nom: np.ndarray,       # (3, K)
        foh,                     # FirstOrderHold instance
        sigma_ref: float,
    ) -> None:
        """
        Append linearized inter-sample obstacle constraints to
        self.extra_constraints, with new slack variables in self.inter_slacks.
        """
        # reset
        self.extra_constraints = []
        self.inter_slacks = []

        for k in range(GLOBAL_K - 1):
            f_seg, _ = make_segment_f(foh, U_nom[:,k], U_nom[:,k+1], sigma=1.0)
            for obs_idx, obstacle in enumerate(self.obstacles):
                # find all interior minima t*∈(0,1)
                t_stars = find_critical_times(
                    xk=X_nom[:, k],
                    uk=U_nom[:, k],
                    f=f_seg,
                    T=np.eye(self.n_x),
                    obstacle=obstacle,
                    dt=1.0,
                )
                for t_star in t_stars:
                    h0, grad_x, grad_u = linearize_h(
                        xk=X_nom[:, k],
                        uk=U_nom[:, k],
                        t_star=t_star,
                        f=f_seg,
                        T=np.eye(self.n_x),
                        obstacle=obstacle,
                    )
                    s = cvx.Variable(
                        name=f"s_intersample_k{k}_obs{obs_idx}_t{int(t_star*1e3)}",
                        nonneg=True,
                    )
                    self.inter_slacks.append(s)
                    expr = (
                        h0
                        + grad_x @ (X_v[:,k] - X_nom[:,k])
                        + grad_u @ (U_v[:,k] - U_nom[:,k])
                        + s
                    )
                    self.extra_constraints.append(expr >= 0)