from __future__ import annotations

from typing import Dict, Tuple

import cvxpy as cvx
import numpy as np

from SCvx.discretization.first_order_hold import FirstOrderHold
from SCvx.global_parameters import TRUST_RADIUS0, WEIGHT_NU, WEIGHT_SIGMA, WEIGHT_SLACK, K
from SCvx.models.game_model import GameUnicycleModel
from SCvx.models.multi_agent_model import MultiAgentModel
from SCvx.optimization.sc_problem import SCProblem


class AgentBestResponse:
    """Solve one agent's best-response (pure Nash, fixed time-scale)."""

    def __init__(
        self,
        i: int,
        multi_agent_model: MultiAgentModel,
    ):
        self.i = i
        self.multi_model = multi_agent_model
        self.model: GameUnicycleModel = multi_agent_model.models[i]
        self.foh = FirstOrderHold(self.model, K)

        # neighbour position parameters
        self.Y_params: Dict[int, cvx.Parameter] = {
            j: cvx.Parameter((2, K)) for j in range(self.multi_model.N) if j != i
        }
        self.X_prev_param = cvx.Parameter((3, K))
        self.scp: SCProblem | None = None

    def setup(
        self,
        X_ref: np.ndarray,
        U_ref: np.ndarray,
        sigma_ref: float,
        discr_mats: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        neighbour_refs: Dict[int, np.ndarray],
        X_prev: np.ndarray,
        neighbour_prev_refs: Dict[int, np.ndarray],
        tr_radius: float = TRUST_RADIUS0,
    ) -> None:
        # 1) build fresh SCProblem
        self.scp = SCProblem(self.model)

        # 2) set previous-trajectory parameters
        self.X_prev_param.value = X_prev
        for j, P in self.Y_params.items():
            P.value = neighbour_refs[j][0:2, :]

        # pack the numpy lists
        neighbour_prev_pos = [neighbour_prev_refs[j][0:2, :] for j in self.Y_params]

        # 3) build the extra cost (this also lazy-inits z_params + slab constraints)
        extra_cost = self.model.get_cost_function(
            X_v=self.scp.var["X"],
            U_v=self.scp.var["U"],
            neighbour_pos=list(self.Y_params.values()),
            X_prev=self.X_prev_param,
            neighbour_prev_pos=neighbour_prev_pos,
        )

        # ────────────────────────────────────────────────────────────────────
        # **NEW**: immediately do one dual update with your initial guess,
        # so z_params are pointing along X_prev → neighbour_prev and satisfy
        # zᵀ (p_i_prev - P_j_prev) ≥ d_min.
        p_i_prev_array = X_prev[0:2, :]
        self.model.update_slabs(p_i_prev_array, neighbour_prev_pos)
        # ────────────────────────────────────────────────────────────────────

        # 4) rebuild the Problem including your slab constraints
        base_obj = self.scp.prob.objective.args[0]
        all_cons = list(self.scp.prob.constraints) + self.model.extra_constraints
        all_cons.append(self.scp.var["sigma"] == sigma_ref)
        self.scp.prob = cvx.Problem(
            cvx.Minimize(base_obj + extra_cost),
            all_cons,
        )

        # 5) set dynamics & trust-region parameters
        A_bar, B_bar, C_bar, S_bar, z_bar = discr_mats
        self.scp.set_parameters(
            A_bar=A_bar,
            B_bar=B_bar,
            C_bar=C_bar,
            S_bar=S_bar,
            z_bar=z_bar,
            X_ref=X_ref,
            U_ref=U_ref,
            sigma_ref=sigma_ref,
            weight_nu=WEIGHT_NU,
            weight_slack=WEIGHT_SLACK,
            weight_sigma=WEIGHT_SIGMA,
            tr_radius=tr_radius,
        )

    def solve(self, solver: str = "ECOS", **solver_kwargs):
        if self.scp is None:
            raise RuntimeError("call setup() before solve()")
        err = self.scp.solve(solver=solver, warm_start=True, **solver_kwargs)
        if err:
            raise RuntimeError("SCProblem error inside AgentBestResponse")

        X_i = self.scp.get_variable("X")
        U_i = self.scp.get_variable("U")
        nu_i = self.scp.get_variable("nu")

        p_i = X_i[0:2, :]
        slack_i = getattr(self.model, "get_linear_cost", lambda: 0.0)()
        return X_i, U_i, nu_i, slack_i, p_i
