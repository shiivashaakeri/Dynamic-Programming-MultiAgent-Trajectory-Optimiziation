"""Best‑response solver for a single‑integrator agent in the 3‑D Nash‑game setting."""

from __future__ import annotations

from typing import Dict, List, Tuple

import cvxpy as cvx
import numpy as np

from SCvx.discretization.first_order_hold import FirstOrderHold
from SCvx.global_parameters import TRUST_RADIUS0, WEIGHT_NU, WEIGHT_SIGMA, WEIGHT_SLACK, K
from SCvx.models.game_si_model import GameSIModel
from SCvx.models.SI_multi_agent_model import SI_MultiAgentModel
from SCvx.optimization.sc_problem import SCProblem


class SI_AgentBestResponse:
    """Solve one agent's best‑response with fixed time‑scale (single‑integrator)."""

    def __init__(self, i: int, multi_agent_model: SI_MultiAgentModel):
        self.i = i
        self.multi_model = multi_agent_model
        self.model: GameSIModel = multi_agent_model.models[i]  # type: ignore[assignment]
        self.foh = FirstOrderHold(self.model, K)

        # parameters for neighbour positions and previous trajectory
        self.Y_params: Dict[int, cvx.Parameter] = {
            j: cvx.Parameter((3, K)) for j in range(self.multi_model.N) if j != i
        }
        self.X_prev_param = cvx.Parameter((3, K))
        self.scp: SCProblem | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def setup(
        self,
        X_ref: np.ndarray,
        U_ref: np.ndarray,
        sigma_ref: float,
        discr_mats: Tuple[np.ndarray, ...],
        neighbour_refs: Dict[int, np.ndarray],
        X_prev: np.ndarray,
        neighbour_prev_refs: Dict[int, np.ndarray],
        tr_radius: float = TRUST_RADIUS0,
    ):
        """Build the convex best‑response problem for this agent."""
        self.scp = SCProblem(self.model)

        # set previous trajectory param
        self.X_prev_param.value = X_prev
        for j, P in self.Y_params.items():
            P.value = neighbour_refs[j][0:3, :]
        neighbour_prev_pos = [neighbour_prev_refs[j][0:3, :] for j in self.Y_params]

        # Extra per‑agent cost & slab constraints
        extra_cost = self.model.get_cost_function(
            X_v=self.scp.var["X"],
            U_v=self.scp.var["U"],
            neighbour_pos=list(self.Y_params.values()),
            X_prev=self.X_prev_param,
            neighbour_prev_pos=neighbour_prev_pos,
        )
        # initialise z‑slabs once
        self.model.update_slabs(X_prev[0:3, :], neighbour_prev_pos)

        # Rebuild objective with extra cost and fixed sigma
        base_obj = self.scp.prob.objective.args[0]
        total_obj = cvx.Minimize(base_obj + extra_cost)
        all_cons = list(self.scp.prob.constraints) + self.model.extra_constraints
        all_cons.append(self.scp.var["sigma"] == sigma_ref)
        self.scp.prob = cvx.Problem(total_obj, all_cons)

        # dynamics / trust‑region params
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
            raise RuntimeError("setup() must be called first")

        # Solve the problem
        self.scp.solve(solver=solver, warm_start=True, **solver_kwargs)

        # --- NEW: Robust check for solver status ---
        # A solver can "finish" without error, but find the problem is infeasible.
        # This is the most likely cause of the NoneType error.
        if self.scp.prob.status not in ["optimal", "optimal_inaccurate"]:
            print(f"!! Solver for Agent {self.i} failed with status: {self.scp.prob.status} !!")
            # You can optionally add more debug prints here to inspect parameter values
            raise RuntimeError(f"SCProblem for agent {self.i} was not solved successfully.")
        
        # Original code continues here
        X_i = self.scp.get_variable("X")
        U_i = self.scp.get_variable("U")
        nu_i = self.scp.get_variable("nu")
        
        # Add a fallback check just in case
        if X_i is None:
            raise ValueError(f"Solver for agent {self.i} reported success but the solution is None.")

        p_i = X_i[0:3, :]
        slack_i = getattr(self.model, "get_linear_cost", lambda: 0.0)()
        return X_i, U_i, nu_i, slack_i, p_i