"""Best-response wrapper around SCvx for Nash games.

Each agent solves its own optimal-control problem (via SCProblem/SCvx)
while treating other agents' position trajectories as **fixed** parameters
- this is exactly an Iterative Best Response step toward a Nash equilibrium.
"""

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
    """Solve one agent's best-response problem given others' trajectories."""

    def __init__(
        self,
        i: int,
        multi_agent_model: MultiAgentModel,
        rho_admm: float = 1.0,  # not used here but kept for consistency  # noqa: ARG002
    ):
        self.i = i
        self.N = multi_agent_model.N
        self.multi_model = multi_agent_model

        # This agent's own GameUnicycleModel
        self.model: GameUnicycleModel = multi_agent_model.models[i]  # type: ignore

        # Discretizer for this agent
        self.foh = FirstOrderHold(self.model, K)

        # 2.K parameters for neighbours' positions
        self.Y_params: Dict[int, cvx.Parameter] = {j: cvx.Parameter((2, K)) for j in range(self.N) if j != i}

        # Own previous state trajectory
        self.X_prev_param = cvx.Parameter((3, K))

        # Placeholder for SCProblem
        self.scp: SCProblem | None = None

    # ------------------------------------------------------------------
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
        """Prepare CVXPY problem for the best-response step."""
        # Initialize SCProblem
        self.scp = SCProblem(self.model)

        # Set trajectory parameters
        self.X_prev_param.value = X_prev
        for j, param in self.Y_params.items():
            param.value = neighbour_refs[j][0:2, :]

        # Prepare previous neighbor positions for constraint linearization
        neighbour_prev_pos = [neighbour_prev_refs[j][0:2, :] for j in self.Y_params]

        # Build extra cost & hard constraints
        extra_cost = self.model.get_cost_function(
            X_v=self.scp.var["X"],
            U_v=self.scp.var["U"],
            neighbour_pos=list(self.Y_params.values()),
            X_prev=self.X_prev_param,
            neighbour_prev_pos=neighbour_prev_pos,
        )

        # Build final objective and constraints
        base_obj = self.scp.prob.objective.args[0]
        self.scp.prob = cvx.Problem(
            cvx.Minimize(base_obj + extra_cost), self.scp.prob.constraints + self.model.extra_constraints
        )

        # Set dynamics and trust-region parameters
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

    # ------------------------------------------------------------------
    def solve(self, solver: str = "ECOS", **solver_kwargs):
        """Solve the best-response SCvx problem and return updated trajectories."""
        if self.scp is None:
            raise RuntimeError("call setup() before solve()")

        err = self.scp.solve(solver=solver, warm_start=True, **solver_kwargs)
        if err:
            raise RuntimeError("SCProblem error inside AgentBestResponse")

        X_i = self.scp.get_variable("X")
        U_i = self.scp.get_variable("U")
        nu_i = self.scp.get_variable("nu")
        sigma_i = self.scp.get_variable("sigma")  # should equal sigma_ref  # noqa: F841
        p_i = X_i[0:2, :]
        slack_i = self.model.get_linear_cost() if hasattr(self.model, "get_linear_cost") else 0.0
        return X_i, U_i, nu_i, slack_i, p_i
