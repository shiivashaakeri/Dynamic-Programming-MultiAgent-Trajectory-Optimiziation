from __future__ import annotations

import time
from typing import List, Tuple

import numpy as np

from SCvx.discretization.first_order_hold import FirstOrderHold
from SCvx.global_parameters import TRUST_RADIUS0, K
from SCvx.models.multi_agent_model import MultiAgentModel
from SCvx.optimization.agent_best_response import AgentBestResponse
from SCvx.utils.reporting import print_iteration, print_summary


class NashSolver:
    """Non-cooperative Nash via Iterative Best Response (no time-term)."""

    def __init__(
        self,
        multi_agent_model: MultiAgentModel,
        max_iter: int = 20,
        tol: float = 1e-3,
    ) -> None:
        self.mam = multi_agent_model
        self.N = multi_agent_model.N
        self.max_iter = max_iter
        self.tol = tol

        self.br_solvers = [AgentBestResponse(i, multi_agent_model) for i in range(self.N)]
        self.fohs = [FirstOrderHold(m, K) for m in multi_agent_model.models]

    def solve(
        self,
        X_refs: List[np.ndarray],
        U_refs: List[np.ndarray],
        sigma_ref: float = 1.0,
        verbose: bool = False,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
        X_curr = [x.copy() for x in X_refs]
        U_curr = [u.copy() for u in U_refs]
        change_hist: List[float] = []
        t0 = time.time()

        for it in range(self.max_iter):
            max_change = 0.0
            if verbose:
                print(f"\n--- Outer iteration {it} ---")

            X_prev_all = [x.copy() for x in X_curr]

            for i, br in enumerate(self.br_solvers):
                mats = self.fohs[i].calculate_discretization(X_curr[i], U_curr[i], sigma_ref)
                neigh_cur = {j: X_curr[j] for j in range(self.N) if j != i}
                neigh_prev = {j: X_prev_all[j] for j in range(self.N) if j != i}
                X_prev_i = X_prev_all[i]

                br.setup(
                    X_ref=X_curr[i],
                    U_ref=U_curr[i],
                    sigma_ref=sigma_ref,
                    discr_mats=mats,
                    neighbour_refs=neigh_cur,
                    X_prev=X_prev_i,
                    neighbour_prev_refs=neigh_prev,
                    tr_radius=TRUST_RADIUS0,
                )
                X_new, U_new, *_ = br.solve()

                delta = np.linalg.norm(X_new - X_curr[i])
                max_change = max(max_change, delta)
                cost_i = br.scp.prob.value
                if verbose:
                    print(f"  Agent {i}:  cost={cost_i:8.3f}   ΔX={delta:6.2e}")

                X_curr[i], U_curr[i] = X_new, U_new

            change_hist.append(max_change)
            if verbose:
                print_iteration(
                    it,
                    nu_norm=0.0,
                    slack_norm=0.0,
                    primal_res=max_change,
                    dual_res=0.0,
                    dx=max_change,
                    ds=0.0,
                    sigma=sigma_ref,
                    tr_radius=0.0,
                )
            if max_change < self.tol:
                if verbose:
                    print("Converged.")
                break

        if verbose:
            print_summary(len(change_hist), sigma_ref, time.time() - t0)

        return X_curr, U_curr, change_hist
