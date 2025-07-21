"""Iterative Best‑Response Nash solver for 3‑D single‑integrator agents."""

from __future__ import annotations

import time
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm, trange

from SCvx.discretization.first_order_hold import FirstOrderHold
from SCvx.global_parameters import TRUST_RADIUS0, K
from SCvx.models.SI_multi_agent_model import SI_MultiAgentModel
from SCvx.optimization.si_agent_best_response import SI_AgentBestResponse
from SCvx.utils.multi_agent_logging import print_iteration, print_summary

# ------------------------------------------------------------------
# Public solve interface
# ------------------------------------------------------------------


class SI_NashSolver:
    """Non‑cooperative Nash equilibrium via Iterative Best Response (3‑D SI agents)."""

    def __init__(
        self,
        multi_agent_model: SI_MultiAgentModel,
        max_iter: int = 20,
        tol: float = 1e-3,
        max_acs_iters: int = 5,
        acs_tol: float = 1e-3,
    ):
        self.mam = multi_agent_model
        self.N = multi_agent_model.N
        self.max_iter = max_iter
        self.tol = tol
        self.max_acs_iters = max_acs_iters
        self.acs_tol = acs_tol
        # per‑agent best‑response solvers and FOH discretisers
        self.br_solvers = [SI_AgentBestResponse(i, multi_agent_model) for i in range(self.N)]
        self.fohs = [FirstOrderHold(m, K) for m in multi_agent_model.models]

    def solve(
        self,
        X_refs: List[np.ndarray],
        U_refs: List[np.ndarray],
        sigma_ref: float = 1.0,
        verbose: bool = False,
        show_progress: bool = True,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
        """Run iterative best-response until trajectories converge."""
        # initialize
        X_curr = [x.copy() for x in X_refs]
        U_curr = [u.copy() for u in U_refs]
        change_hist: List[float] = []

        t0 = time.time()
        # Outer Nash loop with progress bar
        outer_iter = trange(self.max_iter, desc="Nash iters", disable=not show_progress)
        for it in outer_iter:
            max_change = 0.0
            if verbose:
                print(f"\n--- Nash outer iteration {it} ---")

            X_prev_all = [x.copy() for x in X_curr]

            # Inner per-agent loop with its own progress bar
            agent_iter = tqdm(range(self.N), desc=f" Agents @ it {it + 1}", disable=not show_progress, leave=False)
            for i in agent_iter:
                br = self.br_solvers[i]

                # 1) Linearise around current trajectory
                mats = self.fohs[i].calculate_discretization(X_curr[i], U_curr[i], sigma_ref)

                # 2) Gather neighbor trajectories
                neigh_cur = {j: X_curr[j] for j in range(self.N) if j != i}
                neigh_prev = {j: X_prev_all[j] for j in range(self.N) if j != i}

                # 3) Setup convex best-response problem
                br.setup(
                    X_ref=X_curr[i],
                    U_ref=U_curr[i],
                    sigma_ref=sigma_ref,
                    discr_mats=mats,
                    neighbour_refs=neigh_cur,
                    X_prev=X_prev_all[i],
                    neighbour_prev_refs=neigh_prev,
                    tr_radius=TRUST_RADIUS0,
                )

                # 4) ACS inner loop (update slabs & re-solve)
                for acs_it in range(self.max_acs_iters):
                    X_new, U_new, *_ = br.solve()
                    p_i = X_new[0:3, :]
                    neigh_prev_list = [X_curr[j][0:3, :] for j in range(self.N) if j != i]
                    br.model.update_slabs(p_i, neigh_prev_list)
                    if np.linalg.norm(X_new - X_curr[i]) < self.acs_tol:
                        break

                # 5) Accept update
                delta = np.linalg.norm(X_new - X_curr[i])
                max_change = max(max_change, delta)
                X_curr[i], U_curr[i] = X_new, U_new

                # 6) Verbose & progress postfix
                if verbose:
                    print(f"  Agent {i}: ΔX={delta:.2e}")
                if show_progress:
                    agent_iter.set_postfix({"ΔX": f"{delta:.2e}"})

            # record & check convergence
            change_hist.append(max_change)
            if show_progress:
                outer_iter.set_postfix({"max_change": f"{max_change:.2e}"})
            if verbose:
                print_iteration(it, 0.0, 0.0, max_change, 0.0, max_change, 0.0, sigma_ref, 0.0)
            if max_change < self.tol:
                if verbose:
                    print("Converged.")
                break

        # final summary
        if verbose:
            print_summary(len(change_hist), sigma_ref, time.time() - t0)
        return X_curr, U_curr, change_hist
