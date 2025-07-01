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
    """Non-cooperative Nash via Iterative Best Response with ACS-based collision handling."""

    def __init__(
        self,
        multi_agent_model: MultiAgentModel,
        max_iter: int = 20,
        tol: float = 1e-3,
        max_acs_iters: int = 5,
        acs_tol: float = 1e-3,
    ) -> None:
        self.mam = multi_agent_model
        self.N = multi_agent_model.N
        self.max_iter = max_iter
        self.tol = tol

        # per-agent best response solvers
        self.br_solvers = [AgentBestResponse(i, multi_agent_model) for i in range(self.N)]
        # first-order hold discretizations
        self.fohs = [FirstOrderHold(m, K) for m in multi_agent_model.models]

        # ACS parameters
        self.max_acs_iters = max_acs_iters
        self.acs_tol = acs_tol

    def solve(  # noqa: C901
        self,
        X_refs: List[np.ndarray],
        U_refs: List[np.ndarray],
        sigma_ref: float = 1.0,
        verbose: bool = False,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
        """
        Args:
            X_refs: warm-start state trajectories for each agent (list of (3,K) arrays)
            U_refs: warm-start control trajectories for each agent (list of (2,K) arrays)
            sigma_ref: fixed time-scale parameter
            verbose: whether to print iteration details

        Returns:
            X_curr: converged state trajectories
            U_curr: converged control trajectories
            change_hist: history of max state-change per outer iteration
        """
        # initialize current trajectories
        X_curr = [x.copy() for x in X_refs]
        U_curr = [u.copy() for u in U_refs]
        change_hist: List[float] = []

        t0 = time.time()

        for it in range(self.max_iter):
            max_change = 0.0
            if verbose:
                print(f"\n--- Outer iteration {it} ---")

            # keep a snapshot for neighbor-reference in each agent's solve
            X_prev_all = [x.copy() for x in X_curr]

            # iterate over each agent's best response
            for i, br in enumerate(self.br_solvers):
                model = self.mam.models[i]
                # compute linearized dynamics matrices
                discr_mats = self.fohs[i].calculate_discretization(X_curr[i], U_curr[i], sigma_ref)

                # build neighbor data dictionaries
                neigh_cur = {j: X_curr[j] for j in range(self.N) if j != i}
                neigh_prev = {j: X_prev_all[j] for j in range(self.N) if j != i}

                # setup the convex subproblem with slab constraints
                br.setup(
                    X_ref=X_curr[i],
                    U_ref=U_curr[i],
                    sigma_ref=sigma_ref,
                    discr_mats=discr_mats,
                    neighbour_refs=neigh_cur,
                    X_prev=X_prev_all[i],
                    neighbour_prev_refs=neigh_prev,
                    tr_radius=TRUST_RADIUS0,
                )

                # ACS inner loop: alternate primal solve and dual slab updates
                X_new, U_new = None, None
                for acs_it in range(self.max_acs_iters):
                    # primal solve with fixed z-vectors
                    X_new, U_new, *_ = br.solve()

                    # extract current agent positions and neighbor positions
                    p_i = X_new[0:2, :]
                    neigh_prev_list = [X_curr[j][0:2, :] for j in range(self.N) if j != i]

                    # dual update: recompute slab multipliers z for each neighbor/time
                    model.update_slabs(p_i, neigh_prev_list)

                    # check ACS primal convergence
                    delta_acs = np.linalg.norm(X_new - X_curr[i])
                    if delta_acs < self.acs_tol:
                        break

                # after ACS, accept the new trajectory
                delta = np.linalg.norm(X_new - X_curr[i])
                max_change = max(max_change, delta)

                if verbose:
                    cost_i = br.scp.prob.value
                    print(f"  Agent {i}: cost={cost_i:8.3f}, ΔX={delta:6.2e}")

                X_curr[i], U_curr[i] = X_new, U_new

            # record convergence history and optionally print
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

            # outer-loop termination
            if max_change < self.tol:
                if verbose:
                    print("Converged.")
                break

        # final summary
        if verbose:
            print_summary(len(change_hist), sigma_ref, time.time() - t0)

        return X_curr, U_curr, change_hist
