import time

import numpy as np

from SCvx.discretization.first_order_hold import FirstOrderHold
from SCvx.global_parameters import K
from SCvx.models.SI_multi_agent_model import SI_MultiAgentModel
from SCvx.optimization.admm_utils import dual_residual, primal_residual
from SCvx.optimization.si_agent_solver import SI_AgentSolver
from SCvx.utils.multi_agent_logging import print_iteration, print_summary


class SI_ADMMCoordinator:
    """
    Coordinates multiple SI_AgentSolver instances via ADMM for 3D multi-agent SCvx.

    Attributes:
        model           : SI_MultiAgentModel
        rho_admm        : ADMM penalty parameter
        max_iter        : maximum ADMM rounds
        agent_solvers   : list of SI_AgentSolver (one per agent)
        discretizers    : list of FirstOrderHold (one per agent)
    """

    def __init__(self,
                 multi_agent_model: SI_MultiAgentModel,
                 rho_admm: float = 1.0,
                 max_iter: int = 10):
        self.model = multi_agent_model
        self.N = multi_agent_model.N
        self.rho_admm = rho_admm
        self.max_iter = max_iter

        # Create per-agent solvers and discretizers
        self.agent_solvers = []
        self.discretizers = []
        for i in range(self.N):
            solver = SI_AgentSolver(i, multi_agent_model, rho_admm)
            self.agent_solvers.append(solver)
            self.discretizers.append(FirstOrderHold(multi_agent_model.models[i], K))

    def solve(self,
              X_refs: list,
              U_refs: list,
              sigma_ref: float,
              verbose: bool = True):
        """
        Run ADMM rounds to coordinate agent trajectories in 3D.

        Args:
            X_refs   : list of np.ndarray, each (n_x x K) initial trajectories
            U_refs   : list of np.ndarray, each (n_u x K) initial inputs
            sigma_ref: shared time scaling
            verbose  : whether to print per-iteration logs

        Returns:
            X_list     : list of updated state trajectories
            U_list     : list of updated control trajectories
            sigma_ref  : unchanged time scaling
            primal_hist: list of average primal residuals per ADMM round
            dual_hist  : list of average dual residuals per ADMM round
        """
        # Initialize consensus and duals for each solver
        for solver in self.agent_solvers:
            for j in solver.Y:
                # 3D consensus init
                solver.Y[j].value = X_refs[j][0:3, :]
                solver.Lambda[j].value = np.zeros((3, K))

        X_curr = list(X_refs)
        U_curr = list(U_refs)
        primal_hist = []
        dual_hist = []

        t0 = time.time()
        for it in range(self.max_iter):
            # --- 1) Local solves ---
            new_positions = [None] * self.N
            for i, solver in enumerate(self.agent_solvers):
                mats = self.discretizers[i].calculate_discretization(
                    X_curr[i], U_curr[i], sigma_ref)
                neighbor_refs = {j: X_curr[j] for j in range(self.N) if j != i}
                solver.setup(
                    X_refs[i], U_refs[i], sigma_ref,
                    mats, neighbor_refs
                )
                X_i, U_i, nu_i, slacks_i, p_i = solver.solve(solver="ECOS")
                X_curr[i], U_curr[i] = X_i, U_i
                new_positions[i] = p_i

            # --- 2) Consensus & dual updates ---
            pr_vals = []
            du_vals = []
            for solver in self.agent_solvers:
                for j in solver.Y:
                    p_j = new_positions[j]
                    Y_old = solver.Y[j].value
                    Y_new = 0.5 * (Y_old + p_j)
                    solver.Y[j].value = Y_new
                    solver.Lambda[j].value += self.rho_admm * (p_j - Y_new)
                    pr_vals.append(primal_residual(p_j, Y_new))
                    du_vals.append(dual_residual(Y_new, Y_old))

            pr_avg = float(np.mean(pr_vals))
            du_avg = float(np.mean(du_vals))
            primal_hist.append(pr_avg)
            dual_hist.append(du_avg)

            if verbose:
                # dx & ds are zero since sigma is static in ADMM
                print_iteration(
                    it,
                    nu_norm=0.0,
                    slack_norm=0.0,
                    primal_res=pr_avg,
                    dual_res=du_avg,
                    dx=0.0,
                    ds=0.0,
                    sigma=sigma_ref,
                    tr_radius=self.rho_admm,
                )

        runtime = time.time() - t0
        if verbose:
            print_summary(len(primal_hist), sigma_ref, runtime)

        return X_curr, U_curr, sigma_ref, primal_hist, dual_hist
