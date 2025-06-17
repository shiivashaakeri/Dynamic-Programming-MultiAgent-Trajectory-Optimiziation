import cvxpy as cvx
import numpy as np

from SCvx.global_parameters import TRUST_RADIUS0, WEIGHT_NU, WEIGHT_SIGMA, WEIGHT_SLACK, K
from SCvx.models.multi_agent_model import MultiAgentModel
from SCvx.optimization.admm_utils import WEIGHT_COLLISION_SLACK
from SCvx.optimization.sc_problem import SCProblem


class AgentSolver:
    """
    Agent-local solver for multi-agent SCvx using ADMM.

    Wraps a SCProblem for one agent and adds inter-agent collision constraints,
    consensus variables, and dual updates.
    """

    def __init__(self, agent_index: int, multi_agent_model: MultiAgentModel, rho_admm: float):
        self.i = agent_index
        self.multi_agent_model = multi_agent_model
        self.model_i = multi_agent_model.models[self.i]
        self.K = K
        self.d_min = multi_agent_model.d_min

        # Base SCProblem for agent i
        self.scp = SCProblem(self.model_i)

        # ADMM parameters
        self.rho_admm = rho_admm

        # Consensus and dual variables per neighbor
        self.Y = {}  # neighbor position estimates (2 x K)
        self.Lambda = {}  # dual multipliers (2 x K)
        self.S = {}  # slack variables (K x 1)

        for j in range(multi_agent_model.N):
            if j == self.i:
                continue
            self.Y[j] = cvx.Parameter((2, self.K))
            self.Lambda[j] = cvx.Parameter((2, self.K))
            self.S[j] = cvx.Variable((self.K, 1), nonneg=True)

    def setup(
        self,
        X_ref_i: np.ndarray,
        U_ref_i: np.ndarray,
        sigma_ref_i: float,
        discretization_mats: tuple,
        neighbor_refs: dict,
    ):
        """
        Prepare the local convex problem for agent i.

        Args:
            X_ref_i: (n_x x K) reference states of agent i
            U_ref_i: (n_u x K) reference inputs of agent i
            sigma_ref_i: reference time scaling
            discretization_mats: (A_bar, B_bar, C_bar, S_bar, z_bar)
            neighbor_refs: dict {j: X_ref_j}
        """
        A_bar, B_bar, C_bar, S_bar, z_bar = discretization_mats

        # Set base SCProblem parameters (use global defaults)
        self.scp.set_parameters(
            A_bar=A_bar,
            B_bar=B_bar,
            C_bar=C_bar,
            S_bar=S_bar,
            z_bar=z_bar,
            X_ref=X_ref_i,
            U_ref=U_ref_i,
            sigma_ref=sigma_ref_i,
            weight_nu=WEIGHT_NU,
            weight_slack=WEIGHT_SLACK,
            weight_sigma=WEIGHT_SIGMA,
            tr_radius=TRUST_RADIUS0,
        )

        # Build inter-agent collision constraints and augmented terms
        collision_cons = []
        aug_obj = 0
        for j, X_ref_j in neighbor_refs.items():
            # Linearize collision for each timestep
            A_ij, b_ij = self.multi_agent_model.linearize_collision(self.i, j, X_ref_i, X_ref_j)
            for k in range(self.K):
                p_i_k = self.scp.var["X"][0:2, k]
                y_j_k = self.Y[j][:, k]
                s_j_k = self.S[j][k, 0]
                # a^T(p_i - y_j) + s >= d_min
                collision_cons.append(A_ij[:, k].T @ (p_i_k - y_j_k) + s_j_k >= self.d_min)
            # Augmented Lagrangian on consensus
            diff = self.scp.var["X"][0:2, :] - self.Y[j]
            aug_obj += cvx.sum(cvx.multiply(self.Lambda[j], diff))
            aug_obj += (self.rho_admm / 2) * cvx.sum_squares(diff)
            aug_obj += WEIGHT_COLLISION_SLACK * cvx.sum(self.S[j])

        # Combine objectives
        base_obj = self.scp.prob.objective.args[0]
        total_obj = cvx.Minimize(base_obj + aug_obj)

        # Form the full problem
        self.prob = cvx.Problem(total_obj, self.scp.prob.constraints + collision_cons)

    def solve(self, **kwargs):
        """
        Solve the local ADMM subproblem.

        Returns:
            X_i, U_i, nu_i, slacks, p_i
        """
        self.prob.solve(**kwargs)
        X_i = self.scp.get_variable("X")
        U_i = self.scp.get_variable("U")
        nu_i = self.scp.get_variable("nu")
        slacks = {j: self.S[j].value for j in self.S}
        p_i = X_i[0:2, :]
        return X_i, U_i, nu_i, slacks, p_i
