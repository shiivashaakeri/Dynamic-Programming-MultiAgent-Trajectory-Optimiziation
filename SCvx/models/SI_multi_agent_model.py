import cvxpy as cvx
import numpy as np

from SCvx.models.single_integrator_model import SingleIntegratorModel


class SI_MultiAgentModel:  # noqa: N801
    """
    Wrapper for multiple single-agent SingleIntegratorModel instances.

    Provides per-agent dynamics, static constraints, and inter-agent collision linearization in 3D.
    """

    _ALLOWED_KEYS = {"r_init", "r_final", "v_max", "bounds", "robot_radius", "obstacles"}

    def __init__(self, agent_params: list, d_min: float = 1.0):
        self.N = len(agent_params)
        self.models = []
        for params in agent_params:
            basic = {k: v for k, v in params.items() if k in self._ALLOWED_KEYS}
            self.models.append(SingleIntegratorModel(**basic))
        self.d_min = d_min

    def get_local_dynamics(self, i: int):
        """
        Return the dynamics functions for agent i.

        Returns:
            f_i, A_i, B_i: callables for f(x,u), df/dx, df/du
        """
        return self.models[i].get_equations()

    def get_static_constraints(
        self, i: int, X: cvx.Variable, U: cvx.Variable, X_ref: cvx.Parameter, U_ref: cvx.Parameter
    ) -> list:
        """
        Get agent-specific static constraints (workspace bounds, obstacle avoidance) for agent i.
        """
        return self.models[i].get_constraints(X, U, X_ref, U_ref)

    def get_objective(
        self, i: int, X: cvx.Variable, U: cvx.Variable, X_ref: cvx.Parameter, U_ref: cvx.Parameter
    ) -> cvx.Expression:
        """
        Get agent-specific objective for agent i.
        """
        return self.models[i].get_objective(X, U, X_ref, U_ref)

    def linearize_inter_agent_collision(self, i: int, j: int, X_ref_i: cvx.Parameter, X_ref_j: cvx.Parameter) -> tuple:  # noqa: ARG002
        """
        Linearize the collision avoidance constraint between agents i and j in 3D.

        Original: ||p_i - p_j||_2 >= d_min
        Linearized at reference positions p_i_ref, p_j_ref.

        Returns:
            A_ij: numpy array of shape (3, K) containing normal vectors at each timestep
            b_ij: numpy array of length K containing offsets for linear constraints
        """
        # Extract 3D positions from reference trajectories
        p_i = X_ref_i[0:3, :]
        p_j = X_ref_j[0:3, :]
        K_steps = p_i.shape[1]

        A_ij = np.zeros((3, K_steps))
        b_ij = np.zeros(K_steps)
        for k in range(K_steps):
            diff = p_i[:, k] - p_j[:, k]
            norm_val = np.linalg.norm(diff) + 1e-6
            a = diff / norm_val
            A_ij[:, k] = a
            # Constraint: a^T p_i >= d_min + a^T p_j
            b_ij[k] = self.d_min + a.dot(p_j[:, k])
        return A_ij, b_ij
