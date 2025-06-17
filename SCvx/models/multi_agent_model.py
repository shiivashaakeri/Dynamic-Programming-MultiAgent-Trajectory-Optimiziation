import cvxpy as cvx
import numpy as np

from SCvx.global_parameters import K as GLOBAL_K
from SCvx.models.unicycle_model import UnicycleModel


class MultiAgentModel:
    """
    Wrapper for multiple single-agent UnicycleModel instances.

    Provides per-agent dynamics, static constraints, and inter-agent collision linearization.
    """

    def __init__(self, agent_params, d_min=1.0):
        """
        Initialize a multi-agent scenario.

        Args:
            agent_params: List of dicts, each with keys for UnicycleModel init ('r_init', 'r_final', etc.)
            d_min: Minimum allowed distance between any two agents.
        """
        self.N = len(agent_params)
        self.models = []
        for params in agent_params:
            # Build init kwargs for UnicycleModel
            kwargs = {}
            if "r_init" in params:
                kwargs["r_init"] = params["r_init"]
            if "r_final" in params:
                kwargs["r_final"] = params["r_final"]
            for key in ("v_max", "w_max", "bounds", "robot_radius"):
                if key in params and params[key] is not None:
                    kwargs[key] = params[key]
            # Instantiate model
            m = UnicycleModel(**kwargs)
            # Override obstacles if provided
            if "obstacles" in params and params["obstacles"] is not None:
                m.obstacles = params["obstacles"]
                # Re-create slack variables to match new obstacles count
                m.s_prime = [cvx.Variable((GLOBAL_K, 1), nonneg=True) for _ in m.obstacles]
            self.models.append(m)

        self.d_min = d_min

    def get_local_dynamics(self, i):
        """
        Return the dynamics functions for agent i.

        Returns:
            f_i, A_i, B_i: callables for f(x,u), df/dx, df/du
        """
        return self.models[i].get_equations()

    def get_static_constraints(self, i, X, U, X_ref, U_ref):
        """
        Get agent-specific static constraints (bounds, obstacles) for agent i.
        """
        return self.models[i].get_constraints(X, U, X_ref, U_ref)

    def linearize_collision(self, i, j, X_ref_i, X_ref_j):  # noqa: ARG002
        """
        Linearize the collision avoidance constraint between agents i and j.

        Original: ||p_i - p_j|| >= d_min
        Linearized at reference positions p_i_ref, p_j_ref.
        """
        p_i = X_ref_i[0:2, :]
        p_j = X_ref_j[0:2, :]
        K_steps = p_i.shape[1]
        A_ij = np.zeros((2, K_steps))
        b_ij = np.zeros(K_steps)
        for k in range(K_steps):
            diff = p_i[:, k] - p_j[:, k]
            norm_val = np.linalg.norm(diff) + 1e-6
            a = diff / norm_val
            A_ij[:, k] = a
            b_ij[k] = self.d_min + a.dot(p_j[:, k])
        return A_ij, b_ij
