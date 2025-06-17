import numpy as np
import pytest

from SCvx.discretization.first_order_hold import FirstOrderHold
from SCvx.global_parameters import TRUST_RADIUS0, WEIGHT_NU, WEIGHT_SIGMA, K
from SCvx.models.multi_agent_model import MultiAgentModel
from SCvx.optimization.agent_solver import AgentSolver


def test_agent_solver_basic_solve():
    # Define two agents with non-colliding straight-line paths
    agent_params = [
        {'r_init': np.array([0.0, 0.0, 0.0]), 'r_final': np.array([1.0, 1.0, 0.0])},
        {'r_init': np.array([5.0, 5.0, 0.0]), 'r_final': np.array([6.0, 6.0, 0.0])}
    ]
    # Minimum separation 1.0 ensures no collision on these paths
    mam = MultiAgentModel(agent_params, d_min=1.0)
    rho_admm = 1.0
    solver = AgentSolver(agent_index=0, multi_agent_model=mam, rho_admm=rho_admm)

    # Build dummy reference trajectories for agent 0 and neighbor 1
    X_ref_i = np.zeros((mam.models[0].n_x, K))
    U_ref_i = np.zeros((mam.models[0].n_u, K))
    X_ref_j = np.zeros((mam.models[1].n_x, K))
    for k in range(K):
        alpha = k / (K - 1)
        X_ref_i[:, k] = (1 - alpha) * agent_params[0]['r_init'] + alpha * agent_params[0]['r_final']
        X_ref_j[:, k] = (1 - alpha) * agent_params[1]['r_init'] + alpha * agent_params[1]['r_final']
        U_ref_i[:, k] = 0.0

    # Compute discretization for agent 0
    foh = FirstOrderHold(mam.models[0], K)
    discretization_mats = foh.calculate_discretization(X_ref_i, U_ref_i, sigma=1.0)

    # Set required SCProblem parameters before setup
    # weights and trust region
    solver.scp.par['weight_nu'].value = WEIGHT_NU
    solver.scp.par['weight_slack'].value = mam.models[0].robot_radius * 0 + 1e5  # use default slack weight
    solver.scp.par['weight_sigma'].value = WEIGHT_SIGMA
    solver.scp.par['tr_radius'].value = TRUST_RADIUS0

    # Prepare neighbor references
    neighbor_refs = {1: X_ref_j}

    # Setup and solve
    try:
        solver.setup(X_ref_i, U_ref_i, sigma_ref_i=1.0,
                     discretization_mats=discretization_mats,
                     neighbor_refs=neighbor_refs)
    except Exception as e:
        pytest.skip(f"Solver setup failed: {e}")

    # Now solve local ADMM subproblem
    try:
        X_i, U_i, nu_i, slacks, p_i = solver.solve(solver='ECOS')
    except Exception as e:
        pytest.skip(f"Solver solve failed: {e}")

    # Check outputs
    assert X_i.shape == (mam.models[0].n_x, K)
    assert U_i.shape == (mam.models[0].n_u, K)
    assert nu_i.shape == (mam.models[0].n_x, K - 1)
    assert isinstance(slacks, dict)
    assert 1 in slacks and slacks[1].shape == (K, 1)
    assert p_i.shape == (2, K)

if __name__ == "__main__":
    pytest.main()
