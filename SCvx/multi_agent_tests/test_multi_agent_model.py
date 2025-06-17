import numpy as np
import pytest

from SCvx.global_parameters import K
from SCvx.models.multi_agent_model import MultiAgentModel
from SCvx.models.unicycle_model import UnicycleModel


def test_multi_agent_model_initialization():
    # Create two agent parameter dicts
    agent_params = [
        {"r_init": np.array([0.0, 0.0, 0.0]), "r_final": np.array([1.0, 1.0, 0.0])},
        {"r_init": np.array([2.0, 2.0, 0.0]), "r_final": np.array([3.0, 3.0, 0.0])},
    ]
    mam = MultiAgentModel(agent_params, d_min=1.5)
    # Check number of agents
    assert mam.N == 2

    assert all(isinstance(m, UnicycleModel) for m in mam.models)
    # d_min is set correctly
    assert mam.d_min == 1.5


def test_get_local_dynamics_shapes():
    # Single agent, but still wrap in multi-agent
    agent_params = [{"r_init": np.array([0, 0, 0]), "r_final": np.array([1, 1, 0])}]
    mam = MultiAgentModel(agent_params)
    f, A, B = mam.get_local_dynamics(0)
    # Test f, A, B on a sample input
    x = np.array([0.1, -0.2, 0.3])
    u = np.array([0.5, -0.1])
    fx = f(x, u)
    assert isinstance(fx, np.ndarray)
    # f may return shape (3,1) or (3,)
    assert fx.shape in [(3,), (3, 1)]
    Ax = A(x, u)
    Bu = B(x, u)
    assert Ax.shape == (3, 3)
    assert Bu.shape == (3, 2)


def test_linearize_collision_correctness():
    # Two agents: one at y=d_min, one at origin
    d_min = 2.0
    agent_params = [
        {"r_init": np.array([0, d_min, 0]), "r_final": np.array([0, d_min, 0])},
        {"r_init": np.array([0, 0, 0]), "r_final": np.array([0, 0, 0])},
    ]
    mam = MultiAgentModel(agent_params, d_min=d_min)
    # Create reference trajectories: constant positions
    X_ref_i = np.tile(np.array([[0.0], [d_min], [0.0]]), (1, K))
    X_ref_j = np.tile(np.array([[0.0], [0.0], [0.0]]), (1, K))
    A_ij, b_ij = mam.linearize_collision(0, 1, X_ref_i, X_ref_j)
    # A_ij should be [0,1]^T for all k
    expected_a = np.array([0.0, 1.0])
    for k in range(K):
        assert np.allclose(A_ij[:, k], expected_a, atol=1e-6)
        # b = d_min + a^T p_j = d_min + 0
        assert pytest.approx(b_ij[k], rel=1e-6) == d_min


if __name__ == "__main__":
    pytest.main()
