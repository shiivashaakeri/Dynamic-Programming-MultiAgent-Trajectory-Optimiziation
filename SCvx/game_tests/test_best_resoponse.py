"""Unit test for AgentBestResponse (Nash-game best-response step)."""

import numpy as np
import pytest

from SCvx.global_parameters import K
from SCvx.models.game_model import GameUnicycleModel
from SCvx.optimization.agent_best_response import AgentBestResponse


class DummyMultiAgent:
    """Minimal container to satisfy AgentBestResponse constructor."""

    def __init__(self, models):
        self.models = models
        self.N = len(models)


@pytest.fixture(scope="module")
def two_agent_dummy_model():
    # Create two GameUnicycleModel instances (no obstacles, collision_weight=0)
    m0 = GameUnicycleModel(
        r_init=np.array([0.0, 0.0, 0.0]),
        r_final=np.array([1.0, 0.0, 0.0]),
        obstacles=[],
        collision_weight=0.0,
    )
    m1 = GameUnicycleModel(
        r_init=np.array([1.0, 1.0, 0.0]),
        r_final=np.array([0.0, 1.0, 0.0]),
        obstacles=[],
        collision_weight=0.0,
    )
    return DummyMultiAgent([m0, m1])


def test_agent_best_response_shapes(two_agent_dummy_model):
    multi = two_agent_dummy_model

    # Initialise straight-line references for both agents
    X_refs = []
    U_refs = []
    for m in multi.models:
        X0 = np.zeros((m.n_x, K))
        U0 = np.zeros((m.n_u, K))
        for k in range(K):
            alpha = k / (K - 1)
            X0[:, k] = (1 - alpha) * m.x_init + alpha * m.x_final
        X_refs.append(X0)
        U_refs.append(U0)

    sigma_ref = 1.0

    # Test best-response for agent 0
    br = AgentBestResponse(0, multi)

    # Compute discretization for agent 0
    A_bar, B_bar, C_bar, S_bar, z_bar = br.foh.calculate_discretization(X_refs[0], U_refs[0], sigma_ref)

    neighbour_refs = {1: X_refs[1]}
    br.setup(
        X_ref=X_refs[0],
        U_ref=U_refs[0],
        sigma_ref=sigma_ref,
        discr_mats=(A_bar, B_bar, C_bar, S_bar, z_bar),
        neighbour_refs=neighbour_refs,
    )

    X_out, U_out, nu_out, slack, p_i = br.solve()

    # shape assertions
    assert X_out.shape == (multi.models[0].n_x, K)
    assert U_out.shape == (multi.models[0].n_u, K)
    assert p_i.shape == (2, K)
    assert np.isfinite(slack)

    # objective must be finite
    assert np.isfinite(br.scp.prob.value)  # type: ignore
