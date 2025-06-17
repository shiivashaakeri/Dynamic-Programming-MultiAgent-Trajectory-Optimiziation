"""Unit test for NashSolver on a minimal two-agent game scenario."""

import numpy as np
import pytest

from SCvx.global_parameters import K
from SCvx.models.game_model import GameUnicycleModel
from SCvx.models.multi_agent_model import MultiAgentModel
from SCvx.optimization.nash_solver import NashSolver


def straight_line(init, final):
    """Return straight-line X (3.K) and zero U (2.K)."""
    X = np.zeros((3, K))
    for k in range(K):
        a = k / (K - 1)
        X[:, k] = (1 - a) * init + a * final
    U = np.zeros((2, K))
    return X, U


@pytest.fixture(scope="module")
def two_agent_setup():
    # Create two simple Game models (no obstacles, collision_weight=0 to keep DCP)
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
    mam = MultiAgentModel(
        [
            {"r_init": m0.x_init, "r_final": m0.x_final, "obstacles": []},
            {"r_init": m1.x_init, "r_final": m1.x_final, "obstacles": []},
        ]
    )
    mam.models[0] = m0  # overwrite with game models containing weights
    mam.models[1] = m1
    return mam


def test_nash_solver_converges(two_agent_setup):
    mam = two_agent_setup

    # straight-line initial guesses
    X0_list, U0_list = [], []
    for mdl in mam.models:
        X0, U0 = straight_line(mdl.x_init, mdl.x_final)
        X0_list.append(X0)
        U0_list.append(U0)

    solver = NashSolver(mam, max_iter=5, tol=1e-2)
    X_fin, U_fin, hist = solver.solve(X0_list, U0_list, sigma_ref=1.0, verbose=False)

    # basic assertions
    assert len(X_fin) == 2 and len(U_fin) == 2
    for X in X_fin:
        assert X.shape == (3, K)
    for U in U_fin:
        assert U.shape == (2, K)

    # ensure we ran at least 1 iteration and convergence metric is finite
    assert len(hist) >= 1 and np.isfinite(hist[-1])
