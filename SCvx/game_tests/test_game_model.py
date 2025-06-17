"""Pytest for GameUnicycleModel cost helper."""

import cvxpy as cvx
import numpy as np
import pytest

from SCvx.global_parameters import K
from SCvx.models.game_model import GameUnicycleModel


@pytest.fixture(scope="module")
def simple_game_model():
    r0 = np.array([0.0, 0.0, 0.0])
    rF = np.array([1.0, 1.0, 0.0])
    # Set collision_weight = 0 to keep expression DCP-compliant for this unit test
    return GameUnicycleModel(r_init=r0, r_final=rF, obstacles=[], collision_weight=0.0)


def test_cost_expression_shape(simple_game_model):
    m = simple_game_model
    X_v = cvx.Variable((3, K))
    U_v = cvx.Variable((2, K))

    # create two neighbour trajectories as parameters
    neighbour_params = []
    for _ in range(2):
        P = cvx.Parameter((2, K))
        P.value = np.random.randn(2, K) * 0.1  # small random offsets
        neighbour_params.append(P)

    cost_expr = m.get_cost_function(X_v, U_v, neighbour_params)

    # expression should be scalar
    assert cost_expr.shape == (), "Cost expression should be scalar."

    # build trivial problem just to ensure CVXPY accepts expression
    prob = cvx.Problem(cvx.Minimize(cost_expr))

    # give random initial values so solver has something finite
    X_v.value = np.tile(np.linspace(0, 1, K), (3, 1))
    U_v.value = np.zeros((2, K))

    # Solve with a fast conic solver
    prob.solve(solver="ECOS", warm_start=True)

    assert prob.status in {cvx.OPTIMAL, cvx.OPTIMAL_INACCURATE}, (
        "Problem did not solve to optimality.")

    # cost should be finite
    assert np.isfinite(prob.value), "Cost should be finite after solve."
