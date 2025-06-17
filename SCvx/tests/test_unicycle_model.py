import cvxpy as cvx
import numpy as np
import pytest

from SCvx.global_parameters import K
from SCvx.models.base_model import BaseModel
from SCvx.models.unicycle_model import UnicycleModel


def test_unicycle_inheritance():
    # UnicycleModel should be subclass of BaseModel
    model = UnicycleModel()
    assert isinstance(model, BaseModel)


def test_get_equations_shapes():
    model = UnicycleModel()
    f, A, B = model.get_equations()
    # Test shapes for dynamics and Jacobians
    x0 = np.array([0.0, 0.0, 0.0])
    u0 = np.array([0.5, 0.1])
    # f returns (3,1)
    fx = f(x0, u0)
    assert isinstance(fx, np.ndarray)
    assert fx.shape in [(3,1), (3,)]
    # A returns (3,3)
    Ax = A(x0, u0)
    assert Ax.shape == (3,3)
    # B returns (3,2)
    Bu = B(x0, u0)
    assert Bu.shape == (3,2)


def test_initialize_trajectory():
    model = UnicycleModel()
    # Create arrays
    X = np.zeros((model.n_x, K))
    U = np.zeros((model.n_u, K))
    X_init, U_init = model.initialize_trajectory(X.copy(), U.copy())
    # Check endpoints
    np.testing.assert_allclose(X_init[:,0], model.x_init)
    np.testing.assert_allclose(X_init[:,-1], model.x_final)
    # Check controls are zero
    assert np.all(U_init == 0)


def test_get_constraints_and_objective():
    model = UnicycleModel()
    # CVXPY variables and parameters
    X_var = cvx.Variable((model.n_x, K))
    U_var = cvx.Variable((model.n_u, K))
    X_ref = cvx.Parameter((model.n_x, K))
    U_ref = cvx.Parameter((model.n_u, K))
    # Assign some dummy values to reference
    X_ref.value = np.zeros((model.n_x, K))
    U_ref.value = np.zeros((model.n_u, K))
    # Get constraints and objective
    constraints = model.get_constraints(X_var, U_var, X_ref, U_ref)
    objective = model.get_objective(X_var, U_var, X_ref, U_ref)
    # Basic checks
    assert isinstance(constraints, list)
    assert all(isinstance(c, cvx.constraints.constraint.Constraint) for c in constraints)
    # Objective should be a cvx.Minimize
    assert isinstance(objective, cvx.Minimize)

if __name__ == "__main__":
    pytest.main()
