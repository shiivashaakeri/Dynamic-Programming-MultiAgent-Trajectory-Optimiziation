import numpy as np
import pytest

from SCvx.discretization.first_order_hold import FirstOrderHold
from SCvx.global_parameters import K
from SCvx.models.unicycle_model import UnicycleModel


def test_discretization_shapes():
    model = UnicycleModel()
    foh = FirstOrderHold(model, K)
    # Create a simple reference trajectory: straight line interpolation for X, zero U
    X_ref = np.zeros((model.n_x, K))
    U_ref = np.zeros((model.n_u, K))
    # Endpoints for X_ref
    X_ref[:, 0] = model.x_init
    X_ref[:, -1] = model.x_final
    # Test sigma = 1.0
    sigma = 1.0
    A_bar, B_bar, C_bar, S_bar, z_bar = foh.calculate_discretization(X_ref, U_ref, sigma)
    # Check shapes
    assert A_bar.shape == (model.n_x * model.n_x, K - 1)
    assert B_bar.shape == (model.n_x * model.n_u, K - 1)
    assert C_bar.shape == (model.n_x * model.n_u, K - 1)
    assert S_bar.shape == (model.n_x, K - 1)
    assert z_bar.shape == (model.n_x, K - 1)


def test_integration_nonlinear_piecewise_shape():
    model = UnicycleModel()
    foh = FirstOrderHold(model, K)
    # Simple linear trajectory and zero U
    X_lin = np.zeros((model.n_x, K))
    U_ref = np.zeros((model.n_u, K))
    sigma = 1.0
    X_nl = foh.integrate_nonlinear_piecewise(X_lin, U_ref, sigma)
    # Should match shape
    assert X_nl.shape == (model.n_x, K)


def test_integration_nonlinear_full_shape():
    model = UnicycleModel()
    foh = FirstOrderHold(model, K)
    # initial state and zero U
    x0 = np.array(model.x_init)
    U_ref = np.zeros((model.n_u, K))
    sigma = 1.0
    X_nl_full = foh.integrate_nonlinear_full(x0, U_ref, sigma)
    assert X_nl_full.shape == (model.n_x, K)

if __name__ == "__main__":
    pytest.main()
