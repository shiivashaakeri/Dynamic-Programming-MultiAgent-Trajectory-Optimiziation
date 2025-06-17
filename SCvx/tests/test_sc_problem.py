import numpy as np
import pytest

from SCvx.discretization.first_order_hold import FirstOrderHold
from SCvx.global_parameters import TRUST_RADIUS0, WEIGHT_NU, WEIGHT_SIGMA, WEIGHT_SLACK, K
from SCvx.models.unicycle_model import UnicycleModel
from SCvx.optimization.sc_problem import SCProblem


def test_sc_problem_trivial_solve():
    # Instantiate model and initial guess
    model = UnicycleModel()
    X0 = np.zeros((model.n_x, K))
    U0 = np.zeros((model.n_u, K))
    X0, U0 = model.initialize_trajectory(X0, U0)
    sigma0 = 1.0

    # Compute discretization matrices
    foh = FirstOrderHold(model, K)
    A_bar, B_bar, C_bar, S_bar, z_bar = foh.calculate_discretization(X0, U0, sigma0)

    # Set up SCProblem
    scp = SCProblem(model)
    scp.set_parameters(
        A_bar=A_bar,
        B_bar=B_bar,
        C_bar=C_bar,
        S_bar=S_bar,
        z_bar=z_bar,
        X_ref=X0,
        U_ref=U0,
        sigma_ref=sigma0,
        weight_nu=WEIGHT_NU,
        weight_slack=WEIGHT_SLACK,
        weight_sigma=WEIGHT_SIGMA,
        tr_radius=TRUST_RADIUS0,
    )

    # Solve the convex subproblem
    error = scp.solve(solver="ECOS", verbose=False)
    assert not error, "SCProblem solve failed"

    # Retrieve solution and check shapes
    X_val = scp.get_variable("X")
    U_val = scp.get_variable("U")
    nu_val = scp.get_variable("nu")
    sigma_val = scp.get_variable("sigma")

    assert X_val.shape == (model.n_x, K)
    assert U_val.shape == (model.n_u, K)
    assert nu_val.shape == (model.n_x, K - 1)
    # sigma is scalar
    assert np.isscalar(sigma_val) or (isinstance(sigma_val, np.ndarray) and sigma_val.shape == ())


if __name__ == "__main__":
    pytest.main()
