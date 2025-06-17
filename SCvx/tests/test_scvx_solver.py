import numpy as np
import pytest

from SCvx.global_parameters import K
from SCvx.models.unicycle_model import UnicycleModel
from SCvx.optimization.scvx_solver import SCVXSolver


def test_scvx_solver_runs_and_logs():
    # Set up model and solver
    model = UnicycleModel()
    solver = SCVXSolver(model)
    # Run solver with verbose off and default settings
    X_sol, U_sol, sigma_sol, logger = solver.solve(verbose=False, initial_sigma=1.0)

    # Check solution shapes
    assert X_sol.shape == (model.n_x, K)
    assert U_sol.shape == (model.n_u, K)
    # sigma should be scalar
    assert np.isscalar(sigma_sol) or (isinstance(sigma_sol, np.ndarray) and sigma_sol.shape == ())

    # Logger should have recorded at least one iteration
    assert hasattr(logger, "records")
    assert isinstance(logger.records, list)
    assert len(logger.records) >= 1

    # Verify metrics in records
    for rec in logger.records:
        assert "iter" in rec
        assert "nu_norm" in rec
        assert "slack_norm" in rec
        assert "dx" in rec
        assert "du" in rec
        assert "ds" in rec
        assert "sigma" in rec

    # Defect and slack norms should be non-negative and finite
    nu_norms = [rec["nu_norm"] for rec in logger.records]
    slack_norms = [rec["slack_norm"] for rec in logger.records]
    assert all(n >= 0 and np.isfinite(n) for n in nu_norms)
    assert all(s >= 0 and np.isfinite(s) for s in slack_norms)

    # Ensure final state is within bounds
    final_position = X_sol[:2, -1]
    assert np.all(final_position <= model.upper_bound + model.robot_radius)
    assert np.all(final_position >= model.lower_bound - model.robot_radius)


if __name__ == "__main__":
    pytest.main()
