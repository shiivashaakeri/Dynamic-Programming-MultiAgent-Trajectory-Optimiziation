import numpy as np
import pytest

from SCvx.optimization.admm_utils import dual_residual, primal_residual, update_rho_admm


def test_primal_residual_zero():
    p = np.zeros((2, 5))
    Y = np.zeros((2, 5))
    assert primal_residual(p, Y) == 0.0


def test_primal_residual_nonzero():
    p = np.array([[1,2,3,4,5],[0,0,0,0,0]], dtype=float)
    Y = np.zeros_like(p)
    expected = np.linalg.norm(p)
    assert pytest.approx(primal_residual(p, Y), rel=1e-6) == expected


def test_dual_residual():
    Y_old = np.zeros((2, 4))
    Y_new = np.array([[1,1,1,1],[2,2,2,2]], dtype=float)
    expected = np.linalg.norm(Y_new - Y_old)
    assert pytest.approx(dual_residual(Y_new, Y_old), rel=1e-6) == expected


def test_update_rho_admm_increase():
    rho0 = 1.0
    # primal much larger than dual
    rho1 = update_rho_admm(rho0, primal_res=100.0, dual_res=1.0, mu=10.0, tau_inc=3.0, tau_dec=2.0)
    assert rho1 == rho0 * 3.0


def test_update_rho_admm_decrease():
    rho0 = 10.0
    # dual much larger than primal
    rho1 = update_rho_admm(rho0, primal_res=1.0, dual_res=100.0, mu=10.0, tau_inc=3.0, tau_dec=5.0)
    assert rho1 == rho0 / 5.0


def test_update_rho_admm_no_change():
    rho0 = 2.0
    # residuals balanced
    rho1 = update_rho_admm(rho0, primal_res=10.0, dual_res=1.0, mu=100.0, tau_inc=4.0, tau_dec=4.0)
    assert rho1 == rho0

if __name__ == "__main__":
    pytest.main()
