import numpy as np

# Penalty weight for inter-agent collision slack variables
# You can tune this separately from static obstacle slack weights
WEIGHT_COLLISION_SLACK = 1e5


def primal_residual(p_j: np.ndarray, Y_ij: np.ndarray) -> float:
    """
    Compute the primal residual for consensus: ||p_j - Y_ij||_2.

    Args:
        p_j: np.ndarray of shape (3, K), true 3D positions of agent j
        Y_ij: np.ndarray of shape (3, K), agent i's estimate of agent j's positions
    Returns:
        float: Euclidean norm of the residual
    """
    return np.linalg.norm(p_j - Y_ij)


def dual_residual(Y_new: np.ndarray, Y_old: np.ndarray) -> float:
    """
    Compute the dual residual for consensus updates: ||Y_new - Y_old||_2.

    Args:
        Y_new: np.ndarray of shape (3, K)
        Y_old: np.ndarray of shape (3, K)
    Returns:
        float: Euclidean norm of the residual
    """
    return np.linalg.norm(Y_new - Y_old)


def update_rho_admm(rho: float, primal_res: float, dual_res: float,
                    mu: float = 10.0, tau_inc: float = 2.0, tau_dec: float = 2.0) -> float:
    """
    Heuristic update of the ADMM penalty parameter rho based on residuals.

    If primal_res > mu * dual_res: increase rho by tau_inc.
    If dual_res > mu * primal_res: decrease rho by tau_dec.
    Otherwise leave unchanged.

    Args:
        rho: current rho value
        primal_res: primal residual
        dual_res: dual residual
        mu: threshold ratio
        tau_inc: factor to increase rho
        tau_dec: factor to decrease rho
    Returns:
        float: updated rho
    """
    if primal_res > mu * dual_res:
        return rho * tau_inc
    elif dual_res > mu * primal_res:
        return rho / tau_dec
    else:
        return rho
