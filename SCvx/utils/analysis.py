"""Utility routines for analysing multi-agent trajectories."""

from typing import List, Tuple

import numpy as np

from SCvx.utils.intersample_collision import make_segment_f


def min_inter_agent_distance(X_list: List[np.ndarray]) -> Tuple[float, np.ndarray]:
    """
    Compute the minimum Euclidean distance between every pair of agents
    over the whole horizon, in full 3D.
    """
    N = len(X_list)
    d_mat = np.zeros((N, N))

    for i in range(N):
        # take x,y,z rows instead of only x,y
        p_i = X_list[i][0:3, :]  # shape (3, K)
        for j in range(i + 1, N):
            p_j = X_list[j][0:3, :]  # shape (3, K)
            # distances at each time step
            d_ij_k = np.linalg.norm(p_i - p_j, axis=0)
            # minimum over time
            d_min_ij = d_ij_k.min()
            d_mat[i, j] = d_mat[j, i] = d_min_ij

    # global minimum (ignore zeros on diagonal)
    d_min_global = d_mat[d_mat > 0].min()
    return d_min_global, d_mat


def min_agent_obstacle_distance(
    X_list: List[np.ndarray], obstacles: List[Tuple[List[float], float]], robot_radius: float
) -> Tuple[float, np.ndarray]:
    """
    Compute the minimum clearance between each agent and each spherical obstacle
    over the whole horizon.

    Returns
    -------
    d_min_global : float
        The smallest agent-to-obstacle clearance (can be negative if penetrating).
    d_mat        : (N,M) ndarray
        d_mat[i,j] = min_k ( ‖p_i(k)-c_j‖ - (robot_radius + r_j) ).
    """
    N = len(X_list)
    M = len(obstacles)
    K = X_list[0].shape[1]
    d_mat = np.full((N, M), np.inf)

    for i in range(N):
        p_i = X_list[i][0:3, :]  # (3,K)
        for j, (centre, r_j) in enumerate(obstacles):
            c = np.array(centre)[:, None]  # (3,1)
            # distances at each timestep minus combined radii
            ds = np.linalg.norm(p_i - c, axis=0) - (robot_radius + r_j)
            d_mat[i, j] = ds.min()

    d_min_global = d_mat.min()
    return d_min_global, d_mat

def compute_intersample_clearance(
    X: np.ndarray,
    U: np.ndarray,
    foh,
    obs_center: np.ndarray,
    obs_radius: float,
    robot_radius: float,
    margin: float,
    resolution: int = 50
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute both discrete and continuous obstacle clearance along a trajectory.

    Returns:
      tau_cont: continuous time stamps (segment + fraction)
      h_cont:  clearance values at each sub-sample
      tau_disc: discrete time stamps (0, 1, ..., K-1)
      h_disc:  clearance at each knot point
    """
    K = X.shape[1]
    total_radius = obs_radius + robot_radius + margin

    # Continuous sampling
    tau_cont_list = []
    h_cont_list   = []
    for k in range(K-1):
        f_seg, _ = make_segment_f(foh, U[:,k], U[:,k+1], sigma=1.0)
        for t in np.linspace(0, 1, resolution, endpoint=False):
            pos       = f_seg(t)
            clearance = np.linalg.norm(pos - obs_center) - total_radius
            tau_cont_list.append(k + t)
            h_cont_list.append(clearance)

    tau_cont = np.array(tau_cont_list)
    h_cont   = np.array(h_cont_list)

    # Discrete clearance (at knot points)
    tau_disc = np.arange(K)
    h_disc   = np.linalg.norm(X.T - obs_center.reshape(1,3), axis=1) - total_radius

    return tau_cont, h_cont, tau_disc, h_disc
