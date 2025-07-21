"""Utility routines for analysing multi-agent trajectories."""

from typing import List, Tuple

import numpy as np


def min_inter_agent_distance(X_list: List[np.ndarray]) -> Tuple[float, np.ndarray]:
    """
    Compute the minimum Euclidean distance between every pair of agents
    over the whole horizon, in full 3D.
    """
    N = len(X_list)
    d_mat = np.zeros((N, N))

    for i in range(N):
        # take x,y,z rows instead of only x,y
        p_i = X_list[i][0:3, :]   # shape (3, K)
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
