"""Utility routines for analysing multi-agent trajectories."""

from typing import List, Tuple

import numpy as np


def min_inter_agent_distance(X_list: List[np.ndarray]) -> Tuple[float, np.ndarray]:
    """
    Compute the minimum Euclidean distance between every pair of agents
    over the whole horizon.

    Parameters
    ----------
    X_list  : list of shape-(3,K) numpy arrays
              State trajectories returned by the solver.

    Returns
    -------
    d_min_global : float
        The smallest distance encountered by any pair at any time step.
    d_matrix     : (N,N) ndarray
        d_matrix[i,j] = min_k ||p_i(k) - p_j(k)||  (with zeros on the diagonal).
    """
    N = len(X_list)
    K = X_list[0].shape[1]  # noqa: F841
    d_mat = np.zeros((N, N))

    for i in range(N):
        p_i = X_list[i][0:2, :]  # 2K
        for j in range(i + 1, N):
            p_j = X_list[j][0:2, :]
            # K distances for this pair
            d_ij_k = np.linalg.norm(p_i - p_j, axis=0)
            d_min_ij = d_ij_k.min()
            d_mat[i, j] = d_mat[j, i] = d_min_ij

    # Ignore diagonal zeros when taking global minimum
    d_min_global = d_mat[d_mat > 0].min()
    return d_min_global, d_mat
