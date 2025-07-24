from typing import Callable, List, Tuple

import numpy as np

# --- SCvx internals we rely on ---
from SCvx.utils.intersample_collision import h_i, make_segment_f

# ---------- universal wrapper for f_seg(t) vs f_seg(u_dummy,t) ------------
def call_f_seg(f_seg, t):
    """
    Evaluate the segment-interpolator returned by make_segment_f, regardless
    of its calling signature.
    """
    n_args = len(inspect.signature(f_seg).parameters)
    if n_args == 1:            #  f_seg(t)
        return f_seg(t)

    # Two-argument cases
    params = list(inspect.signature(f_seg).parameters)
    first = params[0]
    if first.startswith("_u") or first in {"u", "u_dummy", "dummy"}:
        return f_seg(None, t)  # f_seg(u_dummy, t)
    return f_seg(t, None)      # f_seg(t, u_dummy)

import inspect

def sample_segment_clearance(
    X: np.ndarray,                   # (3, K) states of ONE agent
    U: np.ndarray,                   # (3, K) controls of that agent
    k_seg: int,                      # segment index 0 … K-2
    foh,                             # FirstOrderHold for that agent
    obstacle: Tuple[np.ndarray, float],   # (center, radius)
    r_robot: float,
    margin: float,
    n_samples: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (ts, h_vals) where ts ∈ [0,1] and
    h(t) = ‖p(t) − c_obs‖ − (r_obs + r_robot + margin).
    Negative h(t) ⇒ collision penetration.
    """
    # 1) continuous transition model on this segment
    f_seg, _ = make_segment_f(foh, U[:, k_seg], U[:, k_seg + 1], sigma=1.0)

    c_obs, r_obs = obstacle
    r_tot = r_obs + r_robot + margin

    ts = np.linspace(0.0, 1.0, n_samples)
    h_vals: List[float] = []
    for t in ts:
        p_t = call_f_seg(f_seg, t)                  # robust call
        h_vals.append(np.linalg.norm(p_t - c_obs) - r_tot)

    return ts, np.asarray(h_vals)


import matplotlib.pyplot as plt   # add near the top of the file

def plot_segment_clearance(ts, h_vals, knot_vals, title=""):
    """
    Show discrete knot clearances vs continuous h(t).
    knot_vals = (h0, h1)
    """
    h0, h1 = knot_vals
    plt.figure(figsize=(6, 3.5))
    plt.plot(ts, h_vals, lw=2, label="continuous h(t)")
    plt.scatter([0, 1], [h0, h1], c="k", zorder=5, label="knots")
    plt.axhline(0, color="r", lw=1)
    plt.xlabel("t (segment local)"); plt.ylabel("clearance h [m]")
    plt.title(title + f"\nmin = {h_vals.min():.3f} m")
    plt.legend(); plt.tight_layout(); plt.show()

def min_clearances_both_grids(
    X: np.ndarray,                     # (3, K) states of ONE agent
    U: np.ndarray,                     # (3, K) controls of ONE agent
    foh,                               # FirstOrderHold for that agent
    obstacles: List[Tuple[np.ndarray, float]],
    r_robot: float,
    margin: float,
    fine_res: int = 50,                # samples per segment
) -> Tuple[float, float]:
    """
    Returns:
        min_h_knot  – min clearance over knot instants only
        min_h_fine  – min clearance over a fine grid inside every segment
    """
    K = X.shape[1]
    min_h_knot = np.inf
    min_h_fine = np.inf

    for k in range(K - 1):

        # ----- knot instants (t = 0 and 1 of segment) --------------------
        for idx in (k, k + 1):
            for c_obs, r_obs in obstacles:
                r_tot = r_obs + r_robot + margin
                h_k = np.linalg.norm(X[:, idx] - c_obs) - r_tot
                min_h_knot = min(min_h_knot, h_k)

        # ----- fine grid inside the segment ------------------------------
        f_seg, _ = make_segment_f(foh, U[:, k], U[:, k + 1], sigma=1.0)
        t_grid = np.linspace(0.0, 1.0, fine_res, endpoint=True)

        for t in t_grid:
            p_t = call_f_seg(f_seg, t)
            for c_obs, r_obs in obstacles:
                r_tot = r_obs + r_robot + margin
                h_t = np.linalg.norm(p_t - c_obs) - r_tot
                min_h_fine = min(min_h_fine, h_t)

    return min_h_knot, min_h_fine