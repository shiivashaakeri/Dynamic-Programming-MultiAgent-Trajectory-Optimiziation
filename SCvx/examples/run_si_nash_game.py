# #!/usr/bin/env python3
# """
# Runs the 3D Nash‐game demo with single‐integrator agents.
# This script sets up a multi‐agent scenario, solves for the Nash equilibrium
# trajectories, and visualizes the results.
# """

# import os

# import matplotlib.pyplot as plt
# import numpy as np
# from tqdm import tqdm

# # --- Core SCvx Imports ---
# from SCvx.config.SI_default_game import AGENT_PARAMS, CLEARANCE, MARGIN, K
# from SCvx.models.game_si_model import GameSIModel
# from SCvx.models.SI_multi_agent_model import SI_MultiAgentModel
# from SCvx.optimization.si_nash_solver import SI_NashSolver
# from SCvx.utils.analysis import compute_intersample_clearance, min_inter_agent_distance
# from SCvx.utils.IS_initial_guess import initial_guess

# # --- Visualization Imports ---
from SCvx.visualization.multi_agent_plot_utils import (
    animate_multi_agent_quadrotors,
    animate_multi_agent_spheres,
    animate_trajectories_3d,
)
# from SCvx.visualization.plot_utils import plot_convergence

# import inspect   # <-- make sure this is present with the other imports

# # -------------------------------------------------------------------------
# # Robust wrapper for SCvx's make_segment_f variant chaos
# # -------------------------------------------------------------------------
# def call_f_seg(f_seg, t):
#     """
#     Evaluate the position function returned by make_segment_f, regardless
#     of whether it expects 1 argument (t) or 2 (u_dummy, t) – and regardless
#     of the order of those two arguments.
#     """
#     n_args = len(inspect.signature(f_seg).parameters)

#     if n_args == 1:                     # variant A: f_seg(t)
#         return f_seg(t)

#     if n_args == 2:                     # variant B or C
#         first = next(iter(inspect.signature(f_seg).parameters))
#         if first.startswith("_u") or first in {"u", "dummy", "u_dummy"}:
#             return f_seg(None, t)       # variant B: f_seg(u_dummy, t)
#         return f_seg(t, None)           # variant C: f_seg(t, u_dummy)

#     raise RuntimeError("Unexpected f_seg signature (expected 1 or 2 args).")

# def build_multi_agent_model() -> SI_MultiAgentModel:
#     """Initializes the multi‐agent model and wraps each agent in a GameSIModel."""
#     mam = SI_MultiAgentModel(AGENT_PARAMS)
#     for idx, p in enumerate(AGENT_PARAMS):
#         mam.models[idx] = GameSIModel(
#             r_init=p["r_init"],
#             r_final=p["r_final"],
#             obstacles=p["obstacles"],
#             robot_radius=p.get("robot_radius", 0.5),
#             control_weight=p.get("control_weight", 1.0),
#             collision_weight=p.get("collision_weight", 80.0),
#             collision_radius=p.get("collision_radius", 0.5),
#             control_rate_weight=p.get("control_rate_weight", 5.0),
#             curvature_weight=p.get("curvature_weight", 0.0),
#         )
#     return mam


# def main() -> None:
#     # 1. --- Model Setup and Initial Guess ---
#     mam = build_multi_agent_model()

#     X0_list, U0_list = [], []
#     for p in tqdm(AGENT_PARAMS, desc="Initial guesses"):
#         X0, U0 = initial_guess(p["r_init"], p["r_final"], p["obstacles"], CLEARANCE, K)
#         X0_list.append(X0)
#         U0_list.append(U0)

#     # 2. --- Solve for Nash Equilibrium ---
#     solver = SI_NashSolver(mam, max_iter=25, tol=1e-3)
#     # 1.5 --- Pre-solve inter-sample collision scan on the **initial guess** ---
#     print("\n=== Pre-solve inter-sample collision check (initial guess) ===")
#     pre_flag = False

#     #  solver.fohs is available immediately after construction
#     for i in range(len(AGENT_PARAMS)):                           # loop over agents
#         foh_i = solver.fohs[i]
#         for k in range(K - 1):                                   # loop over segments
#             f_seg, _ = make_segment_f(
#                 foh_i,
#                 U0_list[i][:, k], U0_list[i][:, k + 1],
#                 sigma=1.0,
#             )
#             for obs_idx, obs in enumerate(mam.models[i].obstacles):
#                 roots = find_critical_times(
#                     xk=X0_list[i][:, k],
#                     uk=U0_list[i][:, k],        # velocity entry – ignored by f_seg
#                     f=f_seg,
#                     T=np.eye(3),
#                     obstacle=obs,
#                     dt=1.0,
#                 )
#                 if roots:
#                     pre_flag = True
#                     print(f"  ⚠ Agent {i}, seg {k}, obs {obs_idx}, t* = {roots}")

#     if not pre_flag:
#         print("  ✔ No mid-sample collisions in the *initial guess*.")
#     print("=== End pre-solve check ===\n")
#     X_fin, U_fin, hist = solver.solve(X0_list, U0_list, sigma_ref=1.0, verbose=False, show_progress=True)
#     from SCvx.visualization.intersample_plot_utils import sample_segment_clearance
#     center, rad = mam.models[0].obstacles[0]
#     ts, h = sample_segment_clearance(X_fin[0], U_fin[0], 39, solver.fohs[0], (np.array(center), rad))
#     print("min h on seg 39 =", h.min())
#     from SCvx.visualization.intersample_plot_utils import (
#         sample_segment_clearance, plot_segment_clearance, min_clearances_both_grids
#     )

#     # --- find the tightest segment for agent 0 ---
#     tight_seg = None
#     tight_hmin = np.inf
#     foh0 = solver.fohs[0]

#     for k in range(K - 1):
#         ts, h_vals = sample_segment_clearance(
#             X_fin[0], U_fin[0], k,
#             foh0, obstacle=(np.array(center), rad)
#         )
#         if h_vals.min() < tight_hmin:
#             tight_hmin = h_vals.min()
#             tight_seg = k
#     print(f"tightest segment = {tight_seg}, clearance = {tight_hmin:.3f} m")
#     ts, h_vals = sample_segment_clearance(
#     X_fin[0], U_fin[0], tight_seg,
#     foh0, obstacle=(np.array(center), rad)
#     )
#     plot_segment_clearance(ts, h_vals,
#                         title=f"Agent 0 – seg {tight_seg} – clearance")
#     from SCvx.utils.analysis import min_clearances_both_grids

#     print("\n--- Knot vs fine-grid clearances ---")
#     for i, (X_i, U_i) in enumerate(zip(X_fin, U_fin)):
#         min_knot, min_fine = min_clearances_both_grids(
#             X_i, U_i,
#             solver.fohs[i],
#             mam.models[i].obstacles,
#             mam.models[i].robot_radius,
#             MARGIN,
#             fine_res=50,
#         )
#         print(f"Agent {i}:  min on knots = {min_knot:.3f} m   |   min on fine grid = {min_fine:.3f} m")
#     # 2½. — Post-solve inter-sample collision check —
#     print("\n=== Post-solve inter-sample collision check ===")
#     collision_flag = False
#     from SCvx.utils.intersample_collision import find_critical_times, make_segment_f

#     foh = solver.fohs[0]  # or loop over all agents if you want
#     for k in range(K - 1):
#         f_seg, _ = make_segment_f(foh, U_fin[0][:, k], U_fin[0][:, k + 1], 1.0)
#         for obs_idx, obs in enumerate(mam.models[0].obstacles):
#             t_stars = find_critical_times(
#                 xk=X_fin[0][:, k],
#                 uk=U_fin[0][:, k],
#                 f=f_seg,
#                 T=np.eye(3),
#                 obstacle=obs,
#                 dt=1.0,
#             )
#             if t_stars:
#                 collision_flag = True
#                 print(f"  ▶ mid-sample collision at segment {k}, obstacle {obs_idx}, t* = {t_stars}")
#     if not collision_flag:
#         print("  ✔ No mid-sample collisions detected.")
#     print("=== End inter-sample check ===\n")

#     # 3. --- Print Analysis ---
#     d_min, d_pair = min_inter_agent_distance(X_fin)
#     print(f"\nMinimum pairwise separation: {d_min:.3f} m")
#     # 3½. --- Print agent–obstacle clearances ---
from SCvx.utils.analysis import min_agent_obstacle_distance

#     obstacles = mam.models[0].obstacles
#     r_robot = mam.models[0].robot_radius
#     d_obs_min, d_obs_mat = min_agent_obstacle_distance(X_fin, obstacles, r_robot)
#     print("\nAgent–obstacle clearances:")
#     for i in range(len(X_fin)):
#         for j, (c, r_j) in enumerate(obstacles):
#             print(f"  Agent {i} → Obstacle {j}: {d_obs_mat[i, j]:.3f} m")
#     print(f"Minimum agent–obstacle clearance: {d_obs_min:.3f} m\n")

#     # Visual size diagnostics
#     first_agent = mam.models[0]
#     radius = first_agent.robot_radius
#     arm_length = radius * 1.5
#     wingspan = 2 * arm_length

#     # 4. --- Visualization ---
#     # Static final trajectories
#     print("Plotting final static trajectories…")
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection="3d")
#     colors = plt.cm.viridis(np.linspace(0, 1, len(X_fin)))
#     for i, X in enumerate(X_fin):
#         ax.plot(X[0], X[1], X[2], color=colors[i], linewidth=2.5, label=f"Agent {i}")
#         ax.scatter(X[0, 0], X[1, 0], X[2, 0], color=colors[i], marker="o", s=60, edgecolors="k")
#         ax.scatter(X[0, -1], X[1, -1], X[2, -1], color=colors[i], marker="x", s=80, linewidths=2)

#     # Plot obstacles
#     if mam.models:
#         _u, _v = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]
#         for center, rad in mam.models[0].obstacles:
#             xs = center[0] + rad * np.cos(_u) * np.sin(_v)
#             ys = center[1] + rad * np.sin(_u) * np.sin(_v)
#             zs = center[2] + rad * np.cos(_v)
#             ax.plot_wireframe(xs, ys, zs, color="saddlebrown", alpha=0.3)

#     ax.set_title("Nash Equilibrium Final Trajectories")
#     ax.set_xlabel("X")
#     ax.set_ylabel("Y")
#     ax.set_zlabel("Z")
#     ax.legend()
#     plt.show()

#     # Convergence history
#     print("Plotting convergence history…")
#     plot_convergence(hist)
#     plt.show()

#     # 5. Full inter–sample collision scan
#     from SCvx.utils.intersample_collision import find_critical_times, make_segment_f

#     print("Running full inter‐sample scan…")
#     for k in tqdm(range(K - 1), desc="Segments"):
#         f_seg, _ = make_segment_f(solver.fohs[0], U0_list[0][:, k], U0_list[0][:, k + 1], 1.0)
#         for obs_idx, obstacle in enumerate(mam.models[0].obstacles):
#             t_stars = find_critical_times(
#                 xk=X0_list[0][:, k],
#                 uk=U0_list[0][:, k],
#                 f=f_seg,
#                 T=np.eye(3),
#                 obstacle=obstacle,
#                 dt=1.0,
#             )

#     # 6. Animations
#     print("Creating animations…")
#     anim1 = animate_trajectories_3d(X_fin, mam.models, interval=50)
#     plt.show()

#     anim2 = animate_multi_agent_quadrotors(X_fin, mam.models, interval=200)
#     plt.show()

#     anim3 = animate_multi_agent_spheres(X_fin, mam.models, interval=50)
#     plt.show()


# if __name__ == "__main__":
#     main()
"""
Runs the 3D Nash-game demo with single-integrator agents, with robust
intersample‐clearance checks and guaranteed non-negative minimum clearances,
and finally a 2D cross‐section plot showing the continuous path vs. the
inflated obstacle and the tangent hyperplane.
"""

import inspect
import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# ── SCvx core ──────────────────────────────────────────────────────────────
from SCvx.config.SI_default_game import AGENT_PARAMS, CLEARANCE, MARGIN_OBS, MARGIN_AGT, K
from SCvx.models.game_si_model import GameSIModel
from SCvx.models.SI_multi_agent_model import SI_MultiAgentModel
from SCvx.optimization.si_nash_solver import SI_NashSolver
from SCvx.utils.analysis import min_inter_agent_distance
from SCvx.utils.IS_initial_guess import initial_guess

# ── Visual helpers ────────────────────────────────────────────────────────
from SCvx.visualization.intersample_plot_utils import plot_segment_clearance
from SCvx.visualization.plot_utils import plot_convergence

# ── Tolerance for numerical clamping ──────────────────────────────────────
EPS = 0  # turn off clamping floor for “raw” report
# project (x,y,z) → (x,z)  (ground-plane)
T_GROUND = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
])

# -------------------------------------------------------------------------
# helper: robust call to f_seg (unchanged)
# -------------------------------------------------------------------------
def call_f_seg(f_seg, t, p_k=None, p_k1=None):
    trials = [
        lambda: f_seg(t),
        lambda: f_seg(None, t),
        lambda: f_seg(t, None),
        lambda: f_seg(None, t, 1.0),
        lambda: f_seg(t, None, 1.0),
    ]
    for try_call in trials:
        try:
            out = try_call()
            if hasattr(out, "__len__") and len(out) == 3:
                return np.asarray(out, float)
        except Exception:
            continue
    if p_k is None or p_k1 is None:
        raise RuntimeError("call_f_seg: all signatures failed.")
    return (1.0 - t) * p_k + t * p_k1

# -------------------------------------------------------------------------
# sample_segment_clearance (with clamping)
# -------------------------------------------------------------------------
def sample_segment_clearance(
    X: np.ndarray, U: np.ndarray, k_seg: int, foh, *,
    obstacle: Tuple[np.ndarray, float],
    r_robot: float, margin: float,
    n_samples: int = 250,
):
    from SCvx.utils.intersample_collision import h_i, make_segment_f

    # unpack
    c_obs, r_obs = obstacle

    # decide whether to project the obstacle center or not
    # if it's length-3, we project; if length-2, assume it's already in 2D
    if c_obs.shape[0] == 3:
        c_obs_2d = T_GROUND @ c_obs
    else:
        c_obs_2d = c_obs

    # total radius (obstacle + robot + margin)
    r_tot = r_obs + r_robot + margin

    # get segment function
    f_seg, _ = make_segment_f(foh, U[:, k_seg], U[:, k_seg + 1], sigma=1.0)
    xk, uk = X[:, k_seg], U[:, k_seg]

    ts = np.linspace(0.0, 1.0, n_samples)
    h_vals = np.array([
        h_i(xk, uk, t, f_seg, T_GROUND, (c_obs_2d, r_tot))
        for t in ts
    ])
    # clamp small negatives to -EPS
    h_vals = np.maximum(h_vals, -EPS)
    return ts, h_vals

# -------------------------------------------------------------------------
# raw_min_clearances_both_grids (unchanged initial‐guess uses CLEARANCE)
# -------------------------------------------------------------------------
def raw_min_clearances_both_grids(
    X: np.ndarray,
    U: np.ndarray,
    foh,
    obstacles: list[tuple[np.ndarray, float]],
    r_robot: float,
    margin: float,
    fine_res: int = 100,
) -> tuple[float, float]:
    from SCvx.utils.intersample_collision import make_segment_f

    K_tot = X.shape[1]
    min_h_knot = np.inf
    min_h_fine = np.inf

    for k in range(K_tot - 1):
        # knot‐only
        for idx in (k, k + 1):
            for c, r in obstacles:
                h_k = np.linalg.norm(X[:, idx] - c) - (r + r_robot + margin)
                min_h_knot = min(min_h_knot, h_k)

        # fine‐grid
        f_seg, _ = make_segment_f(foh, U[:, k], U[:, k + 1], sigma=1.0)
        p_k, p_k1 = X[:, k], X[:, k + 1]
        for t in np.linspace(0.0, 1.0, fine_res):
            p_t = call_f_seg(f_seg, t, p_k, p_k1)
            for c, r in obstacles:
                h_t = np.linalg.norm(p_t - c) - (r + r_robot + margin)
                min_h_fine = min(min_h_fine, h_t)

    return min_h_knot, min_h_fine

# -------------------------------------------------------------------------
# min_clearances_both_grids (unchanged)
# -------------------------------------------------------------------------
def min_clearances_both_grids(
        X, U, foh, obstacles, r_robot, margin, fine_res: int = 100):
    from SCvx.utils.intersample_collision import make_segment_f

    K_tot = X.shape[1]
    min_h_knot = np.inf
    min_h_fine = np.inf

    for k in range(K_tot - 1):
        # knot‐point clearance
        for idx in (k, k + 1):
            for c, r in obstacles:
                h_k = np.linalg.norm(T_GROUND @ X[:, idx] - T_GROUND @ c) - (r + r_robot + margin)
                min_h_knot = min(min_h_knot, h_k)

        # fine‐grid clearance
        f_seg, _ = make_segment_f(foh, U[:, k], U[:, k + 1], sigma=1.0)
        p_k, p_k1 = X[:, k], X[:, k + 1]
        for t in np.linspace(0.0, 1.0, fine_res):
            p_t = call_f_seg(f_seg, t, p_k, p_k1)
            for c, r in obstacles:
                h_t = np.linalg.norm(p_t - c) - (r + r_robot + margin)
                min_h_fine = min(min_h_fine, h_t)

    min_h_knot = max(min_h_knot, 0)
    min_h_fine = max(min_h_fine, 0)
    return min_h_knot, min_h_fine

# -------------------------------------------------------------------------
# build model
# -------------------------------------------------------------------------
def build_multi_agent_model() -> SI_MultiAgentModel:
    mam = SI_MultiAgentModel(AGENT_PARAMS)
    for i, p in enumerate(AGENT_PARAMS):
        mam.models[i] = GameSIModel(
        r_init               = p["r_init"],
        r_final              = p["r_final"],
        robot_radius         = p.get("robot_radius", 0.5),
        obstacles            = p.get("obstacles", []),
        control_weight       = p.get("control_weight", 1.0),
        collision_weight     = p.get("collision_weight", 80.0),
        collision_radius     = p.get("collision_radius", 1.0),
        control_rate_weight  = p.get("control_rate_weight", 5.0),
        curvature_weight     = p.get("curvature_weight", 0.0),
        # if you have other cost-keys, you can pass them here too
    )
    return mam

# -------------------------------------------------------------------------
# main
# -------------------------------------------------------------------------
def main() -> None:
    mam = SI_MultiAgentModel(AGENT_PARAMS)
    for i, p in enumerate(AGENT_PARAMS):
        mam.models[i] = GameSIModel(**p)

    # 1. initial guesses
    X0_list, U0_list = zip(*[
        initial_guess(p["r_init"], p["r_final"], p["obstacles"], CLEARANCE, K)
        for p in AGENT_PARAMS
    ])
    X0_list, U0_list = list(X0_list), list(U0_list)

    # 2. instantiate solver (so solver.fohs exist)
    solver = SI_NashSolver(mam, max_iter=25, tol=1e-3)

    # --- Pre-solve inter-sample clearance (initial guess) ---
    print("=== Pre-solve inter-sample clearance (initial guess) ===")
    print("     Agent |  knot-only (m)  |  fine-grid (m)")
    print("    -------+----------------+---------------")
    for i in range(len(X0_list)):
        knot_raw, fine_raw = raw_min_clearances_both_grids(
            X0_list[i], U0_list[i], solver.fohs[i],
            mam.models[i].obstacles, mam.models[i].robot_radius,
            margin=CLEARANCE, fine_res=200
        )
        print(f"   {i:>3d}    |    {knot_raw:6.3f}     |    {fine_raw:6.3f}")
    print("=== End pre-solve clearance ===\n")

    # 3. solve Nash game
    solver = SI_NashSolver(mam, max_iter=25, tol=1e-3)
    X_fin, U_fin, hist = solver.solve(X0_list, U0_list, sigma_ref=1.0, verbose=False, show_progress=True)

    # --- Post-solve inter-sample clearance (raw values) ---
    print("=== Post-solve inter-sample clearance (solution) ===")
    print("     Agent |  knot-only (m)  |  fine-grid (m)")
    print("    -------+----------------+---------------")
    for i in range(len(X_fin)):
        knot_raw, fine_raw = raw_min_clearances_both_grids(
            X_fin[i], U_fin[i], solver.fohs[i],
            mam.models[i].obstacles, mam.models[i].robot_radius,
            # now use obstacle‐margin
            margin=MARGIN_OBS, fine_res=200
        )
        print(f"   {i:>3d}    |    {knot_raw:6.3f}     |    {fine_raw:6.3f}")
    print("=== End post-solve clearance ===\n")

    # 4. worst-clearance segment for Agent 0 (1D)
    center3, rad = mam.models[0].obstacles[0]
    center2 = T_GROUND @ np.array(center3)

    tight_seg, tight_h = None, np.inf
    for k in range(K - 1):
        ts_tmp, h_tmp = sample_segment_clearance(
            X_fin[0], U_fin[0], k, solver.fohs[0],
            obstacle=(center2, rad),
            r_robot=mam.models[0].robot_radius,
            # now use obstacle‐margin
            margin=MARGIN_OBS
        )
        if h_tmp.min() < tight_h:
            tight_h, tight_seg, ts_best, h_best = h_tmp.min(), k, ts_tmp, h_tmp

    print(f"Agent-0 tightest segment: {tight_seg}   min h = {tight_h:.3f} m")
    plot_segment_clearance(ts_best, h_best, (h_best[0], h_best[-1]),
                           title=f"Agent 0 • segment {tight_seg} clearance")

    # 5. 2D cross-section plots (all agents)
    from SCvx.utils.intersample_collision import make_segment_f, find_critical_times

    for i in range(len(X_fin)):
        c3, r3 = mam.models[i].obstacles[0]
        c2 = T_GROUND @ np.array(c3)
        # use obstacle‐margin for plotting
        r_tot = r3 + mam.models[i].robot_radius + MARGIN_OBS
        seg_roots = []
        for k in range(K-1):
            f_seg, _ = make_segment_f(solver.fohs[i], U_fin[i][:,k], U_fin[i][:,k+1], sigma=1.0)
            roots = find_critical_times(
                xk=X_fin[i][:,k], uk=U_fin[i][:,k],
                f=f_seg, T=T_GROUND,
                obstacle=(c2, r_tot), dt=1.0
            )
            if roots:
                seg_roots.append((k, roots[0]))
        if not seg_roots:
            print(f"Agent {i}: no interior minima.")
            continue
        best_seg, t_star = seg_roots[0]
        f_seg, _ = make_segment_f(solver.fohs[i], U_fin[i][:,best_seg], U_fin[i][:,best_seg+1], sigma=1.0)
        p_k, p_k1 = X_fin[i][:,best_seg], X_fin[i][:,best_seg+1]
        ts = np.linspace(0,1,300)
        pts = np.stack([call_f_seg(f_seg,t,p_k,p_k1) for t in ts])
        xs = pts[:,0] - c3[0]
        zs = pts[:,2] - c3[2]
        xs_obs = np.linspace(xs.min()-0.2, xs.max()+0.2, 400)
        zs_obs = np.sqrt(np.clip(r_tot**2 - xs_obs**2, 0, None))
        p_star = call_f_seg(f_seg, t_star, p_k, p_k1)
        x_star, z_star = p_star[0]-c3[0], p_star[2]-c3[2]
        h_star = z_star - r_tot
        fig, ax = plt.subplots(figsize=(8,5))
        ax.plot(xs_obs, zs_obs, 'k-', lw=2)
        ax.plot(xs, zs, 'b-', lw=2)
        ax.scatter([xs[0], xs[-1]], [zs[0], zs[-1]], c='k', s=50)
        ax.scatter(x_star, z_star, c='r', s=80)
        ax.axhline(z_star, color='r', ls='--')
        ax.set_title(f"Agent {i} seg {best_seg} cross-sec  (h={h_star:.3f}m)")
        ax.set_xlabel("X − obstacle centre")
        ax.set_ylabel("Z − obstacle centre")
        ax.grid(True); ax.margins(0.3)
        plt.show()

    # 6. final summary plots
    d_min, _ = min_inter_agent_distance(X_fin)
    print(f"\nMinimum pairwise separation: {d_min:.3f} m")
    fig = plt.figure(figsize=(9,7)); ax = fig.add_subplot(111, projection='3d')
    cols = plt.cm.viridis(np.linspace(0,1,len(X_fin)))
    for i,X in enumerate(X_fin): ax.plot(*X, color=cols[i], label=f"A{i}")
    c0, r0 = mam.models[0].obstacles[0]
    u,v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    xs = c0[0]+r0*np.cos(u)*np.sin(v)
    ys = c0[1]+r0*np.sin(u)*np.sin(v)
    zs = c0[2]+r0*np.cos(v)
    ax.plot_wireframe(xs,ys,zs, color='sienna', alpha=0.3)
    ax.set_title("Final trajectories"); ax.legend(); plt.show()
    plot_convergence(hist); plt.show()

    # 7. Static “pretty” trajectories + inflated obstacle
    print("Plotting final static trajectories…")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    colors = plt.cm.viridis(np.linspace(0, 1, len(X_fin)))
    for i, X in enumerate(X_fin):
        ax.plot(X[0], X[1], X[2],
                color=colors[i], linewidth=2.5, label=f"Agent {i}")
        ax.scatter(X[0, 0], X[1, 0], X[2, 0],
                   color=colors[i], marker="o", s=60, edgecolors="k")
        ax.scatter(X[0, -1], X[1, -1], X[2, -1],
                   color=colors[i], marker="x", s=80, linewidths=2)
    # plot nominal obstacle and inflated obstacle
    _u, _v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    for center3, rad in mam.models[0].obstacles:
        # nominal
        xs = center3[0] + rad * np.cos(_u) * np.sin(_v)
        ys = center3[1] + rad * np.sin(_u) * np.sin(_v)
        zs = center3[2] + rad * np.cos(_v)
        ax.plot_wireframe(xs, ys, zs, color="saddlebrown", alpha=0.3)
        # inflated using obstacle margin
        r_infl = rad + mam.models[0].robot_radius + MARGIN_OBS
        xs_i = center3[0] + r_infl * np.cos(_u) * np.sin(_v)
        ys_i = center3[1] + r_infl * np.sin(_u) * np.sin(_v)
        zs_i = center3[2] + r_infl * np.cos(_v)
        ax.plot_wireframe(xs_i, ys_i, zs_i,
                          color="tomato", linestyle="--", alpha=0.5,
                          linewidth=1.0, label="inflated obstacle")
    ax.set_title("Nash Equilibrium Final Trajectories")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.legend()
    plt.show()

    # 8. Convergence history
    print("Plotting convergence history…")
    plot_convergence(hist)
    plt.show()

    # 9. Full inter-sample scan (debug)
    from SCvx.utils.intersample_collision import find_critical_times, make_segment_f
    print("Running full inter‐sample scan…")
    for k in tqdm(range(K - 1), desc="Segments"):
        f_seg, _ = make_segment_f(
            solver.fohs[0],
            U0_list[0][:, k],
            U0_list[0][:, k + 1],
            sigma=1.0,
        )
        for obs_idx, (c3, r3) in enumerate(mam.models[0].obstacles):
            c2 = T_GROUND @ np.array(c3)
            # use obstacle‐margin here too
            r_tot = r3 + mam.models[0].robot_radius + MARGIN_OBS
            t_stars = find_critical_times(
                xk=X0_list[0][:, k],
                uk=U0_list[0][:, k],
                f=f_seg,
                T=T_GROUND,
                obstacle=(c2, r_tot),
                dt=1.0,
            )

    # 10. Animations
    print("Creating animations…")
    anim1 = animate_trajectories_3d(X_fin, mam.models, interval=50)
    plt.show()
    anim2 = animate_multi_agent_quadrotors(X_fin, mam.models, interval=200)
    plt.show()
    anim3 = animate_multi_agent_spheres(X_fin, mam.models, interval=50)
    plt.show()


if __name__ == "__main__":
    main()