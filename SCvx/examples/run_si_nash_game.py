#!/usr/bin/env python3
"""
Runs the 3D Nash‐game demo with single‐integrator agents.
This script sets up a multi‐agent scenario, solves for the Nash equilibrium
trajectories, and visualizes the results.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# --- Core SCvx Imports ---
from SCvx.config.SI_default_game import AGENT_PARAMS, CLEARANCE, K
from SCvx.models.game_si_model import GameSIModel
from SCvx.models.SI_multi_agent_model import SI_MultiAgentModel
from SCvx.optimization.si_nash_solver import SI_NashSolver
from SCvx.utils.analysis import min_inter_agent_distance
from SCvx.utils.IS_initial_guess import initial_guess

# --- Visualization Imports ---
from SCvx.visualization.multi_agent_plot_utils import (
    animate_multi_agent_quadrotors,
    animate_multi_agent_spheres,
    animate_trajectories_3d,
)
from SCvx.visualization.plot_utils import plot_convergence


def build_multi_agent_model() -> SI_MultiAgentModel:
    """Initializes the multi‐agent model and wraps each agent in a GameSIModel."""
    mam = SI_MultiAgentModel(AGENT_PARAMS)
    for idx, p in enumerate(AGENT_PARAMS):
        mam.models[idx] = GameSIModel(
            r_init=p["r_init"],
            r_final=p["r_final"],
            obstacles=p["obstacles"],
            robot_radius=p.get("robot_radius", 0.5),
            control_weight=p.get("control_weight", 1.0),
            collision_weight=p.get("collision_weight", 80.0),
            collision_radius=p.get("collision_radius", 0.5),
            control_rate_weight=p.get("control_rate_weight", 5.0),
            curvature_weight=p.get("curvature_weight", 0.0),
        )
    return mam


def main() -> None:
    # 1. --- Model Setup and Initial Guess ---
    mam = build_multi_agent_model()

    X0_list, U0_list = [], []
    for p in tqdm(AGENT_PARAMS, desc="Initial guesses"):
        X0, U0 = initial_guess(p["r_init"], p["r_final"], p["obstacles"], CLEARANCE, K)
        X0_list.append(X0)
        U0_list.append(U0)

    # 2. --- Solve for Nash Equilibrium ---
    solver = SI_NashSolver(mam, max_iter=25, tol=1e-3)
    X_fin, U_fin, hist = solver.solve(
        X0_list,
        U0_list,
        sigma_ref=1.0,
        verbose=False,
        show_progress=True)
    # 2½. — Post-solve inter-sample collision check —
    print("\n=== Post-solve inter-sample collision check ===")
    collision_flag = False
    from SCvx.utils.intersample_collision import find_critical_times, make_segment_f

    foh = solver.fohs[0]  # or loop over all agents if you want
    for k in range(K-1):
        f_seg, _ = make_segment_f(foh, U_fin[0][:,k], U_fin[0][:,k+1], 1.0)
        for obs_idx, obs in enumerate(mam.models[0].obstacles):
            t_stars = find_critical_times(
                xk       = X_fin[0][:,k],
                uk       = U_fin[0][:,k],
                f        = f_seg,
                T        = np.eye(3),
                obstacle = obs,
                dt       = 1.0,
            )
            if t_stars:
                collision_flag = True
                print(f"  ▶ mid-sample collision at segment {k}, obstacle {obs_idx}, t* = {t_stars}")
    if not collision_flag:
        print("  ✔ No mid-sample collisions detected.")
    print("=== End inter-sample check ===\n")
    # 3. --- Print Analysis ---
    d_min, d_pair = min_inter_agent_distance(X_fin)
    print(f"\nMinimum pairwise separation: {d_min:.3f} m")

    # Visual size diagnostics
    first_agent = mam.models[0]
    radius = first_agent.robot_radius
    arm_length = radius * 1.5
    wingspan = 2 * arm_length

    # 4. --- Visualization ---
    # Static final trajectories
    print("Plotting final static trajectories…")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    colors = plt.cm.viridis(np.linspace(0, 1, len(X_fin)))
    for i, X in enumerate(X_fin):
        ax.plot(X[0], X[1], X[2], color=colors[i], linewidth=2.5, label=f"Agent {i}")
        ax.scatter(X[0, 0], X[1, 0], X[2, 0], color=colors[i], marker='o', s=60, edgecolors='k')
        ax.scatter(X[0, -1], X[1, -1], X[2, -1], color=colors[i], marker='x', s=80, linewidths=2)

    # Plot obstacles
    if mam.models:
        _u, _v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        for center, rad in mam.models[0].obstacles:
            xs = center[0] + rad * np.cos(_u) * np.sin(_v)
            ys = center[1] + rad * np.sin(_u) * np.sin(_v)
            zs = center[2] + rad * np.cos(_v)
            ax.plot_wireframe(xs, ys, zs, color='saddlebrown', alpha=0.3)

    ax.set_title("Nash Equilibrium Final Trajectories")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.legend()
    plt.show()

    # Convergence history
    print("Plotting convergence history…")
    plot_convergence(hist)
    plt.show()

    # 5. Full inter–sample collision scan
    from SCvx.utils.intersample_collision import find_critical_times, make_segment_f

    print("Running full inter‐sample scan…")
    for k in tqdm(range(K - 1), desc="Segments"):
        f_seg, _ = make_segment_f(solver.fohs[0], U0_list[0][:, k], U0_list[0][:, k+1], 1.0)
        for obs_idx, obstacle in enumerate(mam.models[0].obstacles):
            t_stars = find_critical_times(
                xk=X0_list[0][:, k],
                uk=U0_list[0][:, k],
                f=f_seg,
                T=np.eye(3),
                obstacle=obstacle,
                dt=1.0,
            )

    # 6. Animations
    print("Creating animations…")
    anim1 = animate_trajectories_3d(X_fin, mam.models, interval=50)
    plt.show()

    anim2 = animate_multi_agent_quadrotors(X_fin, mam.models, interval=200)
    plt.show()

    anim3 = animate_multi_agent_spheres(X_fin, mam.models, interval=50)
    plt.show()


if __name__ == "__main__":
    main()