"""
Runs the 3D Nash-game demo with single-integrator agents.
This script sets up a multi-agent scenario, solves for the Nash equilibrium
trajectories, and visualizes the results.
"""
import os

import matplotlib.pyplot as plt
import numpy as np

# --- Core SCvx Imports ---
from SCvx.config.SI_default_game import AGENT_PARAMS, CLEARANCE, K
from SCvx.models.game_si_model import GameSIModel
from SCvx.models.SI_multi_agent_model import SI_MultiAgentModel
from SCvx.optimization.si_nash_solver import SI_NashSolver
from SCvx.utils.analysis import min_inter_agent_distance
from SCvx.utils.IS_initial_guess import initial_guess

# --- Visualization Imports ---
# Import only the necessary, clean visualization functions
from SCvx.visualization.multi_agent_plot_utils import (
    animate_multi_agent_quadrotors,
    animate_multi_agent_spheres,
    animate_trajectories_3d,
)
from SCvx.visualization.plot_utils import plot_convergence


def build_multi_agent_model() -> SI_MultiAgentModel:
    """Initializes the multi-agent model and wraps each agent in a GameSIModel."""
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


def main() -> None:  # noqa: PLR0915
    # 1. --- Model Setup and Initial Guess ---
    print("Building multi-agent model and generating initial guess...")
    mam = build_multi_agent_model()

    X0_list, U0_list = [], []
    for p in AGENT_PARAMS:
        X0, U0 = initial_guess(p["r_init"], p["r_final"], p["obstacles"], CLEARANCE, K)
        X0_list.append(X0)
        U0_list.append(U0)

    # 2. --- Solve for Nash Equilibrium ---
    print("Solving for Nash equilibrium...")
    solver = SI_NashSolver(mam, max_iter=25, tol=1e-3)
    X_fin, U_fin, hist = solver.solve(X0_list, U0_list, sigma_ref=1.0, verbose=True)

    # 3. --- Print Analysis ---
    d_min, d_pair = min_inter_agent_distance(X_fin)
    print(f"\nMinimum pairwise separation: {d_min:.3f} m")
    print("Pairwise minima (m):\n", np.round(d_pair, 3))
    # --- NEW: Calculate and print the actual visual size ---
    # Get the model for the first agent to check its properties
    first_agent_model = mam.models[0]
    radius_used = first_agent_model.robot_radius
    arm_length = radius_used * 1.5  # From the build_local_quad_3d function
    total_wingspan = 2 * arm_length

    print("\n--- Visual Size Diagnostics ---")
    print(f"The radius used to draw the quadrotor is: {radius_used:.2f} m")
    print(f"This gives each arm a length of: {arm_length:.2f} m")
    print(f"The total wingspan of each quadrotor is: {total_wingspan:.2f} m")
    print("-----------------------------------")
    # 4. --- Visualization ---

    # Static Plot of Final Trajectories
    print("Plotting final static trajectories...")
    fig_static = plt.figure(figsize=(10, 8))
    ax_static = fig_static.add_subplot(111, projection="3d")
    agent_colors = plt.cm.viridis(np.linspace(0, 1, len(X_fin)))

    for i, x in enumerate(X_fin):
        ax_static.plot(x[0], x[1], x[2], color=agent_colors[i], linewidth=2.5, label=f"Agent {i}")
        ax_static.scatter(x[0,0], x[1,0], x[2,0], color=agent_colors[i], marker='o', s=60, edgecolors='k')
        ax_static.scatter(x[0,-1], x[1,-1], x[2,-1], color=agent_colors[i], marker='x', s=80, linewidths=2)

    # Plot obstacles
    if mam.models:
        _u, _v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        for c, r in mam.models[0].obstacles:
            xs = c[0] + r*np.cos(_u)*np.sin(_v)
            ys = c[1] + r*np.sin(_u)*np.sin(_v)
            zs = c[2] + r*np.cos(_v)
            ax_static.plot_wireframe(xs, ys, zs, color="saddlebrown", alpha=0.3)

    ax_static.set_title("Nash Equilibrium Final Trajectories")
    ax_static.set_xlabel("X")
    ax_static.set_ylabel("Y")
    ax_static.set_zlabel("Z")
    ax_static.legend()
    plt.show()

    # Convergence History Plot
    plot_convergence(hist)
    plt.show()

    # Final Animation
    print("Creating multi-agent animation...")
    anim = animate_trajectories_3d(X_fin, mam.models, interval=50)
    plt.show()

    save_path = os.path.join("SCvx", "docs", "img", "3d_game_quad_animation.gif")
    anim2 = animate_multi_agent_quadrotors(X_fin, mam.models, interval=200)
    anim2.save(save_path, writer="pillow", fps=5)
    plt.show()

    anim = animate_multi_agent_spheres(X_fin, mam.models, interval=50)  # noqa: F841
    plt.show()


if __name__ == "__main__":
    main()
