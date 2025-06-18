import os

import matplotlib.pyplot as plt

from SCvx.config.default_scenario import AGENT_PARAMS, CLEARANCE, D_MIN, K
from SCvx.models.multi_agent_model import MultiAgentModel
from SCvx.optimization.admm_coordinator import ADMMCoordinator
from SCvx.utils.initial_guess import initial_guess
from SCvx.visualization.multi_agent_plot_utils import animate_multi_agents, plot_admm_convergence
from SCvx.visualization.plot_utils import plot_trajectory


def main():
    # Load scenario configuration
    agent_params = AGENT_PARAMS
    d_min = D_MIN
    clearance = CLEARANCE

    # Instantiate the multi-agent model
    mam = MultiAgentModel(agent_params, d_min=d_min)

    # Generate warm-start trajectories, one per agent
    X_refs = []
    U_refs = []
    sigma_ref = 1.0

    # Build an initial guess that avoids *all* obstacles per agent
    for params in agent_params:
        start = params["r_init"]
        goal = params["r_final"]
        obs_list = params.get("obstacles", [])
        X0, U0 = initial_guess(start, goal, obs_list, clearance, K)
        X_refs.append(X0)
        U_refs.append(U0)

    # Coordinate trajectories via ADMM
    coord = ADMMCoordinator(mam, rho_admm=1.0, max_iter=10)
    X_list, U_list, sigma_out, pr_hist, du_hist = coord.solve(X_refs, U_refs, sigma_ref)

    # Plot final trajectories over static obstacles
    fig, ax = plt.subplots()
    for i, X_sol in enumerate(X_list):
        plot_trajectory(X_sol, mam.models[i], ax=ax)
    ax.set_title("Multi-Agent SCvx via ADMM")
    plt.show()

    # Plot ADMM convergence residuals
    fig2, ax2 = plt.subplots()
    plot_admm_convergence(pr_hist, du_hist, ax=ax2)
    plt.show()
    save_path = os.path.join("SCvx", "docs", "img", "multi_trajectory_animation.gif")
    # Animate all agents together with obstacles
    anim = animate_multi_agents(X_list, mam.models, interval=200)
    anim.save(save_path, writer="pillow", fps=5)  # Save as GIF using Pillow writer
    plt.show()


if __name__ == "__main__":
    main()
