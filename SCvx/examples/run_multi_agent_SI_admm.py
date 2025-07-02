import os

import matplotlib.pyplot as plt

from SCvx.config.SI_default_scenario import AGENT_PARAMS, CLEARANCE, D_MIN, K
from SCvx.models.SI_multi_agent_model import SI_MultiAgentModel
from SCvx.optimization.si_admm_coordinator import SI_ADMMCoordinator
from SCvx.utils.IS_initial_guess import initial_guess
from SCvx.visualization.multi_agent_plot_utils import (
    animate_trajectories_3d,
    plot_admm_convergence,
)
from SCvx.visualization.plot_utils import plot_multi_agent_trajectories_3d


def main():
    # Load scenario configuration
    agent_params = AGENT_PARAMS
    d_min = D_MIN
    clearance = CLEARANCE

    # Instantiate the single-integrator multi-agent model
    mam = SI_MultiAgentModel(agent_params, d_min=d_min)

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

    # Coordinate trajectories via ADMM (SI version)
    coord = SI_ADMMCoordinator(mam, rho_admm=1.0, max_iter=10)
    X_list, U_list, sigma_out, pr_hist, du_hist = coord.solve(X_refs, U_refs, sigma_ref)

    # Plot final trajectories over static obstacles
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    plot_multi_agent_trajectories_3d(X_list, mam)
    ax.set_title("Multi-Agent SI SCvx via ADMM")
    plt.show()

    # Plot ADMM convergence residuals
    fig2, ax2 = plt.subplots()
    plot_admm_convergence(pr_hist, du_hist, ax=ax2)
    plt.show()

    # Animate all agents together with obstacles (no saving)
    # OLD

    anim = animate_trajectories_3d(X_list, mam.models, interval=200)
    plt.show()


if __name__ == "__main__":
    main()
