"""Example: Non-cooperative Nash equilibrium with the default game scenario.

Uses the scenario defined in ``SCvx/config/game_scenarios/default_game.py``
so you can tweak agent start/goal, obstacle list, weights, etc. in one spot.
"""

import matplotlib.pyplot as plt
import numpy as np

from SCvx.config.default_game import (
    AGENT_PARAMS,
    CLEARANCE,
    K,
)
from SCvx.models.game_model import GameUnicycleModel
from SCvx.models.multi_agent_model import MultiAgentModel
from SCvx.optimization.nash_solver import NashSolver
from SCvx.utils.analysis import min_inter_agent_distance
from SCvx.utils.initial_guess import initial_guess
from SCvx.visualization.multi_agent_plot_utils import animate_multi_agents
from SCvx.visualization.plot_utils import plot_trajectory


def build_multi_agent_model():
    """Instantiate MultiAgentModel and wrap each agent in GameUnicycleModel."""

    # First create a container with bare params (MultiAgentModel only needs r_init/r_final/obstacles)
    mam = MultiAgentModel(AGENT_PARAMS)

    # Replace each inner model with GameUnicycleModel that includes cost weights
    for idx, p in enumerate(AGENT_PARAMS):
        mam.models[idx] = GameUnicycleModel(
            r_init=p["r_init"],
            r_final=p["r_final"],
            obstacles=p["obstacles"],
            control_weight=p.get("control_weight", 1.0),
            collision_weight=p.get("collision_weight", 10.0),
            collision_radius=p.get("collision_radius", 0.3),
        )
    return mam


def main():
    mam = build_multi_agent_model()

    # ----- Warm-start trajectories -------------------------------------------------
    X0_list, U0_list = [], []
    for p in AGENT_PARAMS:
        X0, U0 = initial_guess(
            p["r_init"],
            p["r_final"],
            p["obstacles"],
            CLEARANCE,
            K,
        )
        X0_list.append(X0)
        U0_list.append(U0)

    # ----- Solve Nash equilibrium --------------------------------------------------
    solver = NashSolver(mam, max_iter=25, tol=1e-3)
    X_fin, U_fin, hist = solver.solve(X0_list, U0_list, sigma_ref=1.0, verbose=True)
    d_min_global, d_pair = min_inter_agent_distance(X_fin)
    print(f"\nMinimum pair-wise separation: {d_min_global:.3f} m")
    print("Pair-wise minima (m):\n", np.round(d_pair, 3))
    # ----- Plot trajectories -------------------------------------------------------
    fig, ax = plt.subplots()
    for i, X in enumerate(X_fin):
        plot_trajectory(X, mam.models[i], ax=ax)
    ax.set_title("Nash Equilibrium Trajectories (Default Scenario)")
    plt.show()

    # ----- Convergence plot --------------------------------------------------------
    fig2, ax2 = plt.subplots()
    ax2.semilogy(hist, marker="o")
    ax2.set_xlabel("Outer iteration")
    ax2.set_ylabel("Max ΔX (F-norm)")
    ax2.set_title("Nash Solver Convergence")
    plt.show()

    # ----- Animation ---------------------------------------------------------------
    # ----- Animation -----------------------------------------------------
    anim = animate_multi_agents(X_fin, mam.models, interval=200)  # noqa: F841
    plt.show()          # keep this line


if __name__ == "__main__":
    main()
