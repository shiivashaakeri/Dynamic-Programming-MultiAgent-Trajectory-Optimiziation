import os

import matplotlib.pyplot as plt

from SCvx.models.unicycle_model import UnicycleModel
from SCvx.optimization.scvx_solver import SCVXSolver
from SCvx.visualization.plot_utils import animate_trajectory, plot_convergence, plot_trajectory


def main():
    # Instantiate model and solver
    model = UnicycleModel()
    solver = SCVXSolver(model)

    # Solve the SCvx problem (now returns a Logger, not a raw history list)
    X, U, sigma, logger = solver.solve(verbose=True, initial_sigma=1.0)

    # Print summary
    print("Final time (sigma):", sigma)
    print("Final position:", X[:2, -1])

    # Plot trajectory
    ax1 = plot_trajectory(X, model)  # noqa: F841
    plt.show()

    # Plot convergence metrics — use logger.records here
    ax2 = plot_convergence(logger.records)  # noqa: F841
    plt.show()

    # Define the path to save the animation
    save_path = os.path.join("SCvx", "docs", "img", "trajectory_animation.gif")
    # Optionally animate trajectory
    anim = animate_trajectory(X, model, interval=200)
    anim.save(save_path, writer="pillow", fps=5)  # Save as GIF using Pillow writer
    plt.show()


if __name__ == "__main__":
    main()
