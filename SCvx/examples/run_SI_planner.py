import matplotlib.pyplot as plt
import numpy as np

from SCvx.models.single_integrator_model import SingleIntegratorModel
from SCvx.optimization.scvx_solver import SCVXSolver
from SCvx.visualization.plot_utils import animate_3d_trajectory, plot_3d_trajectory, plot_convergence


def main():
    # Define only the specified obstacles
    obstacles = [
        ([-5.0, -4.0, -5.0], 2.0),
        ([ 0.0,  0.0,  0.0], 3.0)
    ]

    # Instantiate 3D single-integrator model with these obstacles
    model = SingleIntegratorModel(
        r_init=np.array([-8.0, -8.0, -8.0]),
        r_final=np.array([ 8.0,  8.0,  8.0]),
        v_max=1.0,
        bounds=(-10.0, 10.0),
        robot_radius=1,
        obstacles=obstacles
    )
    solver = SCVXSolver(model)

    # Solve the SCvx problem
    X, U, sigma, logger = solver.solve(verbose=True, initial_sigma=1.0)

    # Extract slack histories
    slacks = [s.value.flatten() for s in model.s_prime]

    # Diagnostics: check actual clearance vs. slack for each obstacle
    centers  = [np.array(c) for c, _ in model.obstacles]
    r_totals = [r + model.robot_radius for _, r in model.obstacles]
    K = X.shape[1]
    for j, (center, r_tot) in enumerate(zip(centers, r_totals)):
        dists = np.linalg.norm(X.T - center, axis=1)
        clearance = dists - r_tot
        print(f"\nObstacle {j} diagnostics (center={center.tolist()}, r_tot={r_tot:.2f}):")
        print(f"  Min actual clearance = {clearance.min():.4f} (negative → penetration)")
        print(f"  Max slack value      = {slacks[j].max():.4f}")
        viol = np.where(clearance < 0)[0]
        if viol.size:
            print(f"  Intersection at steps: {viol.tolist()}")
        else:
            print("  No actual intersection detected.")

    # Summary
    print("\nSummary:")
    print(f"  Final time scale (sigma): {sigma:.3f}")
    print(f"  Final position: {X[:, -1].tolist()}")

    # Plot 3D trajectory and obstacles
    # plot_3d_trajectory(X, model)
    # plt.show()

    # Plot convergence metrics
    plot_convergence(logger.records)
    plt.show()

    # Animate the trajectory in-place
    anim = animate_3d_trajectory(X, model, interval=200)
    plt.show()


if __name__ == "__main__":
    main()
