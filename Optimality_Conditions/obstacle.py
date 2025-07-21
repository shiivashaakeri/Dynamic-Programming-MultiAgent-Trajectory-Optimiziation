import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

# ==============================================================================
# 1. Define the Time-Scaled System, Cost, and Constraints
# ==============================================================================


def objective_function(vars_flat, n_nodes, w):
    """Calculates the total weighted cost (time + effort)."""
    tf = vars_flat[-1]
    u = vars_flat[:-1].reshape(n_nodes, 3)
    tau_nodes = np.linspace(0, 1, n_nodes)
    cost_effort = np.trapz(np.sum(u**2, axis=1), tau_nodes) * tf
    total_cost = w * tf + (1 - w) * cost_effort
    return total_cost


def final_state_constraint(vars_flat, n_nodes, x0, xf):
    """Equality constraint: Ensures the trajectory reaches the final state xf."""
    tf = vars_flat[-1]
    u = vars_flat[:-1].reshape(n_nodes, 3)
    x_current = x0.copy()
    tau_nodes = np.linspace(0, 1, n_nodes)
    for i in range(n_nodes - 1):
        d_tau = tau_nodes[i + 1] - tau_nodes[i]
        x_current += u[i] * tf * d_tau
    return x_current - xf


def obstacle_constraint(vars_flat, n_nodes, x0, x_obs, r_obs):
    """
    Inequality constraint: Ensures the trajectory stays outside the obstacle.
    Returns a list of values that must all be >= 0.
    """
    tf = vars_flat[-1]
    u = vars_flat[:-1].reshape(n_nodes, 3)

    # Simulate the path and check distance at each node
    x_path = [x0.copy()]
    tau_nodes = np.linspace(0, 1, n_nodes)
    for i in range(n_nodes - 1):
        d_tau = tau_nodes[i + 1] - tau_nodes[i]
        x_next = x_path[-1] + u[i] * tf * d_tau
        x_path.append(x_next)

    # Calculate squared distance from path to obstacle center
    distances_sq = [np.sum((pos - x_obs) ** 2) for pos in x_path]

    # The constraint is dist^2 - radius^2 >= 0
    return np.array(distances_sq) - r_obs**2


# ==============================================================================
# 2. Main Optimization Setup
# ==============================================================================
if __name__ == "__main__":
    # --- Define Problem Parameters ---
    x0 = np.array([0.0, 0.0, 0.0])
    xf = np.array([10.0, 5.0, 8.0])
    n_nodes = 50
    w = 0.5  # Balanced weight for time vs. effort

    # --- Define and Place a Random Obstacle ---
    r_obs = 2  # Radius of the obstacle
    # Place obstacle near the midpoint, with some randomness
    midpoint = (x0 + xf) / 2
    x_obs = midpoint + np.random.uniform(-1.5, 1.5, size=3)
    print(f"🔵 Obstacle generated at {np.round(x_obs, 2)} with radius {r_obs}")

    # --- Initial Guess ---
    tf_guess = 15.0  # Guess a longer time due to obstacle
    u_guess_flat = np.tile((xf - x0) / tf_guess, (n_nodes, 1)).flatten()
    vars_initial_guess = np.concatenate([u_guess_flat, [tf_guess]])

    # --- Bounds ---
    bounds = [(None, None)] * (3 * n_nodes) + [(0.01, None)]

    # --- ALL Constraints (Equality and Inequality) ---
    constraints = [
        {"type": "eq", "fun": final_state_constraint, "args": (n_nodes, x0, xf)},
        {"type": "ineq", "fun": obstacle_constraint, "args": (n_nodes, x0, x_obs, r_obs)},
    ]

    # --- Run the Optimization ---
    print("🚀 Starting optimization with obstacle avoidance...")
    result = minimize(
        lambda v: objective_function(v, n_nodes, w),
        vars_initial_guess,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"disp": True, "maxiter": 300, "ftol": 1e-5},
    )

    # --- Plotting the Results ---
    if result.success:
        optimal_vars = result.x
        tf_optimal = optimal_vars[-1]
        u_optimal = optimal_vars[:-1].reshape(n_nodes, 3)

        print("\n✅ Optimization finished successfully!")
        print(f"   Optimal Final Time (tf): {tf_optimal:.2f} seconds")

        # --- Simulate and Plot ---
        t_plot = np.linspace(0, tf_optimal, 200)
        tau_nodes = np.linspace(0, 1, n_nodes)
        u_plot = np.array([np.interp(t_plot / tf_optimal, tau_nodes, u_optimal[:, i]) for i in range(3)]).T

        x_trajectory = [x0]
        for i in range(len(t_plot) - 1):
            dt = t_plot[i + 1] - t_plot[i]
            x_new = x_trajectory[-1] + u_plot[i] * dt
            x_trajectory.append(x_new)
        x_trajectory = np.array(x_trajectory)

        fig = plt.figure(figsize=(18, 6))

        # Plot 1 & 2: State and Control vs. Time (omitted for brevity, same as before)
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.plot(t_plot, x_trajectory[:, 0], label="x1")
        ax1.plot(t_plot, x_trajectory[:, 1], label="x2")
        ax1.plot(t_plot, x_trajectory[:, 2], label="x3")
        ax1.set_title("State vs. Time")
        ax1.legend()
        ax2 = fig.add_subplot(1, 3, 2)
        ax2.plot(t_plot, u_plot[:, 0], label="u1")
        ax2.plot(t_plot, u_plot[:, 1], label="u2")
        ax2.plot(t_plot, u_plot[:, 2], label="u3")
        ax2.set_title("Control vs. Time")
        ax2.legend()

        # Plot 3: 3D Trajectory with Obstacle
        ax3 = fig.add_subplot(1, 3, 3, projection="3d")
        ax3.plot(x_trajectory[:, 0], x_trajectory[:, 1], x_trajectory[:, 2], "g-", lw=2, label="Trajectory")
        ax3.scatter(x0[0], x0[1], x0[2], color="b", s=50, label="Start")
        ax3.scatter(xf[0], xf[1], xf[2], color="r", s=50, label="End")

        # Draw the obstacle sphere
        u, v = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]
        sphere_x = x_obs[0] + r_obs * np.cos(u) * np.sin(v)
        sphere_y = x_obs[1] + r_obs * np.sin(u) * np.sin(v)
        sphere_z = x_obs[2] + r_obs * np.cos(v)
        ax3.plot_wireframe(sphere_x, sphere_y, sphere_z, color="grey", alpha=0.6)

        ax3.set_title("3D Position Trajectory")
        ax3.set_xlabel("x1")
        ax3.set_ylabel("x2")
        ax3.set_zlabel("x3")
        ax3.legend()

        plt.tight_layout()
        plt.show()
    else:
        print(f"\n❌ Optimization failed with message: {result.message}")
