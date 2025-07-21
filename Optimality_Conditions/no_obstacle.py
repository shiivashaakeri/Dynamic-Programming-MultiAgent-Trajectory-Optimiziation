import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

# ==============================================================================
# 1. Define the Time-Scaled System, Cost, and Constraints
# ==============================================================================


def objective_function(vars_flat, n_nodes, w):
    """
    Calculates the total weighted cost from a flat vector of variables.
    The vector contains the control trajectory and the final time tf.
    """
    # --- Unpack variables ---
    tf = vars_flat[-1]
    u = vars_flat[:-1].reshape(n_nodes, 3)

    # Objective 1: Minimize time
    cost_time = tf

    # Objective 2: Minimize control effort (time-scaled)
    # J_effort = integral from 0 to 1 of ||u(τ)||^2 * tf dτ
    tau_nodes = np.linspace(0, 1, n_nodes)
    cost_effort = np.trapz(np.sum(u**2, axis=1), tau_nodes) * tf

    # --- Total Weighted Cost ---
    total_cost = w * cost_time + (1 - w) * cost_effort
    return total_cost


def constraint_function(vars_flat, n_nodes, x0, xf):
    """
    Calculates the final state error based on the scaled dynamics.
    """
    # --- Unpack variables ---
    tf = vars_flat[-1]
    u = vars_flat[:-1].reshape(n_nodes, 3)

    # --- Integrate scaled dynamics: dx/dτ = u(τ) * tf ---
    x_current = x0.copy()
    tau_nodes = np.linspace(0, 1, n_nodes)
    for i in range(n_nodes - 1):
        d_tau = tau_nodes[i + 1] - tau_nodes[i]
        # Dynamics are scaled by tf
        x_current += u[i] * tf * d_tau

    # Return the error between the simulated final state and the target
    return x_current - xf


# ==============================================================================
# 2. Main Optimization Setup
# ==============================================================================
if __name__ == "__main__":
    # --- Define Problem Parameters ---
    x0 = np.array([0.0, 0.0, 0.0])
    xf = np.array([10.0, 5.0, 8.0])
    n_nodes = 50

    # --- Multi-Objective Weight (TRY CHANGING THIS VALUE from 0.0 to 1.0) ---
    # w = 1.0 -> Minimize time only (aggressive, high effort)
    # w = 0.0 -> Minimize effort only (gentle, slow)
    # w = 0.5 -> Balanced approach
    w = 0.5

    # --- Initial Guess ---
    # Guess for control u and final time tf
    tf_guess = 10.0  # A reasonable guess for the final time
    u_guess_flat = np.tile((xf - x0) / tf_guess, (n_nodes, 1)).flatten()
    # Combine into a single vector of decision variables
    vars_initial_guess = np.concatenate([u_guess_flat, [tf_guess]])

    # --- Bounds ---
    # No bounds on control u, but tf must be positive
    bounds = [(None, None)] * (3 * n_nodes) + [(0.01, None)]

    # --- Constraints ---
    constraints = {"type": "eq", "fun": constraint_function, "args": (n_nodes, x0, xf)}

    # --- Run the Optimization ---
    print(f"🚀 Starting optimization with weight w = {w}...")
    result = minimize(
        lambda v: objective_function(v, n_nodes, w),
        vars_initial_guess,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"disp": True, "maxiter": 200, "ftol": 1e-5},
    )

    # --- Plotting the Results ---
    if result.success:
        # Unpack the optimal solution
        optimal_vars = result.x
        tf_optimal = optimal_vars[-1]
        u_optimal = optimal_vars[:-1].reshape(n_nodes, 3)

        print("\n✅ Optimization finished successfully!")
        print(f"   Optimal Final Time (tf): {tf_optimal:.2f} seconds")

        # --- Simulate and Plot ---
        t_plot = np.linspace(0, tf_optimal, 200)  # Use the optimal tf for the time axis
        tau_nodes = np.linspace(0, 1, n_nodes)

        u_plot = np.array([np.interp(t_plot / tf_optimal, tau_nodes, u_optimal[:, i]) for i in range(3)]).T

        x_trajectory = [x0]
        for i in range(len(t_plot) - 1):
            dt = t_plot[i + 1] - t_plot[i]
            x_new = x_trajectory[-1] + u_plot[i] * dt
            x_trajectory.append(x_new)
        x_trajectory = np.array(x_trajectory)

        fig = plt.figure(figsize=(18, 6))

        # Plot 1: State Trajectories
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.plot(t_plot, x_trajectory[:, 0], label="x1")
        ax1.plot(t_plot, x_trajectory[:, 1], label="x2")
        ax1.plot(t_plot, x_trajectory[:, 2], label="x3")
        ax1.set_title("State vs. Time")
        ax1.set_xlabel("Time (s)")
        ax1.legend()

        # Plot 2: Control Inputs
        ax2 = fig.add_subplot(1, 3, 2)
        ax2.plot(t_plot, u_plot[:, 0], label="u1 (vx)")
        ax2.plot(t_plot, u_plot[:, 1], label="u2 (vy)")
        ax2.plot(t_plot, u_plot[:, 2], label="u3 (vz)")
        ax2.set_title("Control vs. Time")
        ax2.set_xlabel("Time (s)")
        ax2.legend()

        # Plot 3: 3D Trajectory
        ax3 = fig.add_subplot(1, 3, 3, projection="3d")
        ax3.plot(x_trajectory[:, 0], x_trajectory[:, 1], x_trajectory[:, 2], "g-", label="Trajectory")
        ax3.scatter(x0[0], x0[1], x0[2], color="b", s=50, label="Start")
        ax3.scatter(xf[0], xf[1], xf[2], color="r", s=50, label="End")
        ax3.set_title("3D Position Trajectory")
        ax3.set_xlabel("x1")
        ax3.set_ylabel("x2")
        ax3.set_zlabel("x3")
        ax3.legend()

        plt.tight_layout()
        plt.show()
    else:
        print(f"\n❌ Optimization failed with message: {result.message}")
