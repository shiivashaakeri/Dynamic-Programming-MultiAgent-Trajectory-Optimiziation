import matplotlib.pyplot as plt
import numpy as np


def plot_trajectory(X: np.ndarray, model, ax=None):
    """
    Plot the 2D trajectory of the unicycle and obstacles.

    Args:
        X: State trajectory array of shape (n_x, K)
        model: UnicycleModel instance with obstacles and robot_radius
        ax: Matplotlib Axes (optional)
    Returns:
        ax: Matplotlib Axes
    """
    if ax is None:
        fig, ax = plt.subplots()
    # Plot trajectory
    ax.plot(X[0, :], X[1, :], "-o", label="Trajectory")
    # Plot start and goal
    ax.plot(X[0, 0], X[1, 0], "gs", label="Start")
    ax.plot(X[0, -1], X[1, -1], "r*", label="Goal")
    # Plot obstacles
    for p, r in model.obstacles:
        circle = plt.Circle(p, r + model.robot_radius, color="r", fill=False, linestyle="--", linewidth=1.5)
        ax.add_patch(circle)
    ax.set_aspect("equal", "box")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Planned Trajectory")
    ax.legend()
    ax.grid(True)
    return ax


def plot_convergence(history: list, ax=None):
    """
    Plot convergence metrics (defect and slack norms) over iterations.

    Args:
        history: List of dicts with keys 'iter', 'nu_norm', 'slack_norm'
        ax: Matplotlib Axes (optional)
    Returns:
        ax: Matplotlib Axes
    """
    if ax is None:
        fig, ax = plt.subplots()
    iters = [h["iter"] for h in history]
    nu_norms = [h["nu_norm"] for h in history]
    slack_norms = [h["slack_norm"] for h in history]
    ax.plot(iters, nu_norms, "-o", label="||nu||1")
    ax.plot(iters, slack_norms, "-x", label="slack sum")
    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Norm (log scale)")
    ax.set_title("Convergence Metrics")
    ax.legend()
    ax.grid(True)
    return ax


def animate_trajectory(X: np.ndarray, model, interval=200):
    """
    Create an animation of the trajectory.

    Args:
        X: State trajectory array of shape (n_x, K)
        model: UnicycleModel instance
        interval: Delay between frames in milliseconds
    Returns:
        anim: FuncAnimation object
    """
    import matplotlib.pyplot as plt
    from matplotlib import animation

    fig, ax = plt.subplots()
    # Plot obstacles once
    for p, r in model.obstacles:
        circle = plt.Circle(p, r + model.robot_radius, color="r", fill=False, linestyle="--", linewidth=1.5)
        ax.add_patch(circle)
    ax.set_aspect("equal", "box")
    ax.set_xlim(np.min(X[0, :]) - 1, np.max(X[0, :]) + 1)
    ax.set_ylim(np.min(X[1, :]) - 1, np.max(X[1, :]) + 1)
    (line,) = ax.plot([], [], "b-o")

    def init():
        line.set_data([], [])
        return (line,)

    def update(frame):
        line.set_data(X[0, :frame], X[1, :frame])
        return (line,)

    anim = animation.FuncAnimation(fig, update, init_func=init, frames=X.shape[1] + 1, interval=interval, blit=True)
    return anim
