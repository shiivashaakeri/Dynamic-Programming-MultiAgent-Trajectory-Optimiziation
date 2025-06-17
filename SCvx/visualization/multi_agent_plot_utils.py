import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle


def plot_admm_convergence(primal_hist, dual_hist, ax=None):
    """
    Plot ADMM primal and dual residuals over iterations.

    Args:
        primal_hist (list of float): primal residuals per ADMM round
        dual_hist (list of float): dual residuals per ADMM round
        ax (matplotlib Axes, optional): axes to plot on. If None, creates a new figure.

    Returns:
        matplotlib Axes: the axes containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots()
    iters = np.arange(1, len(primal_hist) + 1)
    ax.plot(iters, primal_hist, marker='o', label='Primal residual')
    ax.plot(iters, dual_hist,   marker='x', label='Dual   residual')
    ax.set_xlabel('ADMM iteration')
    ax.set_ylabel('Residual')
    ax.set_title('ADMM Convergence')
    ax.grid(True)
    ax.legend()
    return ax


def animate_multi_agents(X_list, models, interval=200):
    """
    Animate multiple agent trajectories on a 2D plot, including static obstacles.

    Args:
        X_list (list of np.ndarray): list of state trajectories, each shape (3, K)
        models (list): list of agent models (for obstacles)
        interval (int): delay between frames in milliseconds

    Returns:
        matplotlib.animation.FuncAnimation: the animation object
    """
    num_agents = len(X_list)
    K = X_list[0].shape[1]

    # Determine plot limits across all agents and obstacles
    all_x = np.hstack([X[0, :] for X in X_list])
    all_y = np.hstack([X[1, :] for X in X_list])
    # Include obstacle centers/radii
    obstacles = getattr(models[0], 'obstacles', [])
    for center, r in obstacles:
        all_x = np.hstack([all_x, center[0] + np.array([-r, r])])
        all_y = np.hstack([all_y, center[1] + np.array([-r, r])])

    xmin, xmax = np.min(all_x), np.max(all_x)
    ymin, ymax = np.min(all_y), np.max(all_y)

    fig, ax = plt.subplots()
    ax.set_xlim(xmin - 0.1*(xmax-xmin), xmax + 0.1*(xmax-xmin))
    ax.set_ylim(ymin - 0.1*(ymax-ymin), ymax + 0.1*(ymax-ymin))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Multi-Agent Trajectory Animation')

        # Draw static obstacles (inflated by robot radius)
    for center, r in obstacles:
        total_r = r + models[0].robot_radius
        circ = Circle(center, total_r, facecolor='none', edgecolor='gray', linestyle='--', lw=2)
        ax.add_patch(circ)

    # Create plot objects for each agent
    lines = []
    for i in range(num_agents):
        line, = ax.plot([], [], 'o-', label=f'agent {i}')
        lines.append(line)
    ax.legend()

    def init():
        for line in lines:
            line.set_data([], [])
        return lines

    def update(frame):
        for line, X in zip(lines, X_list):
            x = X[0, :frame+1]
            y = X[1, :frame+1]
            line.set_data(x, y)
        return lines

    anim = FuncAnimation(fig, update, frames=range(K), init_func=init,
                         blit=True, interval=interval)
    return anim
