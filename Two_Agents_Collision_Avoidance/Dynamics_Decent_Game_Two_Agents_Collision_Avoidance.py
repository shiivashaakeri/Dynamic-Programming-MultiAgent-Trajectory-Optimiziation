import random

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.patches import Circle
from scipy.optimize import minimize

# --- Global Constants and Definitions ---

# Environment
MAX_X = 20
MAX_Y = 20
TIMESTEP = 0.4  # Duration of one simulation step in seconds

# Agent Dynamics
THRUST_POWER = 3.0
MAX_SPEED = 5.0
AGENT_RADIUS = 0.5

# RRT* (Global Planner) Parameters
RRT_ITERATIONS = 2500
RRT_GOAL_SAMPLE_RATE = 0.15
RRT_NEIGHBOR_RADIUS = 4.0

# Optimal Control (Local Planner) Parameters
OPTIMAL_CONTROL_ITERATIONS = 50
# Weights for the cost function
W_POS = 1.0  # Weight for staying close to the reference path
W_VEL = 0.1  # Weight for minimizing velocity (encourages stopping)
W_CONTROL = 0.01  # Weight for minimizing control effort (fuel)
W_FINAL_POS = 100.0  # Strong weight for reaching the final goal position
W_FINAL_VEL = 100.0  # Strong weight for stopping at the final goal

# Goal Checking
GOAL_TOLERANCE_POS = 1.0

# Static Obstacle Definition
OBSTACLE_CENTER = (10, 8)
OBSTACLE_RADIUS = 2.0

# Agent definitions
AGENTS = {
    0: {"start": (2.0, 2.0, 0.0, 0.0), "goal": (18.0, 18.0)},
    1: {"start": (18.0, 2.0, 0.0, 0.0), "goal": (2.0, 18.0)},
}


# --- Physics and Collision Checking ---


def simulate_dynamics(state, action, dt=TIMESTEP):
    """Calculates the next state given the current state and an action."""
    x, y, vx, vy = state
    ax, ay = action
    vx_new = vx + ax * dt
    vy_new = vy + ay * dt
    speed = np.sqrt(vx_new**2 + vy_new**2)
    if speed > MAX_SPEED:
        vx_new = (vx_new / speed) * MAX_SPEED
        vy_new = (vy_new / speed) * MAX_SPEED
    x_new = x + vx_new * dt
    y_new = y + vy_new * dt
    return np.array([x_new, y_new, vx_new, vy_new])


def is_path_segment_in_obstacle(p1, p2):
    """Checks if the line segment p1-p2 intersects the circular obstacle."""
    p1 = np.array(p1[:2])
    p2 = np.array(p2[:2])
    center = np.array(OBSTACLE_CENTER)
    # Vector from p1 to p2
    d = p2 - p1
    # Vector from circle center to p1
    f = p1 - center
    a = d.dot(d)
    b = 2 * f.dot(d)
    c = f.dot(f) - OBSTACLE_RADIUS**2
    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        return False
    else:
        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (2 * a)
        t2 = (-b + discriminant) / (2 * a)
        if (t1 >= 0 and t1 <= 1) or (t2 >= 0 and t2 <= 1):
            return True
    return False


# --- Stage 1: RRT* for Global Pathfinding ---


class RRTNode:
    def __init__(self, state, parent=None):
        self.state = np.array(state)
        self.parent = parent


def rrt_star_search(start_state, goal_pos):
    """Finds a rough, collision-free sequence of waypoints using RRT*."""
    start_node = RRTNode(start_state)
    nodes = [start_node]
    for _ in range(RRT_ITERATIONS):
        if random.random() < RRT_GOAL_SAMPLE_RATE:
            rand_pos = goal_pos
        else:
            rand_pos = (random.uniform(0, MAX_X), random.uniform(0, MAX_Y))

        nearest_node = min(nodes, key=lambda n: np.linalg.norm(n.state[:2] - rand_pos))

        # Simple steering: move one step towards the random point
        direction = rand_pos - nearest_node.state[:2]
        direction = direction / (np.linalg.norm(direction) + 1e-6)
        action = direction * THRUST_POWER

        new_state = simulate_dynamics(nearest_node.state, action)

        if not (0 <= new_state[0] < MAX_X and 0 <= new_state[1] < MAX_Y):
            continue
        if is_path_segment_in_obstacle(nearest_node.state, new_state):
            continue
        # NOTE: Inter-agent collision checking is simplified here for speed

        nodes.append(RRTNode(new_state, parent=nearest_node))

    # Find path that gets closest to goal
    goal_node = min(nodes, key=lambda n: np.linalg.norm(n.state[:2] - goal_pos))
    path = []
    current = goal_node
    while current is not None:
        path.append(current.state)
        current = current.parent
    return path[::-1]


# --- Stage 2: Optimal Control for Trajectory Smoothing ---


def cost_function(controls, initial_state, ref_path):
    """Calculates total cost of a trajectory for the optimizer."""
    controls = controls.reshape(-1, 2)
    horizon = len(ref_path)

    # Simulate forward pass
    trajectory = np.zeros((horizon, 4))
    trajectory[0] = initial_state
    total_cost = 0

    for k in range(horizon - 1):
        trajectory[k + 1] = simulate_dynamics(trajectory[k], controls[k])

        # Running cost
        pos_error = np.linalg.norm(trajectory[k][:2] - ref_path[k][:2])
        vel_cost = np.linalg.norm(trajectory[k][2:])
        control_cost = np.linalg.norm(controls[k])
        total_cost += W_POS * pos_error**2 + W_VEL * vel_cost**2 + W_CONTROL * control_cost**2

    # Terminal cost
    goal_state = np.array([ref_path[-1][0], ref_path[-1][1], 0, 0])
    final_error = np.linalg.norm(trajectory[-1] - goal_state)
    total_cost += (W_FINAL_POS + W_FINAL_VEL) * final_error**2
    return total_cost


def optimal_control_solve(initial_state, ref_path):
    """Finds a smooth trajectory that follows the reference path."""
    horizon = len(ref_path)
    initial_guess = np.zeros(2 * (horizon - 1))  # Initial controls are all zero

    # Use a standard optimizer (from SciPy) to find the best controls
    result = minimize(
        cost_function,
        initial_guess,
        args=(initial_state, ref_path),
        method="SLSQP",  # A good gradient-based method
        options={"maxiter": OPTIMAL_CONTROL_ITERATIONS},
    )

    optimal_controls = result.x.reshape(-1, 2)

    # Re-simulate to get the final, optimized trajectory
    final_trajectory = np.zeros((horizon, 4))
    final_trajectory[0] = initial_state
    for k in range(horizon - 1):
        final_trajectory[k + 1] = simulate_dynamics(final_trajectory[k], optimal_controls[k])

    return [tuple(row) for row in final_trajectory]


# --- High-Level Logic: Hybrid Prioritized Planning ---


def hybrid_prioritized_planning(agents):
    """Solves MAPF with dynamics using a hybrid RRT*/Optimal Control planner."""
    print("--- Starting Hybrid Dynamic Prioritized Planner ---")
    priority_order = sorted(agents.keys())
    final_trajectories = {}
    path_constraints = []

    for agent_id in priority_order:
        print(f"Planning for Agent {agent_id}...")
        start_state = agents[agent_id]["start"]
        goal_pos = agents[agent_id]["goal"]

        # Stage 1: Get a rough reference path from RRT*
        print("  - Stage 1: Finding global path with RRT*...")
        ref_path = rrt_star_search(start_state, goal_pos)
        if ref_path is None:
            print(f"  FAILED: RRT* could not find a path for Agent {agent_id}.")
            return None

        # Stage 2: Optimize and smooth the path
        print(f"  - Stage 2: Optimizing trajectory with {len(ref_path)} waypoints...")
        final_trajectory = optimal_control_solve(start_state, ref_path)

        print(f"  SUCCESS: Path found with {len(final_trajectory) - 1} steps.")
        final_trajectories[agent_id] = final_trajectory
        path_constraints.append(final_trajectory)

    return final_trajectories


# --- Visualization ---


def visualize_dynamic_solution(solution):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, MAX_X)
    ax.set_ylim(0, MAX_Y)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    obstacle_patch = Circle(OBSTACLE_CENTER, OBSTACLE_RADIUS, color="gray", alpha=0.6)
    ax.add_patch(obstacle_patch)
    colors = ["blue", "red"]
    agent_circles = [Circle((0, 0), AGENT_RADIUS, color=colors[i], fill=True, alpha=0.8) for i in solution]
    trajectory_lines = [ax.plot([], [], lw=2, color=colors[i], alpha=0.5)[0] for i in solution]
    for i, agent_id in enumerate(solution.keys()):
        goal = AGENTS[agent_id]["goal"]
        ax.plot(goal[0], goal[1], "x", color=colors[i], markersize=10, markeredgewidth=2)
        ax.add_patch(agent_circles[i])
    max_len = max(len(p) for p in solution.values())

    def animate(t):
        ax.set_title(f"Time: {t * TIMESTEP:.2f}s")
        for i, agent_id in enumerate(solution.keys()):
            path = solution[agent_id]
            state = path[t if t < len(path) else -1]
            agent_circles[i].center = (state[0], state[1])
            path_slice = path[: t + 1]
            x_coords = [s[0] for s in path_slice]
            y_coords = [s[1] for s in path_slice]
            trajectory_lines[i].set_data(x_coords, y_coords)
        return agent_circles + trajectory_lines

    animation.FuncAnimation(fig, animate, frames=max_len, interval=100, blit=True, repeat=False)
    plt.show()


# --- Main Execution ---

if __name__ == "__main__":
    solution = hybrid_prioritized_planning(AGENTS)
    if solution:
        print("\n--- Final Solution Summary ---")
        for agent_id, path in solution.items():
            print(f"Agent {agent_id}: Path length = {len(path) - 1} steps, Time = {(len(path) - 1) * TIMESTEP:.2f}s")
        visualize_dynamic_solution(solution)
    else:
        print("\nCould not find a solution for all agents.")
