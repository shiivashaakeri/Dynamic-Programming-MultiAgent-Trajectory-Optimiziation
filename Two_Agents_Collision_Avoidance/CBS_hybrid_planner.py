import heapq
import random

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.patches import Circle
from scipy.optimize import minimize

# --- Global Constants and Definitions ---

# Environment
MAX_X, MAX_Y = 20, 20
TIMESTEP = 0.4

# Agent Dynamics & Collision
THRUST_POWER = 3.0
MAX_SPEED = 5.0
AGENT_RADIUS = 0.5

# RRT* (Global Planner) Parameters
RRT_ITERATIONS = 2000
RRT_GOAL_SAMPLE_RATE = 0.2

# Optimal Control (Local Planner) Parameters
OPTIMAL_CONTROL_ITERATIONS = 40
W_POS, W_VEL, W_CONTROL = 1.0, 0.1, 0.01
W_FINAL_POS, W_FINAL_VEL = 100.0, 100.0
W_OBSTACLE = 200.0  # NEW: High penalty for obstacle violation

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


# --- Data Structures for CBS ---
class HighLevelNode:
    """A node in the Conflict-Based Search tree."""

    def __init__(self):
        self.solution = {}
        self.constraints = set()
        self.cost = 0

    def __lt__(self, other):
        return self.cost < other.cost


# --- Physics and Low-Level Collision Checking ---
def simulate_dynamics(state, action, dt=TIMESTEP):
    """Calculates the next state given the current state and an action."""
    x, y, vx, vy = state
    ax, ay = action
    vx_new = vx + ax * dt
    vy_new = vy + ay * dt
    speed = np.sqrt(vx_new**2 + vy_new**2)
    if speed > MAX_SPEED:
        vx_new, vy_new = (vx_new / speed) * MAX_SPEED, (vy_new / speed) * MAX_SPEED
    x_new = x + vx_new * dt
    y_new = y + vy_new * dt
    return np.array([x_new, y_new, vx_new, vy_new])


def is_path_segment_in_obstacle(p1_state, p2_state):
    """
    Checks if the line segment between two states intersects the circular obstacle.
    This uses a geometric line-segment vs. circle intersection test.
    """
    p1 = p1_state[:2]
    p2 = p2_state[:2]
    center = np.array(OBSTACLE_CENTER)

    d = p2 - p1  # Direction vector of the segment
    f = p1 - center  # Vector from circle center to segment start

    a = np.dot(d, d)
    b = 2 * np.dot(f, d)
    c = np.dot(f, f) - OBSTACLE_RADIUS**2

    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        # No real intersection with the line.
        return False
    else:
        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (2 * a)
        t2 = (-b + discriminant) / (2 * a)

        # Check if any intersection point is within the segment (t between 0 and 1)
        if (t1 >= 0 and t1 <= 1) or (t2 >= 0 and t2 <= 1):
            return True
        # Check for the case where the segment is entirely inside the circle
        return bool(t1 < 0 and t2 > 1)


# --- Stage 1: RRT for Global Pathfinding (Low-Level) ---
class RRTNode:
    def __init__(self, state, parent=None):
        self.state = np.array(state)
        self.parent = parent


def rrt_search(start_state, goal_pos, constraints):
    """Finds a feasible path using RRT, respecting CBS constraints."""
    start_node = RRTNode(start_state)
    nodes = [start_node]
    agent_id = constraints[0][0] if constraints else -1

    for _ in range(RRT_ITERATIONS):
        rand_pos = (
            goal_pos if random.random() < RRT_GOAL_SAMPLE_RATE else (random.uniform(0, MAX_X), random.uniform(0, MAX_Y))
        )
        nearest_node = min(nodes, key=lambda n: np.linalg.norm(n.state[:2] - rand_pos))

        direction = rand_pos - nearest_node.state[:2]
        action = (direction / (np.linalg.norm(direction) + 1e-6)) * THRUST_POWER
        new_state = simulate_dynamics(nearest_node.state, action)

        if not (0 <= new_state[0] < MAX_X and 0 <= new_state[1] < MAX_Y):
            continue

        # FIXED: Use segment-based collision check
        if is_path_segment_in_obstacle(nearest_node.state, new_state):
            continue

        valid = True
        for cons_agent_id, cons_loc, cons_time in constraints:
            if cons_agent_id == agent_id and np.linalg.norm(new_state[:2] - cons_loc) < AGENT_RADIUS:
                valid = False
                break
        if not valid:
            continue

        nodes.append(RRTNode(new_state, parent=nearest_node))

    goal_node = min(nodes, key=lambda n: np.linalg.norm(n.state[:2] - goal_pos), default=None)
    if goal_node is None:
        return None
    path = []
    current = goal_node
    while current is not None:
        path.append(current.state)
        current = current.parent
    return path[::-1]


# --- Stage 2: Optimal Control for Smoothing (Low-Level) ---
def optimal_control_solve(initial_state, ref_path):
    """Finds a smooth trajectory that follows the reference path."""
    horizon = len(ref_path)

    def cost_function(controls, initial_state, ref_path):
        controls = controls.reshape(-1, 2)
        trajectory = np.zeros((horizon, 4))
        trajectory[0] = initial_state
        total_cost = 0
        for k in range(horizon - 1):
            trajectory[k + 1] = simulate_dynamics(trajectory[k], controls[k])
            pos_error = np.linalg.norm(trajectory[k][:2] - ref_path[k][:2])
            vel_cost = np.linalg.norm(trajectory[k][2:])
            control_cost = np.linalg.norm(controls[k])

            # FIXED: Add a penalty for violating the static obstacle
            obstacle_penalty = 0
            dist_to_obs = np.linalg.norm(trajectory[k][:2] - np.array(OBSTACLE_CENTER))
            if dist_to_obs < OBSTACLE_RADIUS:
                obstacle_penalty = W_OBSTACLE * (OBSTACLE_RADIUS - dist_to_obs) ** 2

            total_cost += W_POS * pos_error**2 + W_VEL * vel_cost**2 + W_CONTROL * control_cost**2 + obstacle_penalty

        goal_state = np.array([ref_path[-1][0], ref_path[-1][1], 0, 0])
        final_error = np.linalg.norm(trajectory[-1] - goal_state)
        total_cost += (W_FINAL_POS + W_FINAL_VEL) * final_error**2
        return total_cost

    initial_guess = np.zeros(2 * (horizon - 1))
    result = minimize(
        cost_function,
        initial_guess,
        args=(initial_state, ref_path),
        method="SLSQP",
        options={"maxiter": OPTIMAL_CONTROL_ITERATIONS},
    )
    optimal_controls = result.x.reshape(-1, 2)
    final_trajectory = np.zeros((horizon, 4))
    final_trajectory[0] = initial_state
    for k in range(horizon - 1):
        final_trajectory[k + 1] = simulate_dynamics(final_trajectory[k], optimal_controls[k])
    return [tuple(row) for row in final_trajectory]


# --- Low-Level Planner Wrapper ---
def plan_path_for_agent(agent_id, start_state, goal_pos, constraints):
    """Runs the full RRT* + OCP pipeline for a single agent."""
    agent_constraints = [c for c in constraints if c[0] == agent_id]
    ref_path = rrt_search(start_state, goal_pos, agent_constraints)
    if ref_path is None:
        return None
    return optimal_control_solve(start_state, ref_path)


# --- High-Level Logic: Conflict-Based Search ---
def find_first_dynamic_conflict(solution):
    agent_ids = sorted(solution.keys())
    max_len = max(len(p) for p in solution.values())
    for t in range(max_len):
        for i in range(len(agent_ids)):
            for j in range(i + 1, len(agent_ids)):
                id1, id2 = agent_ids[i], agent_ids[j]
                path1, path2 = solution[id1], solution[id2]
                pos1 = path1[t][:2] if t < len(path1) else path1[-1][:2]
                pos2 = path2[t][:2] if t < len(path2) else path2[-1][:2]
                if np.linalg.norm(np.array(pos1) - np.array(pos2)) < 2 * AGENT_RADIUS:
                    return {"agents": (id1, id2), "loc": pos1, "time": t}
    return None


def cbs_hybrid_planner(agents):
    print("--- Starting CBS Hybrid Planner ---")
    open_list = []
    root = HighLevelNode()
    for agent_id, info in agents.items():
        print(f"Planning initial path for Agent {agent_id}...")
        path = plan_path_for_agent(agent_id, info["start"], info["goal"], [])
        if path is None:
            print(f"FATAL: Could not find initial path for Agent {agent_id}.")
            return None
        root.solution[agent_id] = path
    root.cost = sum(len(p) for p in root.solution.values())
    heapq.heappush(open_list, root)
    while open_list:
        current_node = heapq.heappop(open_list)
        conflict = find_first_dynamic_conflict(current_node.solution)
        if conflict is None:
            print("SUCCESS: Found a conflict-free solution!")
            return current_node.solution
        print(
            (
                f"Conflict found at t={conflict['time']} between agents {conflict['agents']}."
                f" Cost: {current_node.cost}. Resolving..."
            )
        )
        id1, id2 = conflict["agents"]
        for agent_to_constrain in [id1, id2]:
            new_node = HighLevelNode()
            new_node.constraints = current_node.constraints.copy()
            new_constraint = (agent_to_constrain, conflict["loc"], conflict["time"])
            new_node.constraints.add(new_constraint)
            print(f"  - Re-planning for Agent {agent_to_constrain} with new constraint...")
            new_node.solution = current_node.solution.copy()
            new_path = plan_path_for_agent(
                agent_to_constrain,
                agents[agent_to_constrain]["start"],
                agents[agent_to_constrain]["goal"],
                list(new_node.constraints),
            )
            if new_path:
                new_node.solution[agent_to_constrain] = new_path
                new_node.cost = sum(len(p) for p in new_node.solution.values())
                heapq.heappush(open_list, new_node)
    return None


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
    trajectory_lines = [ax.plot([], [], lw=2, color=colors[i], alpha=0.6)[0] for i in solution]
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

    ani = animation.FuncAnimation(fig, animate, frames=max_len, interval=200, blit=True, repeat=False)  # noqa: F841
    plt.show()


# --- Main Execution ---
if __name__ == "__main__":
    solution = cbs_hybrid_planner(AGENTS)
    if solution:
        print("\n--- Final Solution Summary (CBS) ---")
        for agent_id, path in solution.items():
            print(f"Agent {agent_id}: Path length = {len(path) - 1} steps, Time = {(len(path) - 1) * TIMESTEP:.2f}s")
        visualize_dynamic_solution(solution)
    else:
        print("\nCould not find a conflict-free solution.")
