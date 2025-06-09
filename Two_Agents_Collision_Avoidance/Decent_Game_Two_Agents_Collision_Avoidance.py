import heapq
import time
import tracemalloc
from collections import namedtuple

import matplotlib.pyplot as plt
from matplotlib import animation

# --- Global Constants and Definitions ---

# Grid dimensions
MAX_ROWS = 20
MAX_COLS = 20

# Agent definitions
AGENTS = {0: {"start": (0, 0), "goal": (19, 19)}, 1: {"start": (0, 19), "goal": (19, 0)}}

# Static obstacle definition
OBSTACLE_CENTER = (10, 8)
OBSTACLE_RADIUS = 2

# A simplified constraint for this algorithm: (location, timestep)
# It represents a space-time point that is occupied by another agent.
Constraint = namedtuple("Constraint", ["loc", "timestep"])


# --- Helper Functions ---


def generate_obstacle_set(center, radius):
    """
    Returns a set of (row, col) tuples that are inside the circular obstacle.
    """
    obstacles = set()
    radius_sq = radius**2
    for r in range(MAX_ROWS):
        for c in range(MAX_COLS):
            dist_sq = (r - center[0]) ** 2 + (c - center[1]) ** 2
            if dist_sq <= radius_sq:
                obstacles.add((r, c))
    return obstacles


def manhattan_distance(loc1, loc2):
    """
    Calculates the Chebyshev distance for a grid with diagonal moves.
    This is the number of steps a king would take on a chessboard.
    """
    return max(abs(loc1[0] - loc2[0]), abs(loc1[1] - loc2[1]))


def reconstruct_path(came_from, current_state):
    """
    Rebuilds the path from the 'came_from' dictionary returned by A*.
    """
    path = []
    while current_state in came_from:
        _, loc = current_state
        path.append(loc)
        current_state = came_from[current_state]
    path.append(current_state[1])
    return path[::-1]


# --- Core Algorithm Functions ---


def a_star_search(start_loc, goal_loc, constraints, obstacles):
    """
    Finds an optimal path for a single agent, respecting dynamic agent constraints
    and static obstacles.
    """
    open_list = []
    g_cost = 0
    h_cost = manhattan_distance(start_loc, goal_loc)
    f_cost = g_cost + h_cost
    heapq.heappush(open_list, (f_cost, g_cost, start_loc, None))
    came_from = {}
    cost_so_far = {(0, start_loc): 0}
    nodes_expanded = 0

    while open_list:
        nodes_expanded += 1
        _, time, loc, _ = heapq.heappop(open_list)

        if loc == goal_loc:
            return reconstruct_path(came_from, (time, loc)), nodes_expanded

        # Explore neighbors (cardinal, diagonal, and wait moves)
        for move in [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            next_loc = (loc[0] + move[0], loc[1] + move[1])
            next_time = time + 1

            # Check grid boundaries
            if not (0 <= next_loc[0] < MAX_ROWS and 0 <= next_loc[1] < MAX_COLS):
                continue

            # Check for collision with static obstacles
            if next_loc in obstacles:
                continue

            # Check for collision with other agents' paths (dynamic constraints)
            if Constraint(next_loc, next_time) in constraints:
                continue

            new_cost = next_time
            if (next_time, next_loc) not in cost_so_far or new_cost < cost_so_far[(next_time, next_loc)]:
                cost_so_far[(next_time, next_loc)] = new_cost
                h_cost = manhattan_distance(next_loc, goal_loc)
                f_cost = new_cost + h_cost
                heapq.heappush(open_list, (f_cost, new_cost, next_loc, (time, loc)))
                came_from[(next_time, next_loc)] = (time, loc)

    return None, nodes_expanded


def prioritized_planning(agents, obstacles):
    """
    Solves the MAPF problem using a decentralized, prioritized planning approach.
    """
    print("--- Starting Decentralized Prioritized Planner ---")
    priority_order = sorted(agents.keys())
    solution = {}
    cumulative_constraints = set()
    total_nodes_expanded = 0

    for agent_id in priority_order:
        print(f"Planning for Agent {agent_id} (Priority {agent_id})...")
        print(f"  Current number of constraints: {len(cumulative_constraints)}")

        path, nodes_expanded = a_star_search(
            agents[agent_id]["start"], agents[agent_id]["goal"], cumulative_constraints, obstacles
        )
        total_nodes_expanded += nodes_expanded

        if path is None:
            print(f"  FAILED: Could not find a path for Agent {agent_id}.")
            return None, total_nodes_expanded

        print(f"  SUCCESS: Path found with cost {len(path) - 1}.")
        solution[agent_id] = path

        # Add this agent's path to the set of constraints for subsequent agents
        for t, pos in enumerate(path):
            cumulative_constraints.add(Constraint(pos, t))
        # Assume agent waits at its goal indefinitely to block the spot
        goal_pos = path[-1]
        for t in range(len(path), len(path) + 50):
            cumulative_constraints.add(Constraint(goal_pos, t))

    print(f"All agents planned successfully. Total nodes expanded: {total_nodes_expanded}")
    return solution, total_nodes_expanded


# --- Visualization and Reporting ---


def print_solution_details(solution):
    """Prints the position of each agent at each timestep."""
    print("\n--- Detailed Path Execution ---")
    if not solution:
        return
    max_len = max(len(p) for p in solution.values())
    agent_ids = sorted(solution.keys())
    for t in range(max_len):
        print(f"Timestep {t}:")
        positions = [solution[aid][t] if t < len(solution[aid]) else solution[aid][-1] for aid in agent_ids]
        for i, pos in enumerate(positions):
            print(f"  - Agent {agent_ids[i]} at {pos}")
        if len(set(positions)) < len(positions):
            print("  >>> COLLISION DETECTED! <<<")


def visualize_solution(solution, agents, obstacles):
    """Animates the solution paths with trajectory trails and obstacles."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-1.5, MAX_COLS - 0.5)
    ax.set_ylim(-1.5, MAX_ROWS - 0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(range(MAX_COLS))
    ax.set_yticks(range(MAX_ROWS))
    ax.grid(True)
    ax.invert_yaxis()

    # Draw the static obstacle cells
    for obs_loc in obstacles:
        rect = plt.Rectangle((obs_loc[1] - 0.5, obs_loc[0] - 0.5), 1, 1, facecolor="gray", alpha=0.6)
        ax.add_patch(rect)

    colors = ["blue", "red"]
    agent_circles = [plt.Circle((0, 0), 0.4, color=colors[i], fill=True) for i in solution]
    goal_crosses = [
        ax.text(0, 0, "X", va="center", ha="center", color=colors[i], fontsize=20, fontweight="bold") for i in solution
    ]
    trajectory_lines = [ax.plot([], [], lw=2, color=colors[i])[0] for i in solution]

    for i, agent_id in enumerate(solution.keys()):
        goal = agents[agent_id]["goal"]
        goal_crosses[i].set_position((goal[1], goal[0]))
        ax.add_patch(agent_circles[i])

    max_len = max(len(p) for p in solution.values())

    def animate(t):
        ax.set_title(f"Timestep: {t}")
        for i, agent_id in enumerate(solution.keys()):
            path = solution[agent_id]
            pos = path[t] if t < len(path) else path[-1]
            agent_circles[i].center = (pos[1], pos[0])
            path_slice = path[: t + 1]
            x_coords = [p[1] for p in path_slice]
            y_coords = [p[0] for p in path_slice]
            trajectory_lines[i].set_data(x_coords, y_coords)
        return agent_circles + trajectory_lines

    ani = animation.FuncAnimation(fig, animate, frames=max_len, interval=250, blit=False, repeat=False)  # noqa: F841
    plt.show()


# --- Main Execution Block ---

if __name__ == "__main__":
    # 1. Generate the obstacle set from the defined circle
    obstacle_set = generate_obstacle_set(OBSTACLE_CENTER, OBSTACLE_RADIUS)
    print(f"Generated {len(obstacle_set)} obstacle cells.")

    # 2. Start benchmarking
    tracemalloc.start()
    start_time = time.perf_counter()

    # 3. Run the planner
    solution, states_processed = prioritized_planning(AGENTS, obstacle_set)

    end_time = time.perf_counter()
    duration = end_time - start_time
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # 4. Print results
    print("\n" + "=" * 50)
    print("    BENCHMARK RESULTS (Decentralized Prioritized)")
    print("=" * 50)

    if solution:
        solution_cost = sum(len(p) - 1 for p in solution.values())
        print("Solution Found!")
        print(f"  - Solution Cost (Total Timesteps): {solution_cost}")
        print(f"  - Execution Time: {duration:.4f} seconds")
        print(f"  - Peak Memory Usage: {peak_mem / 10**6:.3f} MB")
        print(f"  - States/Nodes Processed: {states_processed}")
        print("=" * 50 + "\n")

        print_solution_details(solution)
        visualize_solution(solution, AGENTS, obstacle_set)
    else:
        print("No solution could be found.")
        print(f"  - Execution Time: {duration:.4f} seconds")
        print(f"  - Peak Memory Usage: {peak_mem / 10**6:.3f} MB")
        print(f"  - States/Nodes Processed: {states_processed}")
        print("=" * 50 + "\n")
