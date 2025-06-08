import collections
import itertools

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

# --- Environment and Agent Definition ---
MAX_ROWS = 20
MAX_COLS = 20

AGENTS = {0: {"start": (0, 0), "goal": (19, 19)}, 1: {"start": (0, 19), "goal": (19, 0)}}


def run_centralized_value_iteration():
    """
    Performs Value Iteration on the joint state space to find the minimum
    cost-to-go from every state, now including diagonal moves.
    """
    print("--- Starting Centralized Dynamic Programming (with Diagonal Moves) ---")

    value_function = np.full((MAX_ROWS, MAX_COLS, MAX_ROWS, MAX_COLS), np.inf)

    goal_state = (AGENTS[0]["goal"], AGENTS[1]["goal"])
    goal_indices = goal_state[0] + goal_state[1]
    value_function[goal_indices] = 0

    queue = collections.deque([goal_state])
    print("Initializing DP... Propagating costs backwards from the goal.")

    # MODIFICATION: Expanded move set to include diagonals
    moves = [
        (0, 0),
        (0, 1),
        (0, -1),
        (1, 0),
        (-1, 0),  # Cardinal + Wait
        (1, 1),
        (1, -1),
        (-1, 1),
        (-1, -1),  # Diagonal
    ]
    all_joint_moves = list(itertools.product(moves, repeat=2))

    count = 0
    while queue:
        count += 1
        if count % 20000 == 0:
            print(f"  Processed {count} states...")

        current_state = queue.popleft()
        current_val_indices = current_state[0] + current_state[1]
        current_cost = value_function[current_val_indices]

        for joint_move in all_joint_moves:
            pred_pos1 = (current_state[0][0] - joint_move[0][0], current_state[0][1] - joint_move[0][1])
            pred_pos2 = (current_state[1][0] - joint_move[1][0], current_state[1][1] - joint_move[1][1])
            predecessor_state = (pred_pos1, pred_pos2)

            if not (
                0 <= pred_pos1[0] < MAX_ROWS
                and 0 <= pred_pos1[1] < MAX_COLS
                and 0 <= pred_pos2[0] < MAX_ROWS
                and 0 <= pred_pos2[1] < MAX_COLS
            ):
                continue
            if pred_pos1 == pred_pos2:
                continue
            if pred_pos1 == current_state[1] and pred_pos2 == current_state[0]:
                continue

            pred_val_indices = pred_pos1 + pred_pos2
            new_cost_for_pred = current_cost + 1

            if new_cost_for_pred < value_function[pred_val_indices]:
                value_function[pred_val_indices] = new_cost_for_pred
                queue.append(predecessor_state)

    print("DP cost propagation complete.")
    return value_function


def extract_path_from_value_function(value_function, agents):
    """
    Reconstructs the optimal path by greedily moving to the state
    with the lowest cost-to-go value, now including diagonal moves.
    """
    print("Extracting optimal path by descending the value function...")
    start_state = (agents[0]["start"], agents[1]["start"])
    goal_state = (agents[0]["goal"], agents[1]["goal"])

    start_val_indices = start_state[0] + start_state[1]
    if np.isinf(value_function[start_val_indices]):
        return None

    solution = {0: [], 1: []}
    current_state = start_state

    # MODIFICATION: Expanded move set to include diagonals
    moves = [
        (0, 0),
        (0, 1),
        (0, -1),
        (1, 0),
        (-1, 0),  # Cardinal + Wait
        (1, 1),
        (1, -1),
        (-1, 1),
        (-1, -1),  # Diagonal
    ]
    all_joint_moves = list(itertools.product(moves, repeat=2))

    while current_state != goal_state:
        solution[0].append(current_state[0])
        solution[1].append(current_state[1])

        best_next_state = None
        min_next_cost = np.inf

        for joint_move in all_joint_moves:
            next_pos1 = (current_state[0][0] + joint_move[0][0], current_state[0][1] + joint_move[0][1])
            next_pos2 = (current_state[1][0] + joint_move[1][0], current_state[1][1] + joint_move[1][1])
            next_state = (next_pos1, next_pos2)

            if not (
                0 <= next_pos1[0] < MAX_ROWS
                and 0 <= next_pos1[1] < MAX_COLS
                and 0 <= next_pos2[0] < MAX_ROWS
                and 0 <= next_pos2[1] < MAX_COLS
            ):
                continue

            if next_pos1 == next_pos2 or (next_pos1 == current_state[1] and next_pos2 == current_state[0]):
                continue

            next_val_indices = next_pos1 + next_pos2
            cost = value_function[next_val_indices]

            if cost < min_next_cost:
                min_next_cost = cost
                best_next_state = next_state

        if best_next_state is None:
            print("Error: Got stuck during path extraction.")
            return None

        current_state = best_next_state

    solution[0].append(goal_state[0])
    solution[1].append(goal_state[1])

    return solution


def visualize_solution(solution, agents):
    """Animates the solution paths with trajectory trails."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-1.5, MAX_COLS - 0.5)
    ax.set_ylim(-1.5, MAX_ROWS - 0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(range(MAX_COLS))
    ax.set_yticks(range(MAX_ROWS))
    ax.grid(True)
    ax.invert_yaxis()

    colors = ["blue", "red"]
    agent_circles = [plt.Circle((0, 0), 0.4, color=colors[i], fill=True) for i in solution]
    goal_crosses = [
        ax.text(0, 0, "X", va="center", ha="center", color=colors[i], fontsize=20, fontweight="bold")
        for i in solution
    ]
    trajectory_lines = [ax.plot([], [], lw=2, color=colors[i])[0] for i in solution]

    for i, agent_id in enumerate(solution.keys()):
        goal = agents[agent_id]["goal"]
        goal_crosses[i].set_position((goal[1], goal[0]))
        ax.add_patch(agent_circles[i])

    max_len = max(len(p) for p in solution.values()) if solution else 0

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

    ani = animation.FuncAnimation(fig, animate, frames=max_len, interval=250, blit=False, repeat=False)
    plt.show()


def print_solution_details(solution):
    """
    Prints the position of each agent at each timestep and checks for collisions.
    """
    print("\n--- Detailed Path Execution ---")

    if not solution:
        print("No solution to detail.")
        return

    # Determine the total duration of the plan (length of the longest path)
    max_len = max(len(p) for p in solution.values()) if solution else 0

    # Get sorted agent IDs for consistent printing order
    agent_ids = sorted(solution.keys())

    for t in range(max_len):
        print(f"Timestep {t}:")
        positions_at_t = {}

        # Get position of each agent at the current timestep
        for agent_id in agent_ids:
            path = solution[agent_id]
            # If agent reached its goal, it waits there indefinitely
            pos = path[t] if t < len(path) else path[-1]
            positions_at_t[agent_id] = pos
            print(f"  - Agent {agent_id} at {pos}")

        # Check for collisions at this timestep
        # We convert the list of positions to a set to find duplicates
        if len(set(positions_at_t.values())) < len(positions_at_t):
            print("  >>> COLLISION HAPPENED! <<<")


if __name__ == "__main__":
    # 1. Run Value Iteration to compute the entire cost-to-go landscape
    value_function = run_centralized_value_iteration()

    # 2. Extract the single optimal path from the DP table
    solution = extract_path_from_value_function(value_function, AGENTS)

    if solution:
        print("\n--- Final Solution Summary ---")
        for agent_id, path in solution.items():
            print(f"Agent {agent_id} (Start: {AGENTS[agent_id]['start']}, Goal: {AGENTS[agent_id]['goal']}):")
            print(f"  Cost (Timesteps): {len(path) - 1}")

        # 3. Print the detailed step-by-step path and check for collisions
        print_solution_details(solution)

        # 4. Animate the solution
        visualize_solution(solution, AGENTS)
    else:
        print("No solution could be found.")
