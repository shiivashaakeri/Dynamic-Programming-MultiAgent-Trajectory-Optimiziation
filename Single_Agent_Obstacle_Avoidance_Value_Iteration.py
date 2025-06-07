import functools

import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import interact

# problem set up
max_rows, max_columns = 20, 20
fire_states = {(10, 10), (11, 10), (10, 11), (11, 11), (13, 4), (13, 5), (14, 4), (14, 5)}
storm_eye = (10, 6)
storm_sigma = 0
goal_states = {(12, 12)}
terminal_states = (None, None)
gamma = 0.95
fire_value = -200
goal_value = 200
travel_value = -1
action_set = ["down", "right", "up", "left"]


def is_terminal_state(state):
    """
    Check if the state is a terminal state.
    Args:
        state: Current state (row, column).
    Returns:
        True if the state is terminal, False otherwise.
    """
    return state == terminal_states


def _state_space(max_rows, max_columns):
    return [(i, j) for i in range(max_rows) for j in range(max_columns)] + [(None, None)]


def _reward(state, fire_states, goal_states, fire_value, goal_value, travel_value):
    """
    Reward function for the grid world.
    Args:
        state: Current state (row, column).
        fire_states: List or set of fire states.
        goal_states: List or set of goal states.
        fire_value: Reward value for fire states.
        goal_value: Reward value for goal states.
    Returns:
        Reward value for the current state.
    """
    #### FILL CODE HERE ####
    if state == terminal_states:
        reward_value = 0
    elif state in goal_states:
        reward_value = goal_value + travel_value
    elif state in fire_states:
        reward_value = fire_value + travel_value
    else:
        reward_value = travel_value
    return reward_value
    ########################


def _transition_function(
    s,
    a,
    w=0,
    max_rows=20,  # number of rows
    max_columns=20,  # number of columns
    goal_states={},
    action_set=["down", "right", "up", "left"],
):
    """
    Transition function for the grid world.
    Args:
        s: Current state (row, column).
        a: Action to take.
        w: Probability of taking the action.
        max_rows: Number of rows in the grid.
        max_columns: Number of columns in the
            grid.
        action_set: List of possible actions.
    Returns:
        New state after taking the action.
    """
    i, j = s
    if is_terminal_state(s) or (s in goal_states):
        return (None, None)
    if (np.random.rand(1) < w)[0]:
        a = np.random.choice(action_set)
    if a == "up":
        return (min(i + 1, max_rows - 1), j)
    if a == "right":
        return (i, min(j + 1, max_columns - 1))
    if a == "down":
        return (max(i - 1, 0), j)
    if a == "left":
        return (i, max(j - 1, 0))


def _compute_omega_probability(state):
    """
    Computes the probability of a state being affected by a storm.
    Args:
        state: Current state (row, column).
        storm_eye: Center of the storm (row, column).
        storm_sigma: Standard deviation of the storm.
    Returns:
        Probability of the state being affected by the storm.
    """
    if state[0] is None:
        print("None")
    if is_terminal_state(state):
        return 0
    else:
        return 0


# fix the problem parameters in the functions to avoid passing them every time
state_space = functools.partial(_state_space, max_rows=max_rows, max_columns=max_columns)  ## fix the desried args
reward = functools.partial(
    _reward,
    fire_states=fire_states,
    goal_states=goal_states,
    fire_value=fire_value,
    goal_value=goal_value,
    travel_value=travel_value,
)  ## fix all args except state
transition_function = functools.partial(
    _transition_function, max_rows=max_rows, max_columns=max_columns, goal_states=goal_states, action_set=action_set
)
compute_omega_probability = functools.partial(_compute_omega_probability)


def probability_function(state, action, next_state, w):
    action_set = ["down", "right", "up", "left"]
    """
    Computes the probability of transitioning to a next state given the current state and action.
    Args:
        state: Current state (row, column).
        action: Action to take.
        next_state: Next state (row, column).
        w: Probability of taking random action.
        action_set: List of possible actions.
    Returns:
        Probability of transitioning to the next state according to the current action.
    """

    #### FILL CODE HERE ####
    # HINT: Our solution takes ~3 lines of code
    next_state_list = get_possible_next_states(state, action_set)  ## all possible next states
    next_state_det = transition_function(state, action)  ## deterministic next state
    if next_state == next_state_det:
        prob = 1 - w + w / len(next_state_list)  # determinsistic + random
    elif next_state in next_state_list:
        prob = w / len(next_state_list) if next_state in next_state_list else 0
    return prob


def get_possible_next_states(state, action_set):
    """
    Returns the set of possible next states given the current state.
    Args:
        state: Current state (row, column).
    Returns:
        Set of possible next states.
    """
    return {transition_function(state, action, w=0) for action in action_set}


def bellman_update(value_tuple, gamma, action_set):
    """
    Performs a Bellman update on the value function.
    Args:
        value_tuple: Current value function. A tuple of (value, value_terminal).
        value: Array representing the value at each state in the grid
        value_terminal: Value of the terminal state.
        gamma: Discount factor.
        action_set: List of possible actions.
    Returns:
        Updated value_tuple and policy as a dictionary.
    """
    value_old = value_tuple[0]
    value_new = np.zeros((max_rows, max_columns))
    policy = {}

    for i in range(max_rows):
        for j in range(max_columns):
            state = (i, j)
            if state == terminal_states:
                continue
            w = compute_omega_probability(state)
            ## get all possible next states
            next_state_all = get_possible_next_states(state, action_set)
            value_new_ij = np.zeros(len(action_set))  ## 4 possible actions
            count = 0
            for action in action_set:
                for next_possble_state in next_state_all:
                    state_reward = reward(next_possble_state)
                    prob_tran = probability_function(state, action, next_possble_state, w)
                    value_old_ij = 0.0 if next_possble_state == terminal_states else value_old[next_possble_state]
                    value_new_ij[count] = value_new_ij[count] + prob_tran * (state_reward + gamma * value_old_ij)
                count = count + 1

            value_new[i, j] = np.max(value_new_ij)
            policy[state] = action_set[np.argmax(value_new_ij)]

    return (value_new, goal_value), policy


def simulate(start_state, policy, num_steps):
    """
    Simulates the agent's trajectory in the grid world.
    Args:
        start_state: Starting state (row, column).
        policy: Policy to follow.
        num_steps: Number of steps to simulate.
    Returns:
        List of states visited during the simulation.
    """
    states = [start_state]
    for _ in range(num_steps):
        action = policy[start_state]
        w = compute_omega_probability(start_state)
        next_state = transition_function(start_state, action, w=w)
        if is_terminal_state(next_state):
            break
        start_state = next_state
        states.append(start_state)
    return states


# Initialize the value function
V = (np.zeros([max_rows, max_columns]), 0)
# keep list of value functions
Vs = [V]
dV = []
num_iterations = 100  # feel free to change this value as needed
for _ in range(num_iterations):
    # perform Bellman update
    V_new, policy = bellman_update(V, gamma, action_set)
    # store the new value function
    Vs.append(V_new)
    dV.append(np.abs(V_new[0] - V[0]).max())

    # check for convergence
    if np.abs(V_new[0] - V[0]).max() < 1e-5:
        print("Converged!")
        break
    # update the value function
    V = V_new

start_state = (0, 0)  # pick a starting state
num_steps = 500  # feel free to change this value as needed
# simulate the trajectory
trajectory = simulate(start_state, policy, num_steps)

plt.figure(figsize=(4, 2))
plt.plot(dV)
plt.title("Convergence of Value Function")
plt.xlabel("Iteration")
plt.ylabel("Max Change in Value Function")
plt.grid()


def plot_policy(policy):
    for (row, col), action in policy.items():
        if row is None or col is None:
            continue
        if action == "up":
            plt.text(col + 0.5, row + 0.5, "↑", ha="center", va="center", color="black", fontsize=8)
        elif action == "down":
            plt.text(col + 0.5, row + 0.5, "↓", ha="center", va="center", color="black", fontsize=8)
        elif action == "left":
            plt.text(col + 0.5, row + 0.5, "←", ha="center", va="center", color="black", fontsize=8)
        elif action == "right":
            plt.text(col + 0.5, row + 0.5, "→", ha="center", va="center", color="black", fontsize=8)


# compute the storm strength for each state for plotting later
storm_strength = np.zeros([max_rows, max_columns])
for state in state_space():
    if not is_terminal_state(state):
        storm_strength[state] = compute_omega_probability(state)


# visualize the value function and storm strength
@interact(iteration=(0, len(Vs) - 1, 1), t=(0, len(trajectory) - 1, 1))
def plot_value_function(iteration, t):
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(
        Vs[iteration][0], origin="lower", extent=[0, max_columns, 0, max_rows], cmap="viridis", interpolation="nearest"
    )
    plt.colorbar(label="Value")
    plt.title("Value Function")
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.xticks(ticks=np.arange(0.5, max_columns, 1), labels=np.arange(0, max_columns))
    plt.yticks(ticks=np.arange(0.5, max_rows, 1), labels=np.arange(0, max_rows))
    plt.scatter(storm_eye[1] + 0.5, storm_eye[0] + 0.5, c="cyan", s=100, label="Storm Eye")
    for fire_state in fire_states:
        plt.scatter(fire_state[1] + 0.5, fire_state[0] + 0.5, c="red", s=100)
    plt.scatter(fire_state[1] + 0.5, fire_state[0] + 0.5, c="red", s=100, label="Fire State")
    for goal_state in goal_states:
        plt.scatter(goal_state[1] + 0.5, goal_state[0] + 0.5, c="green", s=100, label="Goal State")

    # Overlay the policy
    plot_policy(policy)
    # Plot the trajectory
    trajectory_x = [state[1] + 0.5 for state in trajectory]
    trajectory_y = [state[0] + 0.5 for state in trajectory]
    plt.plot(trajectory_x, trajectory_y, color="orange", label="Trajectory", linewidth=2)
    plt.scatter(trajectory_x[t], trajectory_y[t], color="orange", s=100, label="Current State")
    plt.legend(loc="lower left", framealpha=0.6)

    plt.subplot(1, 2, 2)
    plt.imshow(
        storm_strength, origin="lower", extent=[0, max_columns, 0, max_rows], cmap="viridis", interpolation="nearest"
    )
    plt.colorbar(label="Storm Strength")
    plt.title("Storm Strength")
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.xticks(ticks=np.arange(0.5, max_columns, 1), labels=np.arange(0, max_columns))
    plt.yticks(ticks=np.arange(0.5, max_rows, 1), labels=np.arange(0, max_rows))
    plt.show()
