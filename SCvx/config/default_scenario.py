import numpy as np

# Number of discretization points
K = 50

# Minimum inter-agent separation distance
D_MIN = 0.5

# Clearance for obstacle inflation in the initializer
CLEARANCE = 0.05

# Additional margin to start/stop outside the clearance band
MARGIN = 0.6

# Static obstacles for the scenario: list of (center, radius)
# All agents reference these by default
OBSTACLES = [
    ([1.0, 1.0], 0.25),  # single small obstacle at (1,1)
]

# Agent definitions - list of dicts passed to MultiAgentModel
AGENT_PARAMS = [
    # Agent 0
    {
        'r_init':    np.array([0.0, 0.0, 0.0]),
        'r_final':   np.array([2.0, 2.0, 0.0]),
        'obstacles': OBSTACLES,
    },
    # Agent 1
    {
        'r_init':    np.array([2.0, 0.0, 0.0]),
        'r_final':   np.array([0.0, 2.0, 0.0]),
        'obstacles': OBSTACLES,
    },
    # Agent 2: start below and end above obstacle with extra margin
    {
        'r_init': np.array([
            OBSTACLES[0][0][0],  # x = obstacle center x
            OBSTACLES[0][0][1] - (OBSTACLES[0][1] + CLEARANCE + MARGIN),
            0.0
        ]),
        'r_final': np.array([
            OBSTACLES[0][0][0],  # x = obstacle center x
            OBSTACLES[0][0][1] + (OBSTACLES[0][1] + CLEARANCE + MARGIN),
            0.0
        ]),
        'obstacles': OBSTACLES,
    },
]
