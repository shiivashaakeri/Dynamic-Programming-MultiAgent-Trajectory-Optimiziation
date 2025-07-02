import numpy as np

# Number of discretization points
K = 100

# Minimum inter-agent separation distance
D_MIN = 0.5

# Clearance for obstacle inflation in the initializer
CLEARANCE = 0.05

# Additional margin to start/stop outside the clearance band
MARGIN = 0.6

# Static spherical obstacles for the scenario: list of (center, radius)
# All agents reference these by default
OBSTACLES = [
    ([0.0, 0.0, 0.0], 1),  # single small obstacle at (1,1,0)
]

# Agent start/end points are now spread out in a range of 8 units
AGENT_PARAMS = [
    # Agent 0: Travels along the x-axis
    {
        'r_init':    np.array([-4.0, 0.0, 0.0]),
        'r_final':   np.array([4.0, 0.0, 0.0]),
        'obstacles': OBSTACLES,
    },
    # Agent 1: Travels along the y-axis, crossing Agent 0's path
    {
        'r_init':    np.array([0.0, -4.0, 0.0]),
        'r_final':   np.array([0.0, 4.0, 0.0]),
        'obstacles': OBSTACLES,
    },
    # Agent 2: Travels along the z-axis, creating a 3D intersection
    {
        'r_init':    np.array([0.0, 0.0, -4.0]),
        'r_final':   np.array([0.0, 0.0, 4.0]),
        'obstacles': OBSTACLES,
    },
]
