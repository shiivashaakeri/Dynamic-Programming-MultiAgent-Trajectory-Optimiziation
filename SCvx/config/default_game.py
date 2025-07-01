"""
Default scenario for Nash-game demo (non-cooperative agents)
"""

import numpy as np

# ---------- Global parameters ----------
K = 50  # discretisation points
D_MIN = 0.5  # desired inter-agent buffer (used only by plots)
CLEARANCE = 0.05  # inflation band for initial guess / obstacle
MARGIN = 0.6  # extra start/goal offset

# ---------- Static obstacles ----------
OBSTACLES = [
    ([1.0, 1.0], 0.25),  # main disk at (1,1)
    ([1.0, -0.3], 0.02)
]
    # ([1.0, -0.3], 0.02)
    # ([1.0, 0.0], 0.02)  # main disk at (1,1)
# ---------- Game-specific cost weights ----------
CTRL_W = 5  # effort term
COLL_W = 10.0  # weight on slack variables
COLL_RAD = 0.5  # d_min inside model cost

CTRL_RATE_W     = 5.0   #  << increase
CURVATURE_W     = 100.0   #  << increase
# ---------- Agent definitions ----------
AGENT_PARAMS = [
    # Agent 0
    {
        "r_init": np.array([0.0, -1.0, 0.0]),
        "r_final": np.array([2.0, 3.0, 0.0]),
        "obstacles": OBSTACLES,
        "control_weight": CTRL_W,
        "collision_weight": COLL_W,
        "collision_radius": COLL_RAD,
        "control_rate_weight": CTRL_RATE_W,
        "curvature_weight": CURVATURE_W,
    },
    # Agent 1
    {
        "r_init": np.array([2.0, -1.0, 0.0]),
        "r_final": np.array([0.0, 3.0, 0.0]),
        "obstacles": OBSTACLES,
        "control_weight": CTRL_W,
        "collision_weight": COLL_W,
        "collision_radius": COLL_RAD,
        "control_rate_weight": CTRL_RATE_W,
        "curvature_weight": CURVATURE_W,
    },
    # Agent 2 (passes bottom→top of obstacle)
    {
        "r_init": np.array([1.0, -1.5, 0.0]),
        "r_final": np.array([OBSTACLES[0][0][0], 3, 0.0]),
        "obstacles": OBSTACLES,
        "control_weight": CTRL_W,
        "collision_weight": COLL_W,
        "collision_radius": COLL_RAD,
        "control_rate_weight": CTRL_RATE_W,
        "curvature_weight": CURVATURE_W,
    },
]
