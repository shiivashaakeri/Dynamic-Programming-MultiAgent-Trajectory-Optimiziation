"""Single-Integrator 3D scenario for the non-cooperative (Nash-game) demo."""

import numpy as np

# ---------- Global parameters ----------
K = 100  # discretisation points
D_MIN = 0.0  # desired inter-agent buffer (used only for plotting)
CLEARANCE = 0.05  # inflation band for initial guess / obstacle
MARGIN = 1  # extra start/goal offset around obstacles

ROBOT_RADIUS = 0.5  # Each quadrotor has a physical radius of 0.3m
# ---------- Static spherical obstacles ----------
# List of (centre_xyz, radius)
OBSTACLES = [
    ([0.0, 0.0, 0.0], 0.8),  # main sphere at origin (r = 1 m)
]

# ---------- Game-specific cost weights ----------
CTRL_W = 5.0  # effort term ( ‖u‖² )
COLL_W = 80.0  # weight on slack variables
COLL_RAD = 2 * ROBOT_RADIUS + MARGIN  # collision radius (inflated by margin)
CTRL_RATE_W = 5.0
CURVATURE_W = 100.0

# ---------- Agent definitions (3D start/end) ----------
# Paths are arranged to intersect near the central obstacle.
AGENT_PARAMS = [
    # Agent 0 : East → West along +X
    {
        "r_init": np.array([-4.0, 0.0, 0.0]),
        "r_final": np.array([4.0, 0.0, 0.0]),
        "obstacles": OBSTACLES,
        "robot_radius": ROBOT_RADIUS,
        "control_weight": CTRL_W,
        "collision_weight": COLL_W,
        "collision_radius": COLL_RAD,
        "control_rate_weight": CTRL_RATE_W,
        "curvature_weight": CURVATURE_W,
    },
    # Agent 1 : South → North along +Y
    {
        "r_init": np.array([0.0, -4.0, 0.0]),
        "r_final": np.array([0.0, 4.0, 0.0]),
        "obstacles": OBSTACLES,
        "robot_radius": ROBOT_RADIUS,
        "control_weight": CTRL_W,
        "collision_weight": COLL_W,
        "collision_radius": COLL_RAD,
        "control_rate_weight": CTRL_RATE_W,
        "curvature_weight": CURVATURE_W,
    },
    # Agent 2 : Down → Up along +Z (vertical)
    {
        "r_init": np.array([0.0, 0.0, -4.0]),
        "r_final": np.array([0.0, 0.0, 4.0]),
        "obstacles": OBSTACLES,
        "robot_radius": ROBOT_RADIUS,
        "control_weight": CTRL_W,
        "collision_weight": COLL_W,
        "collision_radius": COLL_RAD,
        "control_rate_weight": CTRL_RATE_W,
        "curvature_weight": CURVATURE_W,
    },
]
