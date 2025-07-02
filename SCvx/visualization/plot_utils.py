# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib import animation
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# _u, _v = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]


# def plot_trajectory(X: np.ndarray, model, ax=None):
#     """
#     Plot the 2D trajectory of the unicycle and obstacles.

#     Args:
#         X: State trajectory array of shape (n_x, K)
#         model: UnicycleModel instance with obstacles and robot_radius
#         ax: Matplotlib Axes (optional)
#     Returns:
#         ax: Matplotlib Axes
#     """
#     if ax is None:
#         fig, ax = plt.subplots()
#     # Plot trajectory
#     ax.plot(X[0, :], X[1, :], "-o", label="Trajectory")
#     # Plot start and goal
#     ax.plot(X[0, 0], X[1, 0], "gs", label="Start")
#     ax.plot(X[0, -1], X[1, -1], "r*", label="Goal")
#     # Plot obstacles
#     for p, r in model.obstacles:
#         circle = plt.Circle(p, r + model.robot_radius, color="r", fill=False, linestyle="--", linewidth=1.5)
#         ax.add_patch(circle)
#     ax.set_aspect("equal", "box")
#     ax.set_xlabel("X")
#     ax.set_ylabel("Y")
#     ax.set_title("Planned Trajectory")
#     ax.legend()
#     ax.grid(True)
#     return ax


# def plot_convergence(history: list, ax=None):
#     """
#     Plot convergence metrics (defect and slack norms) over iterations.

#     Args:
#         history: List of dicts with keys 'iter', 'nu_norm', 'slack_norm'
#         ax: Matplotlib Axes (optional)
#     Returns:
#         ax: Matplotlib Axes
#     """
#     if ax is None:
#         fig, ax = plt.subplots()
#     iters = [h["iter"] for h in history]
#     nu_norms = [h["nu_norm"] for h in history]
#     slack_norms = [h["slack_norm"] for h in history]
#     ax.plot(iters, nu_norms, "-o", label="||nu||1")
#     ax.plot(iters, slack_norms, "-x", label="slack sum")
#     ax.set_yscale("log")
#     ax.set_xlabel("Iteration")
#     ax.set_ylabel("Norm (log scale)")
#     ax.set_title("Convergence Metrics")
#     ax.legend()
#     ax.grid(True)
#     return ax


# def animate_trajectory(X: np.ndarray, model, interval=200):
#     """
#     Create an animation of the trajectory.

#     Args:
#         X: State trajectory array of shape (n_x, K)
#         model: UnicycleModel instance
#         interval: Delay between frames in milliseconds
#     Returns:
#         anim: FuncAnimation object
#     """
#     import matplotlib.pyplot as plt
#     from matplotlib import animation

#     fig, ax = plt.subplots()
#     # Plot obstacles once
#     for p, r in model.obstacles:
#         circle = plt.Circle(p, r + model.robot_radius, color="r", fill=False, linestyle="--", linewidth=1.5)
#         ax.add_patch(circle)
#     ax.set_aspect("equal", "box")
#     ax.set_xlim(np.min(X[0, :]) - 1, np.max(X[0, :]) + 1)
#     ax.set_ylim(np.min(X[1, :]) - 1, np.max(X[1, :]) + 1)
#     (line,) = ax.plot([], [], "b-o")

#     def init():
#         line.set_data([], [])
#         return (line,)

#     def update(frame):
#         line.set_data(X[0, :frame], X[1, :frame])
#         return (line,)

#     anim = animation.FuncAnimation(fig, update, init_func=init, frames=X.shape[1] + 1, interval=interval, blit=True)
#     return anim


# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib import animation
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# # --- Constants and Setup ---

# # Earthy muted color palette
# COLOR_PATH = "black"
# COLOR_STATIC_PATH = (0.5, 0.5, 0.5, 0.3)
# COLOR_START = "#8B8C47"
# COLOR_GOAL = "#CC4E5C"
# COLOR_ROBOT = "#5F9EA0"
# COLOR_BLADE = "black"  # Blades will be black
# COLOR_MOTOR_EDGE = "black"
# COLOR_OBSTACLE = "#8B4513"
# COLOR_CENTER_CYLINDER = "black"
# COLOR_STATIC_MARKER = (0.4, 0.4, 0.4, 0.9)  # Darker gray for start/goal markers
# COLOR_ANIMATED_PATH = "#5F9EA0"  # Muted blue for the traversed path

# # Parametric sphere grid for obstacles
# _u2, _v2 = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]

# def plot_3d_trajectory(X, model, ax, color=None, label=None):
#     """
#     Plots a single 3D trajectory, its start/end points, and its obstacles.
#     The axis limits are NOT set here.
#     """
#     # Plot the trajectory path
#     ax.plot(X[0, :], X[1, :], X[2, :], color=color, linewidth=2, label=label)

#     # Plot start and goal points
#     ax.scatter(X[0, 0], X[1, 0], X[2, 0], color=color, marker="o", s=50)
#     ax.scatter(X[0, -1], X[1, -1], X[2, -1], color=color, marker="*", s=100)

#     # Plot the obstacles as wireframe spheres
#     # Note: This will redraw obstacles for each agent. This is fine but can be optimized
#     # by plotting them only once outside the loop if all agents share the same obstacles.
#     u = np.linspace(0, 2 * np.pi, 20)
#     v = np.linspace(0, np.pi, 10)
#     for center, radius in model.obstacles:
#         x_obs = center[0] + radius * np.outer(np.cos(u), np.sin(v))
#         y_obs = center[1] + radius * np.outer(np.sin(u), np.sin(v))
#         z_obs = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
#         ax.plot_wireframe(x_obs, y_obs, z_obs, color='gray', alpha=0.3)

#     return ax


# # --- Core Functions (build_local_quad_3d is unchanged) ---


# def build_local_quad_3d(model):
#     """
#     Builds the 3D quadrotor mesh template.
#     """
#     L = model.robot_radius * 1.5
#     w = model.robot_radius * 0.2
#     h = model.robot_radius * 0.2
#     rv = model.robot_radius * 0.6
#     blade_width = rv * 0.1
#     verts, faces, blade_face_indices = [], [], []

#     # Arms
#     for dx, dy in [(L, 0), (-L, 0), (0, L), (0, -L)]:
#         cx, cy = dx / 2, dy / 2
#         wx = w if dy == 0 else abs(dx)
#         wy = w if dx == 0 else abs(dy)
#         idx = len(verts)
#         for sx in [-wx / 2, wx / 2]:
#             for sy in [-wy / 2, wy / 2]:
#                 for sz in [-h / 2, h / 2]:
#                     verts.append((cx + sx, cy + sy, sz))
#         i0, i1, i2, i3 = idx, idx + 1, idx + 2, idx + 3
#         i4, i5, i6, i7 = idx + 4, idx + 5, idx + 6, idx + 7
#         faces += [[i0, i1, i2], [i0, i2, i3], [i4, i7, i6], [i4, i6, i5]]
#         for a, b, c, d in [(i0, i4, i5, i1), (i1, i5, i6, i2), (i2, i6, i7, i3), (i3, i7, i4, 0)]:
#             faces += [[a, b, c], [a, c, d]]

#     # Rotors and Blades
#     SEG = 12
#     for dx, dy in [(L, 0), (-L, 0), (0, L), (0, -L)]:
#         cx, cy = dx, dy
#         thetas = np.linspace(0, 2 * np.pi, SEG, endpoint=False)

#         # FIX IS HERE: Use np.full_like to match array dimensions
#         top = np.column_stack([rv * np.cos(thetas), rv * np.sin(thetas), np.full_like(thetas, h / 2)])
#         bot = np.column_stack([rv * np.cos(thetas), rv * np.sin(thetas), np.full_like(thetas, -h / 2)])

#         base = len(verts)
#         for v in np.vstack([top, bot]):
#             verts.append((v[0] + cx, v[1] + cy, v[2]))
#         for i in range(SEG):
#             ni = (i + 1) % SEG
#             faces.append([base + i, base + ni, base + SEG + ni, base + SEG + i])
#         faces.append(list(range(base, base + SEG)))
#         faces.append(list(range(base + SEG, base + 2 * SEG)))
#         for angle in [0, np.pi / 2]:
#             bidx = len(verts)
#             d = np.array([np.cos(angle), np.sin(angle)]) * rv
#             p = np.array([-np.sin(angle), np.cos(angle)]) * blade_width
#             v0, v1 = np.array([cx, cy]) - d + p, np.array([cx, cy]) - d - p
#             v2, v3 = np.array([cx, cy]) + d - p, np.array([cx, cy]) + d + p
#             blade_verts = [(v0[0], v0[1], 0.0), (v1[0], v1[1], 0.0), (v2[0], v2[1], 0.0), (v3[0], v3[1], 0.0)]
#             for bv in blade_verts:
#                 verts.append(bv)
#             face_idx_start = len(faces)
#             faces += [[bidx, bidx + 1, bidx + 2], [bidx, bidx + 2, bidx + 3]]
#             blade_face_indices.extend([face_idx_start, face_idx_start + 1])

#     return np.array(verts), faces, blade_face_indices


# def animate_3d_trajectory(X, model, interval=50):
#     """
#     Animates the 3D trajectory with the requested visual style.
#     """
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection="3d")

#     ax.plot(X[0], X[1], X[2], "-", color=COLOR_STATIC_PATH, linewidth=2, zorder=1)

#     for c, r in model.obstacles:
#         xs = c[0] + r * np.cos(_u2) * np.sin(_v2)
#         ys = c[1] + r * np.sin(_u2) * np.sin(_v2)
#         zs = c[2] + r * np.cos(_v2)
#         ax.plot_wireframe(xs, ys, zs, color=COLOR_OBSTACLE, alpha=0.2, zorder=1)

#     m = 1.5
#     ax.set_xlim(X[0].min() - m, X[0].max() + m)
#     ax.set_ylim(X[1].min() - m, X[1].max() + m)
#     ax.set_zlim(X[2].min() - m, X[2].max() + m)
#     ax.set_title("Quadrotor Path Following")

#     verts, faces, blade_face_indices = build_local_quad_3d(model)
#     rotor_face_map = {face_idx: i // 4 for i, face_idx in enumerate(blade_face_indices)}

#     (animated_line,) = ax.plot([], [], [], "-", color=COLOR_ANIMATED_PATH, linewidth=2, zorder=5)

#     def make_quad(frame):
#         if frame < X.shape[1] - 1:
#             dy = X[1, frame + 1] - X[1, frame]
#             dx = X[0, frame + 1] - X[0, frame]
#             psi = np.arctan2(dy, dx)
#         else:
#             psi = np.arctan2(X[1, -1] - X[1, -2], X[0, -1] - X[0, -2])
#         R_yaw = np.array([[np.cos(psi), -np.sin(psi), 0], [np.sin(psi), np.cos(psi), 0], [0, 0, 1]])
#         world_verts = (R_yaw @ verts.T).T + X[:, frame]

#         L = model.robot_radius * 1.5
#         rotor_offsets = np.array([[L, 0, 0], [-L, 0, 0], [0, L, 0], [0, -L, 0]])
#         world_rotor_centers = (R_yaw @ rotor_offsets.T).T + X[:, frame]
#         spin_angle = frame * 0.8
#         c_spin, s_spin = np.cos(spin_angle), np.sin(spin_angle)
#         R_spin = np.array([[c_spin, -s_spin, 0], [s_spin, c_spin, 0], [0, 0, 1]])
#         spun_verts = world_verts.copy()
#         for face_idx in blade_face_indices:
#             rotor_idx = rotor_face_map[face_idx]
#             center = world_rotor_centers[rotor_idx]
#             for v_idx in faces[face_idx]:
#                 p_local = world_verts[v_idx] - center
#                 p_spun = R_spin @ p_local
#                 spun_verts[v_idx] = p_spun + center

#         facecolors = [COLOR_BLADE if i in blade_face_indices else "white" for i, f in enumerate(faces)]

#         return Poly3DCollection(
#             [[spun_verts[i] for i in f] for f in faces],
#             facecolors=facecolors,
#             edgecolors=COLOR_MOTOR_EDGE,
#             linewidths=0.5,
#             alpha=0.95,
#         )

#     def init():
#         animated_line.set_data([], [])
#         animated_line.set_3d_properties([])
#         return [animated_line]

#     def update(frame):
#         for collection in ax.collections[:]:
#             collection.remove()

#         for c, r in model.obstacles:
#             xs = c[0] + r * np.cos(_u2) * np.sin(_v2)
#             ys = c[1] + r * np.sin(_u2) * np.sin(_v2)
#             zs = c[2] + r * np.cos(_v2)
#             ax.plot_wireframe(xs, ys, zs, color=COLOR_OBSTACLE, alpha=0.2, zorder=1)

#         ax.scatter([X[0, 0]], [X[1, 0]], [X[2, 0]], marker="o", color=COLOR_STATIC_MARKER, s=50, zorder=2)
#         ax.scatter([X[0, -1]], [X[1, -1]], [X[2, -1]], marker="X", color=COLOR_STATIC_MARKER, s=75, zorder=2)

#         quad_collection = make_quad(frame)
#         ax.add_collection3d(quad_collection)
#         quad_collection.set_zorder(10)

#         center = X[:, frame]

#         # *** FIX IS HERE: Cylinder radius slightly reduced to prevent clipping ***
#         cyl_r, cyl_h = model.robot_radius * 0.45, model.robot_radius * 0.3

#         SEG, thetas = 20, np.linspace(0, 2 * np.pi, 20, endpoint=False)
#         verts_c, faces_c = [], []
#         for th in thetas:
#             verts_c.append((center[0] + cyl_r * np.cos(th), center[1] + cyl_r * np.sin(th), center[2] - cyl_h / 2))
#         for th in thetas:
#             verts_c.append((center[0] + cyl_r * np.cos(th), center[1] + cyl_r * np.sin(th), center[2] + cyl_h / 2))
#         for i in range(SEG):
#             faces_c.append([i, (i + 1) % SEG, (i + 1) % SEG + SEG, i + SEG])
#         faces_c.append(list(range(SEG)))
#         faces_c.append(list(range(SEG, 2 * SEG)))
#         central_cylinder = Poly3DCollection(
#             [[verts_c[idx] for idx in face] for face in faces_c],
#             facecolors=COLOR_CENTER_CYLINDER,
#             edgecolors="none",
#             zorder=11,
#         )
#         ax.add_collection3d(central_cylinder)

#         animated_line.set_data(X[0, : frame + 1], X[1, : frame + 1])
#         animated_line.set_3d_properties(X[2, : frame + 1])

#         return [animated_line, quad_collection, central_cylinder]

#     anim = animation.FuncAnimation(fig, update, init_func=init, frames=X.shape[1], interval=interval, blit=False)
#     ax.set_xlabel("X")
#     ax.set_ylabel("Y")
#     ax.set_zlabel("Z")
#     plt.tight_layout()
#     return anim

# # ---------------------------------------------------------------------
# # Multi-agent 3-D animation with spinning blades
# # ---------------------------------------------------------------------
# from itertools import cycle
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# import numpy as np


# def animate_multiagent_trajectories_3d(X_list, models, interval=200):
#     """
#     Animate several 3-D single-integrator agents, each drawn with the same
#     quadrotor template (spinning blades) and coloured distinctly.

#     Parameters
#     ----------
#     X_list : list[np.ndarray]   (3,K) per agent
#     models  : list              models[i].obstacles & .robot_radius used
#     interval : int              ms between frames

#     Returns
#     -------
#     anim : matplotlib.animation.FuncAnimation
#     """
#     n_agents = len(X_list)
#     K = X_list[0].shape[1]
#     colors = cycle(plt.cm.tab10(np.linspace(0, 1, 10)))

#     # ---------- figure / axes ----------
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection="3d")

#     # global extents
#     all_xyz = np.hstack([X for X in X_list])
#     margin = 1.0
#     ax.set_xlim(all_xyz[0].min()-margin, all_xyz[0].max()+margin)
#     ax.set_ylim(all_xyz[1].min()-margin, all_xyz[1].max()+margin)
#     ax.set_zlim(all_xyz[2].min()-margin, all_xyz[2].max()+margin)
#     ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
#     ax.set_title("Multi-Agent 3-D trajectories")

#     # obstacles (assume all agents share the same list)
#     _u, _v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
#     for c, r in models[0].obstacles:
#         xs = c[0] + r*np.cos(_u)*np.sin(_v)
#         ys = c[1] + r*np.sin(_u)*np.sin(_v)
#         zs = c[2] + r*np.cos(_v)
#         ax.plot_wireframe(xs, ys, zs, color="gray", alpha=0.3, linewidth=0.5)

#     # ---------- per-agent graphics ----------
#     # Pre-compute one quad template per distinct robot radius
#     from SCvx.visualization.multi_agent_plot_utils import build_local_quad_3d
#     quad_cache = {}
#     for m in models:
#         if m.robot_radius not in quad_cache:
#             quad_cache[m.robot_radius] = build_local_quad_3d(m)

#     # one line + quad collection per agent
#     lines = []
#     quads = []
#     templates = []     # (verts, faces, blade_face_idx)
#     for i, (X, m) in enumerate(zip(X_list, models)):
#         color = next(colors)
#         ln, = ax.plot([], [], [], '-', color=color, linewidth=2)
#         lines.append(ln)
#         verts, faces, blade_faces = quad_cache[m.robot_radius]
#         templates.append((verts, faces, blade_faces))
#         coll = Poly3DCollection([])  # dummy, will be filled in update()
#         quads.append(coll); ax.add_collection3d(coll)

#     # ---------- frame update ----------
#     def update(frame):
#         spin_angle = frame * 0.8
#         c_spin, s_spin = np.cos(spin_angle), np.sin(spin_angle)
#         R_spin = np.array([[c_spin, -s_spin, 0],
#                            [s_spin,  c_spin, 0],
#                            [0,      0,       1]])

#         for ln, coll, (verts, faces, blade_faces), X, m in zip(
#                 lines, quads, templates, X_list, models):

#             # update path
#             ln.set_data(X[0, :frame+1], X[1, :frame+1])
#             ln.set_3d_properties(X[2, :frame+1])

#             # pose of body (yaw only, like single-agent version)
#             if frame < K-1:
#                 dy = X[1, frame+1]-X[1, frame]
#                 dx = X[0, frame+1]-X[0, frame]
#                 psi = np.arctan2(dy, dx)
#             else:
#                 psi = np.arctan2(X[1,-1]-X[1,-2], X[0,-1]-X[0,-2])
#             R_yaw = np.array([[np.cos(psi), -np.sin(psi), 0],
#                               [np.sin(psi),  np.cos(psi), 0],
#                               [0, 0, 1]])
#             wv = (R_yaw @ verts.T).T + X[:, frame]

#             # spin blades
#             for f_idx in blade_faces:
#                 for vid in faces[f_idx]:
#                     local = wv[vid] - X[:, frame]
#                     wv[vid] = X[:, frame] + R_spin @ local

#             facecolors = [
#                 COLOR_BLADE if i in blade_faces else
#                 "white"     if len(f) > 3       else
#                 COLOR_ROBOT
#                 for i, f in enumerate(faces)
#             ]
#             coll.set_verts([[wv[i] for i in f] for f in faces])
#             coll.set_facecolors(facecolors)
#             coll.set_edgecolors(COLOR_MOTOR_EDGE)
#             coll.set_alpha(0.95)

#         return lines + quads

#     anim = FuncAnimation(fig, update, frames=K, interval=interval, blit=False)
#     return anim

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle


def plot_trajectory_2d(X: np.ndarray, model, ax=None):
    """
    Plots the 2D trajectory of a unicycle-style agent and its obstacles.
    This function is for 2D plots only.

    Args:
        X: State trajectory array of shape (n_x, K).
        model: A model instance with 'obstacles' and 'robot_radius' attributes.
        ax: An existing Matplotlib Axes object (optional).
    """
    if ax is None:
        fig, ax = plt.subplots()

    # Plot trajectory path, start, and goal markers
    ax.plot(X[0, :], X[1, :], "-o", label="Trajectory")
    ax.plot(X[0, 0], X[1, 0], "gs", markersize=8, label="Start")
    ax.plot(X[0, -1], X[1, -1], "r*", markersize=10, label="Goal")

    # Plot obstacles as 2D circles
    for p, r in model.obstacles:
        obstacle_circle = Circle(p, r + model.robot_radius, color="red", fill=False, linestyle="--")
        ax.add_patch(obstacle_circle)

    ax.set_aspect("equal", "box")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Planned 2D Trajectory")
    ax.legend()
    ax.grid(True)
    return ax


def plot_trajectory_3d(X: np.ndarray, model, ax, color=None, label=""):
    """
    Plots a single 3D trajectory and its spherical obstacles on a given 3D axis.
    This function does NOT set axis limits.

    Args:
        X: State trajectory array of shape (3, K).
        model: A model instance with an 'obstacles' attribute.
        ax: An existing 3D Matplotlib Axes object.
        color: The color for the trajectory line and markers.
        label: The label for the trajectory line.
    """
    # Plot the 3D path
    ax.plot(X[0, :], X[1, :], X[2, :], color=color, linewidth=2, label=label)

    # Plot start and goal markers
    ax.scatter(X[0, 0], X[1, 0], X[2, 0], color=color, marker="o", s=50)
    ax.scatter(X[0, -1], X[1, -1], X[2, -1], color=color, marker="*", s=100)

    # Plot obstacles as 3D wireframe spheres
    _u, _v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    for center, radius in model.obstacles:
        x_obs = center[0] + radius * np.cos(_u) * np.sin(_v)
        y_obs = center[1] + radius * np.sin(_u) * np.sin(_v)
        z_obs = center[2] + radius * np.cos(_v)
        ax.plot_wireframe(x_obs, y_obs, z_obs, color='saddlebrown', alpha=0.3)

    return ax


def plot_convergence(history: list, ax=None):
    """
    Plots convergence metrics over iterations.

    Args:
        history: A list of dicts, where each dict contains convergence data.
        ax: An existing Matplotlib Axes object (optional).
    """
    if ax is None:
        fig, ax = plt.subplots()

    # Example for a history list of simple values (e.g., from Nash solver)
    if history and isinstance(history[0], (int, float)):
        ax.semilogy(history, marker="o")
        ax.set_ylabel("Max Trajectory Change (log scale)")
    # Example for a history list of dicts (e.g., from another solver)
    elif history and isinstance(history[0], dict):
        iters = [h.get("iter", i) for i, h in enumerate(history)]
        nu_norms = [h.get("nu_norm", 0) for h in history]
        slack_norms = [h.get("slack_norm", 0) for h in history]
        ax.plot(iters, nu_norms, "-o", label="||nu||1")
        ax.plot(iters, slack_norms, "-x", label="slack sum")
        ax.set_yscale("log")
        ax.set_ylabel("Norm (log scale)")
        ax.legend()

    ax.set_xlabel("Iteration")
    ax.set_title("Convergence Metrics")
    ax.grid(True)
    return ax

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# --- Constants and Setup ---

# Earthy muted color palette
COLOR_PATH = "black"
COLOR_STATIC_PATH = (0.5, 0.5, 0.5, 0.3)
COLOR_START = "#8B8C47"
COLOR_GOAL = "#CC4E5C"
COLOR_ROBOT = "#5F9EA0"
COLOR_BLADE = "black"  # Blades will be black
COLOR_MOTOR_EDGE = "black"
COLOR_OBSTACLE = "#8B4513"
COLOR_CENTER_CYLINDER = "black"
COLOR_STATIC_MARKER = (0.4, 0.4, 0.4, 0.9)  # Darker gray for start/goal markers
COLOR_ANIMATED_PATH = "#5F9EA0"  # Muted blue for the traversed path

# Parametric sphere grid for obstacles
_u2, _v2 = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]

def plot_3d_trajectory(X, model, ax, color=None, label=None):
    """
    Plots a single 3D trajectory, its start/end points, and its obstacles.
    The axis limits are NOT set here.
    """
    # Plot the trajectory path
    ax.plot(X[0, :], X[1, :], X[2, :], color=color, linewidth=2, label=label)

    # Plot start and goal points
    ax.scatter(X[0, 0], X[1, 0], X[2, 0], color=color, marker="o", s=50)
    ax.scatter(X[0, -1], X[1, -1], X[2, -1], color=color, marker="*", s=100)

    # Plot the obstacles as wireframe spheres
    # Note: This will redraw obstacles for each agent. This is fine but can be optimized
    # by plotting them only once outside the loop if all agents share the same obstacles.
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 10)
    for center, radius in model.obstacles:
        x_obs = center[0] + radius * np.outer(np.cos(u), np.sin(v))
        y_obs = center[1] + radius * np.outer(np.sin(u), np.sin(v))
        z_obs = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_wireframe(x_obs, y_obs, z_obs, color='gray', alpha=0.3)

    return ax


# --- Core Functions (build_local_quad_3d is unchanged) ---


def build_local_quad_3d(model):
    """
    Builds the 3D quadrotor mesh template.
    """
    L = model.robot_radius * 1.5
    w = model.robot_radius * 0.2
    h = model.robot_radius * 0.2
    rv = model.robot_radius * 0.6
    blade_width = rv * 0.1
    verts, faces, blade_face_indices = [], [], []

    # Arms
    for dx, dy in [(L, 0), (-L, 0), (0, L), (0, -L)]:
        cx, cy = dx / 2, dy / 2
        wx = w if dy == 0 else abs(dx)
        wy = w if dx == 0 else abs(dy)
        idx = len(verts)
        for sx in [-wx / 2, wx / 2]:
            for sy in [-wy / 2, wy / 2]:
                for sz in [-h / 2, h / 2]:
                    verts.append((cx + sx, cy + sy, sz))
        i0, i1, i2, i3 = idx, idx + 1, idx + 2, idx + 3
        i4, i5, i6, i7 = idx + 4, idx + 5, idx + 6, idx + 7
        faces += [[i0, i1, i2], [i0, i2, i3], [i4, i7, i6], [i4, i6, i5]]
        for a, b, c, d in [(i0, i4, i5, i1), (i1, i5, i6, i2), (i2, i6, i7, i3), (i3, i7, i4, 0)]:
            faces += [[a, b, c], [a, c, d]]

    # Rotors and Blades
    SEG = 12
    for dx, dy in [(L, 0), (-L, 0), (0, L), (0, -L)]:
        cx, cy = dx, dy
        thetas = np.linspace(0, 2 * np.pi, SEG, endpoint=False)

        # FIX IS HERE: Use np.full_like to match array dimensions
        top = np.column_stack([rv * np.cos(thetas), rv * np.sin(thetas), np.full_like(thetas, h / 2)])
        bot = np.column_stack([rv * np.cos(thetas), rv * np.sin(thetas), np.full_like(thetas, -h / 2)])

        base = len(verts)
        for v in np.vstack([top, bot]):
            verts.append((v[0] + cx, v[1] + cy, v[2]))
        for i in range(SEG):
            ni = (i + 1) % SEG
            faces.append([base + i, base + ni, base + SEG + ni, base + SEG + i])
        faces.append(list(range(base, base + SEG)))
        faces.append(list(range(base + SEG, base + 2 * SEG)))
        for angle in [0, np.pi / 2]:
            bidx = len(verts)
            d = np.array([np.cos(angle), np.sin(angle)]) * rv
            p = np.array([-np.sin(angle), np.cos(angle)]) * blade_width
            v0, v1 = np.array([cx, cy]) - d + p, np.array([cx, cy]) - d - p
            v2, v3 = np.array([cx, cy]) + d - p, np.array([cx, cy]) + d + p
            blade_verts = [(v0[0], v0[1], 0.0), (v1[0], v1[1], 0.0), (v2[0], v2[1], 0.0), (v3[0], v3[1], 0.0)]
            for bv in blade_verts:
                verts.append(bv)
            face_idx_start = len(faces)
            faces += [[bidx, bidx + 1, bidx + 2], [bidx, bidx + 2, bidx + 3]]
            blade_face_indices.extend([face_idx_start, face_idx_start + 1])

    return np.array(verts), faces, blade_face_indices


def animate_3d_trajectory(X, model, interval=50):
    """
    Animates the 3D trajectory with the requested visual style.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(X[0], X[1], X[2], "-", color=COLOR_STATIC_PATH, linewidth=2, zorder=1)

    for c, r in model.obstacles:
        xs = c[0] + r * np.cos(_u2) * np.sin(_v2)
        ys = c[1] + r * np.sin(_u2) * np.sin(_v2)
        zs = c[2] + r * np.cos(_v2)
        ax.plot_wireframe(xs, ys, zs, color=COLOR_OBSTACLE, alpha=0.2, zorder=1)

    m = 1.5
    ax.set_xlim(X[0].min() - m, X[0].max() + m)
    ax.set_ylim(X[1].min() - m, X[1].max() + m)
    ax.set_zlim(X[2].min() - m, X[2].max() + m)
    ax.set_title("Quadrotor Path Following")

    verts, faces, blade_face_indices = build_local_quad_3d(model)
    rotor_face_map = {face_idx: i // 4 for i, face_idx in enumerate(blade_face_indices)}

    (animated_line,) = ax.plot([], [], [], "-", color=COLOR_ANIMATED_PATH, linewidth=2, zorder=5)

    def make_quad(frame):
        if frame < X.shape[1] - 1:
            dy = X[1, frame + 1] - X[1, frame]
            dx = X[0, frame + 1] - X[0, frame]
            psi = np.arctan2(dy, dx)
        else:
            psi = np.arctan2(X[1, -1] - X[1, -2], X[0, -1] - X[0, -2])
        R_yaw = np.array([[np.cos(psi), -np.sin(psi), 0], [np.sin(psi), np.cos(psi), 0], [0, 0, 1]])
        world_verts = (R_yaw @ verts.T).T + X[:, frame]

        L = model.robot_radius * 1.5
        rotor_offsets = np.array([[L, 0, 0], [-L, 0, 0], [0, L, 0], [0, -L, 0]])
        world_rotor_centers = (R_yaw @ rotor_offsets.T).T + X[:, frame]
        spin_angle = frame * 0.8
        c_spin, s_spin = np.cos(spin_angle), np.sin(spin_angle)
        R_spin = np.array([[c_spin, -s_spin, 0], [s_spin, c_spin, 0], [0, 0, 1]])
        spun_verts = world_verts.copy()
        for face_idx in blade_face_indices:
            rotor_idx = rotor_face_map[face_idx]
            center = world_rotor_centers[rotor_idx]
            for v_idx in faces[face_idx]:
                p_local = world_verts[v_idx] - center
                p_spun = R_spin @ p_local
                spun_verts[v_idx] = p_spun + center

        facecolors = [COLOR_BLADE if i in blade_face_indices else "white" for i, f in enumerate(faces)]

        return Poly3DCollection(
            [[spun_verts[i] for i in f] for f in faces],
            facecolors=facecolors,
            edgecolors=COLOR_MOTOR_EDGE,
            linewidths=0.5,
            alpha=0.95,
        )

    def init():
        animated_line.set_data([], [])
        animated_line.set_3d_properties([])
        return [animated_line]

    def update(frame):
        for collection in ax.collections[:]:
            collection.remove()

        for c, r in model.obstacles:
            xs = c[0] + r * np.cos(_u2) * np.sin(_v2)
            ys = c[1] + r * np.sin(_u2) * np.sin(_v2)
            zs = c[2] + r * np.cos(_v2)
            ax.plot_wireframe(xs, ys, zs, color=COLOR_OBSTACLE, alpha=0.2, zorder=1)

        ax.scatter([X[0, 0]], [X[1, 0]], [X[2, 0]], marker="o", color=COLOR_STATIC_MARKER, s=50, zorder=2)
        ax.scatter([X[0, -1]], [X[1, -1]], [X[2, -1]], marker="X", color=COLOR_STATIC_MARKER, s=75, zorder=2)

        quad_collection = make_quad(frame)
        ax.add_collection3d(quad_collection)
        quad_collection.set_zorder(10)

        center = X[:, frame]

        # *** FIX IS HERE: Cylinder radius slightly reduced to prevent clipping ***
        cyl_r, cyl_h = model.robot_radius * 0.45, model.robot_radius * 0.3

        SEG, thetas = 20, np.linspace(0, 2 * np.pi, 20, endpoint=False)
        verts_c, faces_c = [], []
        for th in thetas:
            verts_c.append((center[0] + cyl_r * np.cos(th), center[1] + cyl_r * np.sin(th), center[2] - cyl_h / 2))
        for th in thetas:
            verts_c.append((center[0] + cyl_r * np.cos(th), center[1] + cyl_r * np.sin(th), center[2] + cyl_h / 2))
        for i in range(SEG):
            faces_c.append([i, (i + 1) % SEG, (i + 1) % SEG + SEG, i + SEG])
        faces_c.append(list(range(SEG)))
        faces_c.append(list(range(SEG, 2 * SEG)))
        central_cylinder = Poly3DCollection(
            [[verts_c[idx] for idx in face] for face in faces_c],
            facecolors=COLOR_CENTER_CYLINDER,
            edgecolors="none",
            zorder=11,
        )
        ax.add_collection3d(central_cylinder)

        animated_line.set_data(X[0, : frame + 1], X[1, : frame + 1])
        animated_line.set_3d_properties(X[2, : frame + 1])

        return [animated_line, quad_collection, central_cylinder]

    anim = animation.FuncAnimation(fig, update, init_func=init, frames=X.shape[1], interval=interval, blit=False)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.tight_layout()
    return anim