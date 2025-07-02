# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.animation import FuncAnimation
# from mpl_toolkits.mplot3d import Axes3D  # Required for 3D projection
# from matplotlib.patches import Circle

# def plot_admm_convergence(primal_hist, dual_hist, ax=None):
#     """
#     Plots the primal and dual residuals to show ADMM convergence.
#     """
#     if ax is None:
#         # Create a new figure and axes if none are provided
#         fig, ax = plt.subplots()

#     iters = np.arange(1, len(primal_hist) + 1)
#     ax.plot(iters, primal_hist, marker="o", label="Primal Residual")
#     ax.plot(iters, dual_hist, marker="x", label="Dual Residual")
#     ax.set_xlabel("ADMM Iteration")
#     ax.set_ylabel("Average Residual")
#     ax.set_title("ADMM Convergence")
#     ax.grid(True)
#     ax.legend()
#     return ax


# def animate_trajectories_3d(trajectory_list, agent_models, interval=150):
#     """
#     Animates the 3D trajectories for multiple agents, including spherical obstacles.

#     Args:
#         trajectory_list (list): A list of numpy arrays, where each array is an agent's
#                                 state trajectory with shape (n_states, K_timesteps).
#         agent_models (list): A list of agent model objects, used to get obstacle data.
#         interval (int): Delay between frames in milliseconds.
#     """
#     num_agents = len(trajectory_list)
#     if num_agents == 0:
#         print("Warning: No trajectories to animate.")
#         return None

#     K_timesteps = trajectory_list[0].shape[1]

#     # --- Create Figure and 3D Axes ---
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection="3d")
#     ax.set_title("Multi-Agent 3D Trajectory Animation")
#     ax.set_xlabel("X coordinate")
#     ax.set_ylabel("Y coordinate")
#     ax.set_zlabel("Z coordinate")

#     # --- Draw Static Obstacles ---
#     # To draw a sphere in 3D, we create a wireframe. We can't use 2D 'Circle' patches.

#     obstacles = getattr(agent_models[0], "obstacles", [])
#     u = np.linspace(0, 2 * np.pi, 20)
#     v = np.linspace(0, np.pi, 10)
#     for center, radius in obstacles:
#         x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
#         y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
#         z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
#         ax.plot_wireframe(x, y, z, color="gray", alpha=0.4, linewidth=0.8)

#     # --- Prepare Agent Trajectory Lines ---
#     # Create a line object for each agent that will be updated in the animation
#     colors = plt.cm.jet(np.linspace(0, 1, num_agents))
#     lines = [ax.plot([], [], [], "o-", color=c, label=f"Agent {i}")[0] for i, c in enumerate(colors)]
#     ax.legend()

#     # --- Set Axis Limits ---
#     # Calculate bounds to make sure the whole trajectory and obstacles are visible
#     all_x = np.concatenate([X[0] for X in trajectory_list])
#     all_y = np.concatenate([X[1] for X in trajectory_list])
#     all_z = np.concatenate([X[2] for X in trajectory_list])
#     ax.set_xlim(all_x.min() - 1, all_x.max() + 1)
#     ax.set_ylim(all_y.min() - 1, all_y.max() + 1)
#     ax.set_zlim(all_z.min() - 1, all_z.max() + 1)

#     # --- Animation Functions ---
#     def init():
#         """Initializes the animation by drawing empty lines."""
#         for line in lines:
#             line.set_data([], [])
#             line.set_3d_properties([])
#         return lines

#     def update(frame):
#         """Updates the plot for each frame of the animation."""
#         for line, trajectory in zip(lines, trajectory_list):
#             # Update the data for each agent's line up to the current frame
#             line.set_data(trajectory[0, : frame + 1], trajectory[1, : frame + 1])
#             line.set_3d_properties(trajectory[2, : frame + 1])
#         return lines

#     # --- Create and Return Animation ---
#     anim = FuncAnimation(fig, update, frames=K_timesteps, init_func=init, blit=False, interval=interval)
#     return anim

# def animate_multi_agents_3d(X_list, models, interval=200):
#     """
#     Animate multiple single-integrator agents in 3-D plus spherical obstacles.

#     Args
#     ----
#     X_list : list[np.ndarray]
#         State trajectories, each (3,K).
#     models : list
#         Corresponding models (for obstacles / robot radius).
#     interval : int
#         Delay between frames in ms.

#     Returns
#     -------
#     anim : matplotlib.animation.FuncAnimation
#     """
#     num_agents = len(X_list)
#     K = X_list[0].shape[1]

#     # extents (include obstacle radii)
#     xs = np.concatenate([X[0] for X in X_list])
#     ys = np.concatenate([X[1] for X in X_list])
#     zs = np.concatenate([X[2] for X in X_list])
#     obstacles = getattr(models[0], "obstacles", [])
#     for c, r in obstacles:
#         xs = np.hstack([xs, c[0] + np.array([-r, r])])
#         ys = np.hstack([ys, c[1] + np.array([-r, r])])
#         zs = np.hstack([zs, c[2] + np.array([-r, r])])

#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection="3d")
#     margin = 0.1
#     ax.set_xlim(xs.min() - margin, xs.max() + margin)
#     ax.set_ylim(ys.min() - margin, ys.max() + margin)
#     ax.set_zlim(zs.min() - margin, zs.max() + margin)
#     ax.set_xlabel("X")
#     ax.set_ylabel("Y")
#     ax.set_zlabel("Z")
#     ax.set_title("Multi-Agent 3-D Trajectories")

#     # obstacles
#     _u, _v = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]
#     for c, r in obstacles:
#         x_s = c[0] + r * np.cos(_u) * np.sin(_v)
#         y_s = c[1] + r * np.sin(_u) * np.sin(_v)
#         z_s = c[2] + r * np.cos(_v)
#         ax.plot_wireframe(
#             x_s,
#             y_s,
#             z_s,
#             color="gray",
#             alpha=0.3,
#             linewidth=0.5,
#             zorder=1,
#         )

#     # agent lines
#     colors = plt.cm.tab10(np.linspace(0, 1, num_agents))
#     lines = [
#         ax.plot([], [], [], "o-", color=colors[i], label=f"agent {i}")[0]
#         for i in range(num_agents)
#     ]
#     ax.legend(loc="upper left")

#     def init():
#         for ln in lines:
#             ln.set_data([], [])
#             ln.set_3d_properties([])
#         return lines

#     def update(frame):
#         for ln, X in zip(lines, X_list):
#             ln.set_data(X[0, : frame + 1], X[1, : frame + 1])
#             ln.set_3d_properties(X[2, : frame + 1])
#         return lines

#     anim = FuncAnimation(
#         fig,
#         update,
#         frames=K,
#         init_func=init,
#         blit=False,
#         interval=interval,
#     )
#     return anim


import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def plot_admm_convergence(primal_hist, dual_hist, ax=None):
    """
    Plots the primal and dual residuals from an ADMM solver.

    Args:
        primal_hist (list): A list of the primal residual values per iteration.
        dual_hist (list): A list of the dual residual values per iteration.
        ax: An existing Matplotlib Axes object (optional).
    """
    if ax is None:
        fig, ax = plt.subplots()

    iters = np.arange(1, len(primal_hist) + 1)
    ax.plot(iters, primal_hist, marker="o", label="Primal Residual")
    ax.plot(iters, dual_hist, marker="x", label="Dual Residual")
    ax.set_xlabel("ADMM Iteration")
    ax.set_ylabel("Average Residual")
    ax.set_title("ADMM Convergence")
    ax.grid(True)
    ax.legend()
    return ax


def animate_trajectories_3d(trajectory_list, agent_models, interval=150):
    """
    Animates multiple agent trajectories as lines in a 3D space with obstacles.

    Args:
        trajectory_list (list): A list of state trajectory arrays, each (3, K).
        agent_models (list): A list of agent model objects for obstacle data.
        interval (int): Delay between frames in milliseconds.
    """
    num_agents = len(trajectory_list)
    if num_agents == 0:
        print("Warning: No trajectories to animate.")
        return None

    K_timesteps = trajectory_list[0].shape[1]

    # --- 1. Setup the Figure and 3D Axes ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Multi-Agent 3D Trajectory Animation")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_zlabel("Z Coordinate")

    # --- 2. Draw Static Obstacles ---
    if agent_models:
        obstacles = getattr(agent_models[0], "obstacles", [])
        _u, _v = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]
        for center, radius in obstacles:
            x = center[0] + radius * np.outer(np.cos(_u), np.sin(_v))
            y = center[1] + radius * np.outer(np.sin(_u), np.sin(_v))
            z = center[2] + radius * np.outer(np.ones(np.size(_u)), np.cos(_v))
            ax.plot_wireframe(x, y, z, color="saddlebrown", alpha=0.3, linewidth=1.0)

    # --- 3. Set Axis Limits ---
    # Calculate global bounds to ensure all trajectories are visible.
    all_x = np.concatenate([X[0] for X in trajectory_list])
    all_y = np.concatenate([X[1] for X in trajectory_list])
    all_z = np.concatenate([X[2] for X in trajectory_list])
    margin = 1.5
    ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
    ax.set_ylim(all_y.min() - margin, all_y.max() + margin)
    ax.set_zlim(all_z.min() - margin, all_z.max() + margin)

    # --- 4. Prepare Artists for Animation ---
    # Create a line artist for each agent that will be updated every frame.
    colors = plt.cm.viridis(np.linspace(0, 1, num_agents))
    lines = [ax.plot([], [], [], "o-", markersize=4, color=c, label=f"Agent {i}")[0] for i, c in enumerate(colors)]
    ax.legend()

    # --- 5. Define Animation Functions ---
    def init():
        """Initializes the animation by drawing empty lines."""
        for line in lines:
            line.set_data([], [])
            line.set_3d_properties([])
        return lines

    def update(frame):
        """Updates the plot for each frame of the animation."""
        for line, trajectory in zip(lines, trajectory_list):
            # Update the data for each agent's line up to the current frame.
            line.set_data(trajectory[0, : frame + 1], trajectory[1, : frame + 1])
            line.set_3d_properties(trajectory[2, : frame + 1])
        return lines

    # --- 6. Create and Return Animation ---
    anim = FuncAnimation(fig, update, frames=K_timesteps, init_func=init, blit=False, interval=interval)
    return anim


# --- Constants copied from your file ---
COLOR_STATIC_PATH = (0.5, 0.5, 0.5, 0.3)
COLOR_OBSTACLE = "#8B4513"
COLOR_BLADE = "black"
COLOR_MOTOR_EDGE = "black"
COLOR_CENTER_CYLINDER = "black"


# Helper function to build the quadrotor mesh
def _build_scaled_quad_mesh(model):
    """Builds a 3D quadrotor mesh that fits entirely within the given robot_radius."""
    bounding_radius = model.robot_radius
    L = bounding_radius * 0.70
    rv = bounding_radius * 0.30
    w = L * 0.15
    h = L * 0.2
    blade_width = rv * 0.2
    verts, faces, blade_indices = [], [], []

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
        i = np.arange(8) + idx
        faces.extend(
            [[i[0], i[1], i[3], i[2]], [i[4], i[5], i[7], i[6]], [i[0], i[2], i[6], i[4]], [i[1], i[3], i[7], i[5]]]
        )

    # Rotors and Blades
    SEG = 12
    for dx, dy in [(L, 0), (-L, 0), (0, L), (0, -L)]:
        cx, cy = dx, dy
        thetas = np.linspace(0, 2 * np.pi, SEG, endpoint=False)
        top = np.column_stack([rv * np.cos(thetas), rv * np.sin(thetas), np.full_like(thetas, h / 2)])
        bot = np.column_stack([rv * np.cos(thetas), rv * np.sin(thetas), np.full_like(thetas, -h / 2)])
        base = len(verts)
        verts.extend([(v[0] + cx, v[1] + cy, v[2]) for v in np.vstack([top, bot])])
        for i in range(SEG):
            ni = (i + 1) % SEG
            faces.append([base + i, base + ni, base + SEG + ni, base + SEG + i])
        for angle in [0, np.pi / 2]:
            bidx = len(verts)
            d = np.array([np.cos(angle), np.sin(angle)]) * rv
            p = np.array([-np.sin(angle), np.cos(angle)]) * blade_width
            v = [
                tuple(pos)
                for pos in [
                    np.array([cx, cy]) - d + p,
                    np.array([cx, cy]) - d - p,
                    np.array([cx, cy]) + d - p,
                    np.array([cx, cy]) + d + p,
                ]
            ]
            verts.extend(
                [(v[0][0], v[0][1], 0.0), (v[1][0], v[1][1], 0.0), (v[2][0], v[2][1], 0.0), (v[3][0], v[3][1], 0.0)]
            )
            faces.append([bidx, bidx + 1, bidx + 3, bidx + 2])
            blade_indices.append(len(faces) - 1)

    return np.array(verts), faces, blade_indices


# Helper function for the central body
def _build_center_body_mesh(model):
    """Builds the mesh for the central cylindrical body."""
    cyl_r = model.robot_radius * 0.35
    cyl_h = getattr(model, "rotor_height", 0.1) * 1.5
    SEG = 20
    thetas = np.linspace(0, 2 * np.pi, SEG)
    top_v = [np.array([cyl_r * np.cos(th), cyl_r * np.sin(th), cyl_h / 2]) for th in thetas]
    bot_v = [np.array([cyl_r * np.cos(th), cyl_r * np.sin(th), -cyl_h / 2]) for th in thetas]
    verts = top_v + bot_v
    faces = []
    for i in range(SEG):
        faces.append([i, (i + 1) % SEG, (i + 1) % SEG + SEG, i + SEG])
    faces.append(list(range(SEG)))
    faces.append(list(range(SEG, 2 * SEG)))
    return np.array(verts), faces


# --- The Final Multi-Agent Animation Function ---
def animate_multi_agent_quadrotors(X_list, models, interval=50):  # noqa: PLR0915
    """Animates multiple agents as correctly scaled 3D quadrotor models."""
    n_agents = len(X_list)
    K = X_list[0].shape[1]

    # --- 💡 NEW: Use a more muted and earthy color map ---
    agent_colors = plt.cm.cividis(np.linspace(0.1, 0.9, n_agents))

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    all_xyz = np.hstack(X_list)
    margin = 1.5
    ax.set_xlim(all_xyz[0].min() - margin, all_xyz[0].max() + margin)
    ax.set_ylim(all_xyz[1].min() - margin, all_xyz[1].max() + margin)
    ax.set_zlim(all_xyz[2].min() - margin, all_xyz[2].max() + margin)
    ax.set_title("Multi-Agent Quadrotor Simulation")

    _u, _v = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]
    if models:
        for c, r in models[0].obstacles:
            xs = c[0] + r * np.cos(_u) * np.sin(_v)
            ys = c[1] + r * np.sin(_u) * np.sin(_v)
            zs = c[2] + r * np.cos(_v)
            ax.plot_wireframe(xs, ys, zs, color="#8B4513", alpha=0.2, zorder=-1)

    # Use the new colors for the paths and markers
    for i, (X, color) in enumerate(zip(X_list, agent_colors)):
        # Ghost path is still transparent
        ax.plot(X[0, :], X[1, :], X[2, :], "-", color=color, linewidth=1.5, alpha=0.4, label=f"Agent {i}")
        # Start/End markers are more solid
        ax.scatter(X[0, 0], X[1, 0], X[2, 0], color=color, marker="o", s=60, edgecolor="black", alpha=0.9)
        ax.scatter(
            X[0, -1], X[1, -1], X[2, -1], color=color, marker="x", s=90, edgecolor="black", alpha=0.9, linewidths=2
        )

    # Prepare animation artists
    templates = [_build_scaled_quad_mesh(m) for m in models]
    body_templates = [_build_center_body_mesh(m) for m in models]

    # --- 💡 NEW: Set alpha to 1.0 for solid quadrotors ---
    quad_collections = [ax.add_collection3d(Poly3DCollection([], alpha=1.0)) for _ in range(n_agents)]
    body_collections = [ax.add_collection3d(Poly3DCollection([], alpha=1.0, zorder=6)) for _ in range(n_agents)]

    def update(frame):
        for i in range(n_agents):
            coll, body_coll = quad_collections[i], body_collections[i]
            verts, faces, blade_indices = templates[i]
            body_verts, body_faces = body_templates[i]
            X, model = X_list[i], models[i]

            if frame < K - 1:
                psi = np.arctan2(X[1, frame + 1] - X[1, frame], X[0, frame + 1] - X[0, frame])
            else:
                psi = np.arctan2(X[1, -1] - X[1, -2], X[0, -1] - X[0, -2])
            R_yaw = np.array([[np.cos(psi), -np.sin(psi), 0], [np.sin(psi), np.cos(psi), 0], [0, 0, 1]])

            # Blade Spinning Logic
            L = model.robot_radius * 0.70
            rotor_offsets = np.array([[L, 0, 0], [-L, 0, 0], [0, L, 0], [0, -L, 0]])
            world_rotor_centers = (R_yaw @ rotor_offsets.T).T + X[:, frame]
            spin_angle = frame * 0.9
            c, s = np.cos(spin_angle), np.sin(spin_angle)
            R_spin = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

            world_verts = (R_yaw @ verts.T).T + X[:, frame]
            spun_verts = world_verts.copy()

            rotor_map = {face_idx: j // 2 for j, face_idx in enumerate(blade_indices)}
            for face_idx in blade_indices:
                rotor_idx = rotor_map[face_idx]
                center = world_rotor_centers[rotor_idx]
                for v_idx in faces[face_idx]:
                    p_local = world_verts[v_idx] - center
                    p_spun = R_spin @ p_local
                    spun_verts[v_idx] = p_spun + center

            # Set colors and vertices
            agent_color = agent_colors[i]
            coll.set_verts([spun_verts[f] for f in faces])
            coll.set_facecolor(agent_color)
            coll.set_edgecolor("black")

            world_body_verts = (R_yaw @ body_verts.T).T + X[:, frame]
            body_coll.set_verts([world_body_verts[f] for f in body_faces])
            body_coll.set_facecolor(agent_color)
            body_coll.set_edgecolor("black")

        return quad_collections + body_collections

    anim = animation.FuncAnimation(fig, update, frames=K, interval=interval, blit=False)
    ax.legend()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    return anim


# # --- The Final Multi-Agent Animation Function ---
# def animate_multi_agent_quadrotors(X_list, models, interval=50):
#     """Animates multiple agents as correctly scaled 3D quadrotor models."""
#     n_agents = len(X_list); K = X_list[0].shape[1]
#     agent_colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_agents))

#     fig = plt.figure(figsize=(12, 10)); ax = fig.add_subplot(111, projection="3d")

#     all_xyz = np.hstack(X_list)
#     margin = 1.5
#     ax.set_xlim(all_xyz[0].min()-margin, all_xyz[0].max()+margin)
#     ax.set_ylim(all_xyz[1].min()-margin, all_xyz[1].max()+margin)
#     ax.set_zlim(all_xyz[2].min()-margin, all_xyz[2].max()+margin)
#     ax.set_title("Multi-Agent Quadrotor Simulation")

#     _u, _v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
#     if models:
#         for c, r in models[0].obstacles:
#             xs=c[0]+r*np.cos(_u)*np.sin(_v); ys=c[1]+r*np.sin(_u)*np.sin(_v); zs=c[2]+r*np.cos(_v)
#             ax.plot_wireframe(xs, ys, zs, color="#8B4513", alpha=0.2, zorder=-1)

#     for i, (X, color) in enumerate(zip(X_list, agent_colors)):
#         ax.plot(X[0,:], X[1,:], X[2,:], '-', color=color, linewidth=1.5, alpha=0.3)
#         ax.scatter(X[0,0], X[1,0], X[2,0], color=color, marker='o', s=50, edgecolor='black', alpha=0.8)
#         ax.scatter(X[0,-1], X[1,-1], X[2,-1], color=color, marker='x', s=80, edgecolor='black', alpha=0.8)

#     templates = [_build_scaled_quad_mesh(m) for m in models]
#     quad_collections = [ax.add_collection3d(Poly3DCollection([], alpha=0.95)) for _ in range(n_agents)]

#     def update(frame):
#         for i in range(n_agents):
#             coll, (verts, faces, blade_indices), X, model = quad_collections[i], templates[i], X_list[i], models[i]
#             if frame<K-1: psi=np.arctan2(X[1,frame+1]-X[1,frame],X[0,frame+1]-X[0,frame])
#             else: psi=np.arctan2(X[1,-1]-X[1,-2],X[0,-1]-X[0,-2])
#             R_yaw = np.array([[np.cos(psi),-np.sin(psi),0],[np.sin(psi),np.cos(psi),0],[0,0,1]])
#             world_verts = (R_yaw @ verts.T).T + X[:, frame]
#             coll.set_verts([world_verts[f] for f in faces])
#             coll.set_facecolor(agent_colors[i]); coll.set_edgecolor("black")
#         return quad_collections

#     anim = animation.FuncAnimation(fig, update, frames=K, interval=interval, blit=False)
#     ax.legend(); ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
#     return anim


def animate_multi_agent_spheres(X_list, models, interval=50):
    """
    Animates multiple agents as simple spheres to clearly visualize their radius
    and separation distance.
    """
    n_agents = len(X_list)
    if not n_agents:
        return None
    K = X_list[0].shape[1]
    agent_colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_agents))

    # --- 1. Setup Figure and Static Elements ---
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Set up global axis limits and plot obstacles
    all_xyz = np.hstack(X_list)
    margin = 1.5
    ax.set_xlim(all_xyz[0].min() - margin, all_xyz[0].max() + margin)
    ax.set_ylim(all_xyz[1].min() - margin, all_xyz[1].max() + margin)
    ax.set_zlim(all_xyz[2].min() - margin, all_xyz[2].max() + margin)
    ax.set_title("Multi-Agent Sphere Animation")

    _u, _v = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]
    if models:
        for c, r in models[0].obstacles:
            xs = c[0] + r * np.cos(_u) * np.sin(_v)
            ys = c[1] + r * np.sin(_u) * np.sin(_v)
            zs = c[2] + r * np.cos(_v)
            ax.plot_wireframe(xs, ys, zs, color="#8B4513", alpha=0.2)

    # Plot static "ghost" trajectories
    for i, (X, color) in enumerate(zip(X_list, agent_colors)):
        ax.plot(X[0, :], X[1, :], X[2, :], "-", color=color, linewidth=1.5, alpha=0.3)

    # --- 2. Prepare for Animation ---
    dynamic_artists = []  # This list will hold the sphere artists

    def update(frame):
        # The "remove and redraw" method is robust for complex animations
        for artist in dynamic_artists:
            artist.remove()
        dynamic_artists.clear()

        # For the current frame, draw a new sphere for each agent
        for i in range(n_agents):
            center_point = X_list[i][:, frame]
            radius = models[i].robot_radius

            # Generate the (x, y, z) coordinates for the sphere's surface
            x_s = center_point[0] + radius * np.cos(_u) * np.sin(_v)
            y_s = center_point[1] + radius * np.sin(_u) * np.sin(_v)
            z_s = center_point[2] + radius * np.cos(_v)

            # Plot the solid surface of the sphere
            sphere = ax.plot_surface(x_s, y_s, z_s, color=agent_colors[i], alpha=0.8)
            dynamic_artists.append(sphere)

        return dynamic_artists

    anim = animation.FuncAnimation(fig, update, frames=K, interval=interval, blit=False)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    return anim
