from typing import List, Tuple

import numpy as np


def line_sphere_intersect(
    p: np.ndarray, q: np.ndarray, center: np.ndarray, r: float
) -> bool:
    """
    Check if the 3D segment p→q intersects the sphere (center, radius r).
    """
    d = q - p
    f = p - center
    a = d.dot(d)
    b = 2 * f.dot(d)
    c = f.dot(f) - r * r
    disc = b * b - 4 * a * c
    if disc < 0:
        return False
    sqrt_disc = np.sqrt(disc)
    t1 = (-b + sqrt_disc) / (2 * a)
    t2 = (-b - sqrt_disc) / (2 * a)
    return (0 < t1 < 1) or (0 < t2 < 1)


def compute_detour_waypoints(
    p0: np.ndarray,
    p1: np.ndarray,
    center: np.ndarray,
    r: float
) -> List[np.ndarray]:
    """
    Generate two 3D detour waypoints around a sphere:
    - Project the center onto the line p0→p1.
    - Pick a unit vector u perpendicular to the motion direction.
    - Offset the projection by ±(r) along u.

    Returns:
        [T1, T2]: two waypoints in R^3.
    """
    d = p1 - p0
    norm_d = np.linalg.norm(d)
    if norm_d < 1e-6:
        raise ValueError("p0 and p1 are too close for detour computation.")
    d_unit = d / norm_d
    # find arbitrary vector not parallel to d_unit
    tmp = np.array([1.0, 0.0, 0.0]) if abs(d_unit[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(d_unit, tmp)
    u = u / np.linalg.norm(u)
    # projection of center onto line
    t = np.dot(center - p0, d_unit)
    proj = p0 + t * d_unit
    # detour radius = r
    return [proj + r * u, proj - r * u]


def generate_piecewise_linear(
    p0: np.ndarray,
    p1: np.ndarray,
    waypoints: List[np.ndarray],
    K: int
) -> np.ndarray:
    """
    Sample exactly K points along segments p0→w1→...→wN→p1 in R^3.
    Returns array of shape (3, K).
    """
    pts = [p0] + waypoints + [p1]  # noqa: RUF005
    # compute segment lengths
    lengths = [np.linalg.norm(pts[i+1] - pts[i]) for i in range(len(pts)-1)]
    total = sum(lengths)
    # allocate samples per segment (at least 2)
    Ns = [max(2, round(K * L / total)) for L in lengths]
    Ns[-1] = K - sum(Ns[:-1])
    segments = []
    for idx, (start, end) in enumerate(zip(pts[:-1], pts[1:])):
        endpoint = (idx == len(pts)-2)
        seg = np.linspace(start, end, Ns[idx], endpoint=endpoint)
        segments.append(seg)
    # stack and transpose to (3, K)
    path = np.vstack(segments)
    if path.shape[0] != K:
        # fallback: uniform straight-line if mismatch
        path = np.linspace(p0, p1, K)
    return path.T


def initial_guess(
    p0: np.ndarray,
    p1: np.ndarray,
    obstacles: List[Tuple[np.ndarray, float]],
    clearance: float,
    K: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a warm-start (X0, U0) for a 3D single-integrator.

    Args:
        p0: start position (3,)
        p1: goal position (3,)
        obstacles: list of (center (3,), radius) tuples
        clearance: extra margin around each obstacle
        K: number of discretization points

    Returns:
        X0: (3, K) initial position trajectory
        U0: (3, K) initial velocity trajectory
    """
    p0 = p0.astype(float)
    p1 = p1.astype(float)
    # inflate obstacle radii by clearance
    inflated = [(c, r + clearance) for c, r in obstacles]
    # collect detour waypoints
    waypoints = []
    for center, r_inf in inflated:
        if not line_sphere_intersect(p0, p1, center, r_inf):
            continue
        T1, T2 = compute_detour_waypoints(p0, p1, center, r_inf)
        waypoints.extend([T1, T2])
    # build 3D path
    X0 = generate_piecewise_linear(p0, p1, waypoints, K)
    # compute velocity warm-start
    U0 = np.zeros_like(X0)
    dt = 1.0 / (K - 1)
    U0[:, :-1] = (X0[:, 1:] - X0[:, :-1]) / dt
    U0[:, -1] = U0[:, -2]
    return X0, U0
