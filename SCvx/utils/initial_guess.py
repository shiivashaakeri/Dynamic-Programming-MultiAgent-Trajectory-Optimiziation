import numpy as np


def line_circle_intersect(p, q, center, r):
    """
    Return True if the segment p→q intersects the circle at center with radius r.
    """
    d = q - p
    f = p - center
    a = np.dot(d, d)
    b = 2 * np.dot(f, d)
    c = np.dot(f, f) - r * r
    disc = b * b - 4 * a * c
    if disc < 0:
        return False
    sqrt_disc = np.sqrt(disc)
    t1 = (-b + sqrt_disc) / (2 * a)
    t2 = (-b - sqrt_disc) / (2 * a)
    return (0 < t1 < 1) or (0 < t2 < 1)


def compute_tangent_points(p, center, r):
    """
    Compute the two tangent points from external point p to circle (center, r).
    Raises if p is inside or on the circle.
    """
    v = p - center
    d = np.linalg.norm(v)
    if d <= r:
        raise ValueError("Point inside/on circle; no tangents.")
    alpha = np.arcsin(r / d)
    theta = np.arctan2(v[1], v[0])
    t1 = theta + alpha
    t2 = theta - alpha
    T1 = center + r * np.array([np.cos(t1), np.sin(t1)])
    T2 = center + r * np.array([np.cos(t2), np.sin(t2)])
    return T1, T2


def generate_piecewise_linear(p0, p1, waypoints, K):
    """
    Sample uniformly along segments p0→w1→...→wN→p1 into exactly K points.
    """
    pts = [p0, *list(waypoints), p1]
    lengths = [np.linalg.norm(pts[i + 1] - pts[i]) for i in range(len(pts) - 1)]
    total = sum(lengths)
    # allocate number of samples per segment (at least 2)
    Ns = [max(2, round(K * L / total)) for L in lengths]
    # adjust last segment so total samples exactly K
    # adjust last segment so total samples exactly K
    Ns[-1] = K - sum(Ns[:-1])
    traj_segments = []
    for idx in range(len(pts) - 1):
        endpoint = idx == len(pts) - 2
        seg = np.linspace(pts[idx], pts[idx + 1], Ns[idx], endpoint=endpoint)
        traj_segments.append(seg)
    path = np.vstack(traj_segments).T  # shape 2xK
    return path


def initial_guess(p0, p1, obstacles, clearance, K):
    """
    Construct a warm-start trajectory (X0, U0) that avoids a list of circular obstacles.
    p0, p1: arrays of shape (3,) -> [x, y, theta]
    obstacles: list of ([x, y], radius)
    clearance: extra margin around obstacles
    K: number of discretization points
    Returns X0 (3xK) and U0 (2xK)
    """
    p0_xy = np.array(p0[:2])
    p1_xy = np.array(p1[:2])
    # inflate obstacles by clearance
    inflated = [(np.array(c), r + clearance) for c, r in obstacles]
    # collect avoidance waypoints
    waypoints = []
    for center, inf_r in inflated:
        if not line_circle_intersect(p0_xy, p1_xy, center, inf_r):
            continue

        # tangents from start and goal
        T1, T2 = compute_tangent_points(p0_xy, center, inf_r)
        G1, G2 = compute_tangent_points(p1_xy, center, inf_r)

        candidates = []
        for Ti in (T1, T2):
            for Gj in (G1, G2):
                L = (np.linalg.norm(p0_xy - Ti) +
                     np.linalg.norm(Ti - Gj) +
                     np.linalg.norm(Gj - p1_xy))
                candidates.append((L, Ti, Gj))

        # pick smallest
        _, best_T, best_G = min(candidates, key=lambda x: x[0])
        # add entry and exit waypoints in that order
        waypoints.extend([best_T, best_G])
    # generate smooth piecewise-linear path
    path2d = generate_piecewise_linear(p0_xy, p1_xy, waypoints, K)
    # lift to full state space and zero controls
    X0 = np.zeros((3, K))
    X0[0:2, :] = path2d
    # compute orientations
    dpts = np.diff(path2d, axis=1)
    thetas = np.arctan2(dpts[1], dpts[0])
    X0[2, :-1] = thetas
    X0[2, -1] = X0[2, -2]
    U0 = np.zeros((2, K))
    return X0, U0
