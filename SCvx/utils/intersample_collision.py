from typing import Callable, List, Tuple

import numpy as np


def h_i(
    xk: np.ndarray,
    uk: np.ndarray,
    t: float,
    f: Callable[[np.ndarray, np.ndarray, float], np.ndarray],
    T: np.ndarray,
    obstacle: Tuple[np.ndarray, float],
) -> float:
    p_c, r = obstacle
    xt = f(xk, uk, t)
    proj = T @ xt
    return np.linalg.norm(proj - p_c) ** 2 - r**2


def find_critical_times(
    xk: np.ndarray,
    uk: np.ndarray,
    f: Callable[[np.ndarray, np.ndarray, float], np.ndarray],
    T: np.ndarray,
    obstacle: Tuple[np.ndarray, float],
    dt: float,
    num_samples: int = 100,
    eps: float = 1e-4,
    tol: float = 1e-6,
) -> List[float]:
    """
    As before, but now only keep roots where d2h/dt2 > 0.
    """

    # (re)define phi = dh/dt via central differences
    def phi(t: float) -> float:
        return (h_i(xk, uk, t + eps, f, T, obstacle) - h_i(xk, uk, t - eps, f, T, obstacle)) / (2 * eps)

    # second derivative
    def phi2(t: float) -> float:
        return (phi(t + eps) - phi(t - eps)) / (2 * eps)

    # find sign‐change roots exactly as before
    ts = np.linspace(eps, dt - eps, num_samples)
    phis = np.array([phi(t) for t in ts])

    raw_roots: List[float] = []
    for i in range(len(ts) - 1):
        if phis[i] == 0 or phis[i] * phis[i + 1] < 0:
            a, b = ts[i], ts[i + 1]
            for _ in range(30):
                c = 0.5 * (a + b)
                if phi(a) * phi(c) <= 0:
                    b = c
                else:
                    a = c
                if abs(b - a) < tol:
                    break
            raw_roots.append(0.5 * (a + b))

    # keep only interior minima (second‐derivative > 0)
    minima = sorted(r for r in raw_roots if 0 < r < dt and phi2(r) > 0)
    return minima


def linearize_h(
    xk: np.ndarray,
    uk: np.ndarray,
    t_star: float,
    f: Callable[[np.ndarray, np.ndarray, float], np.ndarray],
    T: np.ndarray,
    obstacle: Tuple[np.ndarray, float],
    eps: float = 1e-4,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute h*, ∇_x h, ∇_u h at (xk, uk, t_star) via finite differences.

    Returns:
      h_val   : float
      grad_x  : np.ndarray, shape (n_x,)
      grad_u  : np.ndarray, shape (n_u,)
    """
    # baseline
    h0 = h_i(xk, uk, t_star, f, T, obstacle)

    # gradient w.r.t. xk
    grad_x = np.zeros_like(xk)
    for j in range(len(xk)):
        xp = xk.copy()
        xp[j] += eps
        xm = xk.copy()
        xm[j] -= eps
        grad_x[j] = (h_i(xp, uk, t_star, f, T, obstacle) - h_i(xm, uk, t_star, f, T, obstacle)) / (2 * eps)

    # gradient w.r.t. uk
    grad_u = np.zeros_like(uk)
    for j in range(len(uk)):
        up = uk.copy()
        up[j] += eps
        um = uk.copy()
        um[j] -= eps
        grad_u[j] = (h_i(xk, up, t_star, f, T, obstacle) - h_i(xk, um, t_star, f, T, obstacle)) / (2 * eps)

    return h0, grad_x, grad_u


from scipy.integrate import odeint


def make_segment_f(
    foh,  # your FirstOrderHold instance
    U_ref_k: np.ndarray,
    U_ref_kp1: np.ndarray,
    sigma: float,
):
    """
    Returns a function f_seg(xk, t) that integrates the true dynamics
    over [0, t * dt_phys], where dt_phys = foh.dt * sigma.
    """
    dt_phys = foh.dt * sigma

    def f_seg(xk: np.ndarray, _u_dummy, t: float) -> np.ndarray:
        # ignore _u_dummy; FOH._dx uses U_ref_k and U_ref_kp1 internally
        t_final = t * dt_phys
        # integrate from 0 → t_final
        xt = odeint(
            foh._dx,
            xk,
            [0.0, t_final],
            args=(U_ref_k, U_ref_kp1, sigma),
        )[1]
        return xt

    return f_seg, dt_phys

