# SCvx/utils/intersample_collision.py

import numpy as np
from scipy.integrate import odeint
from typing import Callable, List, Tuple

def h_i(
    xk: np.ndarray,
    uk: np.ndarray,
    t: float,
    f: Callable[[np.ndarray, np.ndarray, float], np.ndarray],
    T: np.ndarray,
    obstacle: Tuple[np.ndarray, float],
) -> float:
    """
    Continuous‐time clearance function on the *projected* state:
      h_i = ‖T f(xk, uk, t) – p_c‖ – r
    where:
      - f(xk, uk, t) integrates the dynamics from xk over [0, t·dt_phys]
      - T projects into the plane you’re checking (e.g. ground‐plane)
      - obstacle = (p_c, r)
    """
    p_c, r = obstacle
    xt   = f(xk, uk, t)
    proj = T @ xt
    return np.linalg.norm(proj - p_c) - r


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
    Find all t* ∈ (0, dt) such that ∂h_i/∂t = 0 and ∂²h_i/∂t² > 0 (interior minima).
    """
    def phi(t: float) -> float:
        return (h_i(xk, uk, t + eps, f, T, obstacle)
              - h_i(xk, uk, t - eps, f, T, obstacle)) / (2 * eps)
    def phi2(t: float) -> float:
        return (phi(t + eps) - phi(t - eps)) / (2 * eps)

    ts   = np.linspace(eps, dt - eps, num_samples)
    phis = np.array([phi(t) for t in ts])
    raw = []
    for i in range(len(ts) - 1):
        if phis[i] == 0 or phis[i] * phis[i+1] < 0:
            a, b = ts[i], ts[i+1]
            for _ in range(30):
                c = 0.5*(a + b)
                if phi(a)*phi(c) <= 0:
                    b = c
                else:
                    a = c
                if abs(b - a) < tol:
                    break
            raw.append(0.5*(a + b))
    # keep only true minima
    return sorted(r for r in raw if 0 < r < dt and phi2(r) > 0)


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
    At (xk, uk, t_star), compute
      h0      = h_i(...)
      grad_x  = ∂h_i/∂xk via central‐difference
      grad_u  = ∂h_i/∂uk via central‐difference
    """
    h0 = h_i(xk, uk, t_star, f, T, obstacle)

    grad_x = np.zeros_like(xk)
    for j in range(len(xk)):
        xp, xm = xk.copy(), xk.copy()
        xp[j] += eps; xm[j] -= eps
        grad_x[j] = (h_i(xp, uk, t_star, f, T, obstacle)
                   - h_i(xm, uk, t_star, f, T, obstacle)) / (2*eps)

    grad_u = np.zeros_like(uk)
    for j in range(len(uk)):
        up, um = uk.copy(), uk.copy()
        up[j] += eps; um[j] -= eps
        grad_u[j] = (h_i(xk, up, t_star, f, T, obstacle)
                   - h_i(xk, um, t_star, f, T, obstacle)) / (2*eps)

    return h0, grad_x, grad_u


def make_segment_f(
    foh,                 # FirstOrderHold instance
    U_ref_k: np.ndarray,
    U_ref_kp1: np.ndarray,
    sigma: float,
):
    """
    Returns (f_seg, dt_phys) where
      f_seg(xk, _u_dummy, t) integrates
        ẋ = foh._dx(...) from 0 → t*dt_phys,
      and dt_phys = foh.dt * sigma.
    """
    dt_phys = foh.dt * sigma

    def f_seg(xk: np.ndarray, _u_dummy, t: float) -> np.ndarray:
        t_final = t * dt_phys
        xt = odeint(
            foh._dx,
            xk,
            [0.0, t_final],
            args=(U_ref_k, U_ref_kp1, sigma),
        )[1]
        return xt

    return f_seg, dt_phys