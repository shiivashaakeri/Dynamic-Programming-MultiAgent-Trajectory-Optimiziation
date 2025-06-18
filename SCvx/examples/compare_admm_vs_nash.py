#!/usr/bin/env python3
"""
compare_admm_vs_nash.py

Run and compare SCvx+ADMM vs. Nash-IBR on the default 3-agent unicycle scenario.
Produces:
  1. Trajectories (dark dashed=ADMM, light solid=Nash)
  2. Convergence profiles
  3. Bar charts: Control Effort & Path Length
  4. Min inter-agent separation vs time

Usage:
    python compare_admm_vs_nash.py
"""

import time
from typing import List, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from SCvx.config.default_game import AGENT_PARAMS as G_PARAMS
from SCvx.config.default_scenario import AGENT_PARAMS, CLEARANCE, D_MIN, K
from SCvx.models.game_model import GameUnicycleModel
from SCvx.models.multi_agent_model import MultiAgentModel
from SCvx.optimization.admm_coordinator import ADMMCoordinator
from SCvx.optimization.nash_solver import NashSolver
from SCvx.utils.analysis import min_inter_agent_distance
from SCvx.utils.initial_guess import initial_guess


def lighten(color, amt=0.7):
    c = np.array(mcolors.to_rgb(color))
    return tuple(c + (1 - c) * amt)


def darken(color, amt=0.7):
    c = np.array(mcolors.to_rgb(color))
    return tuple(c * (1 - amt + 0.2))


def compute_control_effort(U_list: List[np.ndarray]) -> float:
    return sum((U**2).sum() for U in U_list)


def compute_path_length(X_list: List[np.ndarray]) -> float:
    total = 0.0
    for X in X_list:
        pts = X[:2]
        total += np.linalg.norm(np.diff(pts, axis=1), axis=0).sum()
    return total


def min_sep_time_series(X_list: List[np.ndarray]) -> np.ndarray:
    N = len(X_list)
    dmin = np.zeros(K)
    for k in range(K):
        d_ks = []
        for i in range(N):
            for j in range(i + 1, N):
                pi, pj = X_list[i][:2, k], X_list[j][:2, k]
                d_ks.append(np.linalg.norm(pi - pj))
        dmin[k] = min(d_ks)
    return dmin


def run_admm(
    X0: List[np.ndarray],
    U0: List[np.ndarray],
    sigma_ref: float = 1.0,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[float], List[float], float]:
    mam = MultiAgentModel(AGENT_PARAMS, d_min=D_MIN)
    coord = ADMMCoordinator(mam, rho_admm=1.0, max_iter=20)
    t0 = time.time()
    X_admm, U_admm, sigma_admm, pr, du = coord.solve(X0, U0, sigma_ref, verbose=False)
    return X_admm, U_admm, pr, du, time.time() - t0


def run_nash(
    X0: List[np.ndarray],
    U0: List[np.ndarray],
    sigma_ref: float = 1.0,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[float], float]:
    mam = MultiAgentModel(G_PARAMS)
    # replace with GameUnicycleModel
    for i, p in enumerate(G_PARAMS):
        mam.models[i] = GameUnicycleModel(
            r_init=p["r_init"],
            r_final=p["r_final"],
            obstacles=p["obstacles"],
            control_weight=p["control_weight"],
            collision_weight=p["collision_weight"],
            collision_radius=p["collision_radius"],
            control_rate_weight=p["control_rate_weight"],
            curvature_weight=p["curvature_weight"],
        )
    solver = NashSolver(mam, max_iter=20, tol=1e-3)
    t0 = time.time()
    Xn, Un, hist = solver.solve(X0, U0, sigma_ref, verbose=False)
    return Xn, Un, hist, time.time() - t0


def warm_start():
    X0_list, U0_list = [], []
    for p in AGENT_PARAMS:
        X0, U0 = initial_guess(p["r_init"], p["r_final"], p.get("obstacles", []), CLEARANCE, K)
        X0_list.append(X0)
        U0_list.append(U0)
    return X0_list, U0_list


def run_solvers(X0_list, U0_list):
    X_admm, U_admm, pr_hist, du_hist, t_admm = run_admm(X0_list, U0_list)
    X_nash, U_nash, nh_hist, t_nash = run_nash(X0_list, U0_list)
    return X_admm, U_admm, pr_hist, du_hist, t_admm, X_nash, U_nash, nh_hist, t_nash


def compute_metrics(X_admm, U_admm, X_nash, U_nash, pr_hist, nh_hist, t_admm, t_nash):
    eff_admm = compute_control_effort(U_admm)
    eff_nash = compute_control_effort(U_nash)
    len_admm = compute_path_length(X_admm)
    len_nash = compute_path_length(X_nash)
    sep_admm, _ = min_inter_agent_distance(X_admm)
    sep_nash, _ = min_inter_agent_distance(X_nash)

    comparison_summary = pd.DataFrame(
        [
            {
                "Method": "ADMM",
                "Iter": len(pr_hist),
                "Time (s)": t_admm,
                "Min-sep": sep_admm,
                "Effort": eff_admm,
                "Length": len_admm,
            },
            {
                "Method": "NASH",
                "Iter": len(nh_hist),
                "Time (s)": t_nash,
                "Min-sep": sep_nash,
                "Effort": eff_nash,
                "Length": len_nash,
            },
        ]
    )
    print("\n=== Comparison Summary ===")
    print(comparison_summary.to_string(index=False))
    return comparison_summary


def plot_trajectories(X_admm, X_nash):
    plt.figure(figsize=(6, 6))
    cmap = plt.get_cmap("tab10")
    obstacles = G_PARAMS[0]["obstacles"]
    robot_radius = G_PARAMS[0]["collision_radius"]

    for i in range(len(G_PARAMS)):
        base = cmap(i)
        light = lighten(base)
        dark = darken(base)

        P = X_admm[i][:2]
        plt.plot(P[0], P[1], "--", color=dark, linewidth=1.5, label=f"Agent {i} (ADMM)")
        Q = X_nash[i][:2]
        plt.plot(Q[0], Q[1], "-", color=light, linewidth=3.0, label=f"Agent {i} (Nash)")

    ax = plt.gca()
    for (cx, cy), r in obstacles:
        circle = plt.Circle((cx, cy), r + robot_radius, color="r", fill=False, linestyle="--", linewidth=1.5)
        ax.add_patch(circle)

    plt.axis("equal")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Trajectories: ADMM (dark dashed) vs Nash (light solid)")
    plt.legend(fontsize="small", ncol=2)
    plt.grid(True)
    plt.tight_layout()


def plot_convergence_profiles(pr_hist, du_hist, nh_hist):
    plt.figure(figsize=(6, 4))
    plt.semilogy(pr_hist, "--", label="ADMM primal", color="C0")
    plt.semilogy(du_hist, "--", label="ADMM dual", color="C1")
    plt.semilogy(nh_hist, "-", label="Nash ‖ΔX‖", color="C2")
    plt.xlabel("Iteration")
    plt.ylabel("Residual")
    plt.title("Convergence Profiles")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


def plot_bar_charts(df):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    df.set_index("Method")[["Effort"]].plot.bar(ax=axes[0], color=["#4C72B0", "#55A868"], legend=False)
    axes[0].set_title("Control Effort")
    axes[0].grid(axis="y")
    df.set_index("Method")[["Length"]].plot.bar(ax=axes[1], color=["#C44E52", "#8172B2"], legend=False)
    axes[1].set_title("Path Length")
    axes[1].grid(axis="y")
    plt.tight_layout()


def plot_min_inter_agent_separation(X_admm, X_nash):
    d_admm = min_sep_time_series(X_admm)
    d_nash = min_sep_time_series(X_nash)

    plt.figure(figsize=(6, 4))
    plt.plot(range(K), d_admm, "--", label="ADMM dₘᵢₙ(k)", color="C3")
    plt.plot(range(K), d_nash, "-", label="Nash dₘᵢₙ(k)", color="C4")
    plt.axhline(D_MIN, linestyle=":", color="k", label=f"Dₘᵢₙ = {D_MIN}")
    plt.xlabel("Time step k")
    plt.ylabel("Min separation [m]")
    plt.title("Inter-agent separation over time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


def main():
    X0_list, U0_list = warm_start()
    X_admm, U_admm, pr_hist, du_hist, t_admm, X_nash, U_nash, nh_hist, t_nash = run_solvers(X0_list, U0_list)
    df = compute_metrics(X_admm, U_admm, X_nash, U_nash, pr_hist, nh_hist, t_admm, t_nash)  # noqa: PD901

    plot_trajectories(X_admm, X_nash)
    plot_convergence_profiles(pr_hist, du_hist, nh_hist)
    plot_bar_charts(df)
    plot_min_inter_agent_separation(X_admm, X_nash)

    plt.show()


if __name__ == "__main__":
    main()
