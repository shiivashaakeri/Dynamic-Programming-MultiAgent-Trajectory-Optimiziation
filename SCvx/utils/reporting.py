"""Console reporting helpers for iterative solvers (SCvx, ADMM, Nash)."""


def print_iteration(
    it: int,
    nu_norm: float,
    slack_norm: float,
    primal_res: float,
    dual_res: float,
    dx: float,
    ds: float,
    sigma: float,
    tr_radius: float,
):
    """Pretty-print a single iteration line."""
    print(
        f"Iter {it:2d} | v={nu_norm:7.3e} | slack={slack_norm:7.3e} "
        f"| p_res={primal_res:7.3e} | d_res={dual_res:7.3e} "
        f"| Δx={dx:6.2e} | Δs={ds:6.2e} | o={sigma:5.3f} | tr={tr_radius:5.3f}"
    )


def print_summary(total_iters: int, sigma_final: float, runtime: float | None = None):
    print("\n=== Solver Summary ===")
    print(f"  Total iterations: {total_iters}")
    print(f"  Final o:         {sigma_final:.3f}")
    if runtime is not None:
        print(f"  Runtime:         {runtime:.2f}s")
    print("======================\n")
