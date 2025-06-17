def print_iteration(it, nu_norm, slack_norm, primal_res, dual_res, dx, ds, sigma, tr_radius):
    print(
        f"Iter {it:2d} | v={nu_norm:7.3e} | slack={slack_norm:7.3e} "
        f"| p_res={primal_res:7.3e} | d_res={dual_res:7.3e} "
        f"| Δx={dx:6.2e} | Δs={ds:6.2e} | o={sigma:6.3f} | tr={tr_radius:6.3f}"
    )


def print_summary(total_iters, sigma_final, runtime=None):
    print("\n=== SCvx+ADMM Summary ===")
    print(f"  Total iterations: {total_iters}")
    print(f"  Final time scale o: {sigma_final:.3f}")
    if runtime is not None:
        print(f"  Total runtime:    {runtime:.2f}s")
    print("=========================\n")
