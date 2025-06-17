import numpy as np

from SCvx.global_parameters import CONV_TOL, MAX_ITER, TRUST_RADIUS0, WEIGHT_NU, WEIGHT_SIGMA, WEIGHT_SLACK, K
from SCvx.utils.logging import Logger

from ..discretization.first_order_hold import FirstOrderHold
from .sc_problem import SCProblem


class SCVXSolver:
    """
    Orchestrator for the Successive Convexification (SCvx) algorithm, with integrated logging.
    """

    def __init__(self, model):
        self.model = model
        self.K = K
        # Algorithm parameters
        self.max_iter = MAX_ITER
        self.conv_tol = CONV_TOL
        self.tr_radius = TRUST_RADIUS0
        self.weight_nu = WEIGHT_NU
        self.weight_slack = WEIGHT_SLACK
        self.weight_sigma = WEIGHT_SIGMA

        # Core components: discretizer and convex problem
        self.discretizer = FirstOrderHold(model, self.K)
        self.problem = SCProblem(model)

        # Logger for iteration metrics
        self.logger = Logger()

    def solve(self, verbose=False, initial_sigma=1.0):
        """
        Run the SCvx iterative loop, logging metrics each iteration.

        Returns:
            X, U, sigma: final trajectory and time scaling
            logger: Logger instance containing per-iteration records
        """
        # Initialize trajectory
        X = np.zeros((self.model.n_x, self.K))
        U = np.zeros((self.model.n_u, self.K))
        X, U = self.model.initialize_trajectory(X, U)
        sigma = initial_sigma

        # Clear previous logs
        self.logger.clear()

        for it in range(self.max_iter):
            # Discretize around current trajectory
            A_bar, B_bar, C_bar, S_bar, z_bar = self.discretizer.calculate_discretization(X, U, sigma)

            # Set parameters for convex subproblem
            self.problem.set_parameters(
                A_bar=A_bar,
                B_bar=B_bar,
                C_bar=C_bar,
                S_bar=S_bar,
                z_bar=z_bar,
                X_ref=X,
                U_ref=U,
                sigma_ref=sigma,
                weight_nu=self.weight_nu,
                weight_slack=self.weight_slack,
                weight_sigma=self.weight_sigma,
                tr_radius=self.tr_radius,
            )

            # Solve convex problem
            error = self.problem.solve(solver="ECOS", warm_start=True)
            if error:
                raise RuntimeError(f"SCvx iteration {it}: convex subproblem infeasible")

            # Extract solution
            X_new = self.problem.get_variable("X")
            U_new = self.problem.get_variable("U")
            nu_new = self.problem.get_variable("nu")
            sigma_new = self.problem.get_variable("sigma")

            # Compute metrics
            nu_norm = np.linalg.norm(nu_new, 1)
            slack_norm = self._compute_slack_norm()
            dx = np.linalg.norm(X_new - X)
            du = np.linalg.norm(U_new - U)
            ds = abs(sigma_new - sigma)

            # Log metrics
            record = {
                "iter": it,
                "nu_norm": nu_norm,
                "slack_norm": slack_norm,
                "dx": dx,
                "du": du,
                "ds": ds,
                "sigma": sigma_new,
            }
            self.logger.log(record)

            if verbose:
                print(f"Iter {it}: nu={nu_norm:.3e}, slack={slack_norm:.3e}, dx={dx:.3e}, ds={ds:.3e}")

            # Check convergence
            if nu_norm < self.conv_tol and slack_norm < self.conv_tol and dx < self.conv_tol and ds < self.conv_tol:
                break

            # Update trust region
            self._update_trust_region(nu_norm, slack_norm)

            # Prepare for next iteration
            X, U, sigma = X_new, U_new, sigma_new

        # Redimensionalize if needed
        X, U = self.model.x_redim(X), self.model.u_redim(U)
        return X, U, sigma, self.logger

    def _compute_slack_norm(self):
        """Compute total slack across all obstacles."""
        total_slack = 0.0
        for s_var in self.model.s_prime:
            val = s_var.value
            total_slack += np.sum(val)
        return total_slack

    def _update_trust_region(self, nu_norm, slack_norm):
        if nu_norm < 1e-2 and slack_norm < 1e-2:
            # both low ⇒ the linear model is good, we can expand
            self.tr_radius = min(self.tr_radius * 1.5, 50.0)
        else:
            # otherwise keep it large (or even expand) to let the solver maneuver
            self.tr_radius = min(self.tr_radius * 1.2, 50.0)
        # enforce a reasonable floor
        self.tr_radius = max(self.tr_radius, 1e-3)
