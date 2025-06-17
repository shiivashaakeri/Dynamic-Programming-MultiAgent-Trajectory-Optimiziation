import numpy as np
from scipy.integrate import odeint


class FirstOrderHold:
    """
    First-Order Hold discretizer for continuous-time nonlinear dynamics.

    Approximates dynamics with a time-varying linear model over discretized intervals
    using first-order hold on control inputs.
    """

    def __init__(self, model, K):
        self.model = model
        self.K = K
        self.n_x = model.n_x
        self.n_u = model.n_u

        # Preallocate arrays for discretization matrices
        self.A_bar = np.zeros((self.n_x * self.n_x, K - 1))
        self.B_bar = np.zeros((self.n_x * self.n_u, K - 1))
        self.C_bar = np.zeros((self.n_x * self.n_u, K - 1))
        self.S_bar = np.zeros((self.n_x, K - 1))
        self.z_bar = np.zeros((self.n_x, K - 1))

        # Indices for slicing the augmented state vector
        x_end = self.n_x
        A_end = x_end + self.n_x * self.n_x
        B_end = A_end + self.n_x * self.n_u
        C_end = B_end + self.n_x * self.n_u
        S_end = C_end + self.n_x
        z_end = S_end + self.n_x

        self.x_ind = slice(0, x_end)
        self.A_ind = slice(x_end, A_end)
        self.B_ind = slice(A_end, B_end)
        self.C_ind = slice(B_end, C_end)
        self.S_ind = slice(C_end, S_end)
        self.z_ind = slice(S_end, z_end)

        # Retrieve dynamics functions
        self.f, self.A, self.B = model.get_equations()

        # Initial condition for augmented state
        self.V0 = np.zeros((z_end,))
        # Set Phi_A initial condition to identity
        self.V0[self.A_ind] = np.eye(self.n_x).reshape(-1, order='F')

        # Time-step (assumes normalized final time = 1)
        self.dt = 1.0 / (K - 1)

    def calculate_discretization(self, X, U, sigma):
        """
        Compute discretization matrices given reference trajectories.

        Args:
            X: np.ndarray, shape (n_x, K)  -- reference states
            U: np.ndarray, shape (n_u, K)  -- reference controls
            sigma: float                   -- total time scaling

        Returns:
            A_bar, B_bar, C_bar, S_bar, z_bar
        """
        for k in range(self.K - 1):
            # Set state in augmented vector
            self.V0[self.x_ind] = X[:, k]
            # Integrate augmented ODE over one dt interval
            V = odeint(
                self._ode_dVdt,
                self.V0,
                [0.0, self.dt],
                args=(U[:, k], U[:, k + 1], sigma)
            )[1]

            # Extract state transition matrix Phi
            Phi = V[self.A_ind].reshape((self.n_x, self.n_x), order='F')

            # Populate discretization matrices
            self.A_bar[:, k] = Phi.flatten(order='F')
            B_mat = V[self.B_ind].reshape((self.n_x, self.n_u), order='F')
            C_mat = V[self.C_ind].reshape((self.n_x, self.n_u), order='F')
            self.B_bar[:, k] = (Phi @ B_mat).flatten(order='F')
            self.C_bar[:, k] = (Phi @ C_mat).flatten(order='F')
            self.S_bar[:, k] = (Phi @ V[self.S_ind]).flatten()
            self.z_bar[:, k] = (Phi @ V[self.z_ind]).flatten()

        return self.A_bar, self.B_bar, self.C_bar, self.S_bar, self.z_bar

    def _ode_dVdt(self, V, t, u0, u1, sigma):
        """
        Augmented ODE for state, transition matrix, and affine terms.
        """
        # FOH interpolation factors
        alpha = (self.dt - t) / self.dt
        beta = t / self.dt

        # Extract x and compute interpolated control
        x = V[self.x_ind]
        u = u0 + (t / self.dt) * (u1 - u0)

        # Precompute Jacobian-based terms
        A_sub = sigma * self.A(x, u)
        B_sub = sigma * self.B(x, u)
        f_sub = self.f(x, u).flatten()

        # Inverse of Phi_A up to current time
        Phi_A = V[self.A_ind].reshape((self.n_x, self.n_x), order='F')
        Phi_A_inv = np.linalg.inv(Phi_A)

        # Initialize derivative
        dV = np.zeros_like(V)
        # State derivative
        dV[self.x_ind] = sigma * f_sub
        # Transition matrix derivative
        dV[self.A_ind] = (A_sub @ Phi_A).reshape(-1, order='F')
        # Input effect derivatives
        dV[self.B_ind] = (Phi_A_inv @ B_sub).reshape(-1, order='F') * alpha
        dV[self.C_ind] = (Phi_A_inv @ B_sub).reshape(-1, order='F') * beta
        # Affine terms derivatives
        dV[self.S_ind] = (Phi_A_inv @ f_sub.reshape((self.n_x, 1))).flatten()
        # z term due to non-homogeneous part
        z_term = -A_sub @ x - B_sub @ u
        dV[self.z_ind] = (Phi_A_inv @ z_term).flatten()

        return dV

    def integrate_nonlinear_piecewise(self, X_lin, U, sigma):
        """
        Numerically integrate the true nonlinear dynamics piecewise over each interval.
        """
        X_nl = np.zeros_like(X_lin)
        X_nl[:, 0] = X_lin[:, 0]
        for k in range(self.K - 1):
            X_nl[:, k + 1] = odeint(
                self._dx,
                X_lin[:, k],
                [0.0, self.dt * sigma],
                args=(U[:, k], U[:, k + 1], sigma)
            )[1]
        return X_nl

    def integrate_nonlinear_full(self, x0, U, sigma):
        """
        Simulate full nonlinear dynamics from initial state.
        """
        X_nl = np.zeros((self.n_x, self.K))
        X_nl[:, 0] = x0
        for k in range(self.K - 1):
            X_nl[:, k + 1] = odeint(
                self._dx,
                X_nl[:, k],
                [0.0, self.dt * sigma],
                args=(U[:, k], U[:, k + 1], sigma)
            )[1]
        return X_nl

    def _dx(self, x, t, u0, u1, sigma):
        """
        True nonlinear dynamics derivative for integration.
        """
        u = u0 + (t / (self.dt * sigma)) * (u1 - u0)
        return self.f(x, u).flatten()
