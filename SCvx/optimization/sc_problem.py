import cvxpy as cvx

from ..global_parameters import K


class SCProblem:
    """
    Defines the convex subproblem for Successive Convexification (SCvx).

    Args:
        model: an instance of BaseModel implementing dynamics and constraints
        K: number of discretization points
    """

    def __init__(self, model):
        self.model = model
        self.n_x = model.n_x
        self.n_u = model.n_u
        self.K = K

        # Decision Variables
        self.var = {}
        self.var["X"] = cvx.Variable((self.n_x, self.K))  # states
        self.var["U"] = cvx.Variable((self.n_u, self.K))  # controls
        self.var["nu"] = cvx.Variable((self.n_x, self.K - 1))  # defect (virtual controls)
        self.var["sigma"] = cvx.Variable(nonneg=True)  # time scale / final time

        # Parameters to be set each iteration
        self.par = {}
        # Discretization matrices
        self.par["A_bar"] = cvx.Parameter((self.n_x * self.n_x, self.K - 1))
        self.par["B_bar"] = cvx.Parameter((self.n_x * self.n_u, self.K - 1))
        self.par["C_bar"] = cvx.Parameter((self.n_x * self.n_u, self.K - 1))
        self.par["S_bar"] = cvx.Parameter((self.n_x, self.K - 1))
        self.par["z_bar"] = cvx.Parameter((self.n_x, self.K - 1))
        # Previous trajectory
        self.par["X_ref"] = cvx.Parameter((self.n_x, self.K))
        self.par["U_ref"] = cvx.Parameter((self.n_u, self.K))
        self.par["sigma_ref"] = cvx.Parameter(nonneg=True)
        # Trust-region and penalty weights
        self.par["weight_nu"] = cvx.Parameter(nonneg=True)
        self.par["weight_sigma"] = cvx.Parameter(nonneg=True)
        self.par["tr_radius"] = cvx.Parameter(nonneg=True)
        self.par["weight_slack"] = cvx.Parameter(nonneg=True)

        # Build constraints list
        constraints = []

        # Model-specific constraints (bounds, linearized obstacles, etc.)
        constraints += model.get_constraints(self.var["X"], self.var["U"], self.par["X_ref"], self.par["U_ref"])

        # Dynamics constraints
        for k in range(self.K - 1):
            A_k = cvx.reshape(self.par["A_bar"][:, k], (self.n_x, self.n_x))
            B_k = cvx.reshape(self.par["B_bar"][:, k], (self.n_x, self.n_u))
            C_k = cvx.reshape(self.par["C_bar"][:, k], (self.n_x, self.n_u))
            S_k = self.par["S_bar"][:, k]
            z_k = self.par["z_bar"][:, k]

            constraints.append(
                self.var["X"][:, k + 1]
                == A_k @ self.var["X"][:, k]
                + B_k @ self.var["U"][:, k]
                + C_k @ self.var["U"][:, k + 1]
                + S_k * self.var["sigma"]
                + z_k
                + self.var["nu"][:, k]
            )

        # Trust-region constraint
        dx = self.var["X"] - self.par["X_ref"]
        du = self.var["U"] - self.par["U_ref"]
        ds = self.var["sigma"] - self.par["sigma_ref"]
        constraints.append(cvx.norm(dx, 1) + cvx.norm(du, 1) + cvx.abs(ds) <= self.par["tr_radius"])

        # Objective: penalize defect, slack, and time
        obj_terms = [
            self.par['weight_nu']    * cvx.norm(self.var['nu'], 1),
            self.par['weight_slack'] * sum(cvx.sum(s) for s in self.model.s_prime),
            self.par['weight_sigma'] * self.var['sigma']
        ]
        objective = cvx.Minimize(sum(obj_terms))        # Define problem
        self.prob = cvx.Problem(objective, constraints)

    def set_parameters(self, **kwargs):
        """
        Set parameter values before solving.
        """
        for key, val in kwargs.items():
            if key in self.par:
                self.par[key].value = val
            else:
                raise KeyError(f"Parameter '{key}' not found in SCProblem.")

    def solve(self, **kwargs):
        """
        Solve the convex subproblem.
        Returns:
            error: bool indicating whether a solver error occurred
        """
        try:
            self.prob.solve(**kwargs)
            return False
        except cvx.SolverError:
            return True

    def get_variable(self, name):
        """
        Retrieve the value of a decision variable.

        Args:
            name: one of 'X', 'U', 'nu', 'sigma'
        Returns:
            numpy array of the variable's value
        """
        if name in self.var:
            return self.var[name].value
        else:
            raise KeyError(f"Variable '{name}' not found.")

    def print_available_parameters(self):
        print("Available parameters:")
        for k in self.par.keys():  # noqa: SIM118
            print(f"  {k}")

    def print_available_variables(self):
        print("Available variables:")
        for k in self.var.keys():  # noqa: SIM118
            print(f"  {k}")
