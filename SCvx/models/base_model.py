from abc import ABC, abstractmethod

import cvxpy as cvx
import numpy as np


class BaseModel(ABC):
    """
    Abstract base class for dynamical system models in Successive Convexification (SCvx).
    Defines the interface for dynamics, constraints, objectives, and trajectory initialization.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_equations(self):
        """
        Returns:
            f_func: function(x, u) -> f(x, u) dynamics
            A_func: function(x, u) -> Jacobian df/dx
            B_func: function(x, u) -> Jacobian df/du
        """
        pass

    @abstractmethod
    def get_constraints(self, X: cvx.Variable, U: cvx.Variable, X_ref: cvx.Parameter, U_ref: cvx.Parameter):
        """
        Define model-specific constraints.
        Args:
            X: cvxpy.Variable for states (n_x x K)
            U: cvxpy.Variable for inputs (n_u x K)
            X_ref: cvxpy.Parameter for previous states
            U_ref: cvxpy.Parameter for previous inputs
        Returns:
            List of cvxpy constraints
        """
        pass

    @abstractmethod
    def get_objective(self, X: cvx.Variable, U: cvx.Variable, X_ref: cvx.Parameter, U_ref: cvx.Parameter):
        """
        Define model-specific cost.
        Args:
            X: cvxpy.Variable for states
            U: cvxpy.Variable for inputs
            X_ref: cvxpy.Parameter for previous states
            U_ref: cvxpy.Parameter for previous inputs
        Returns:
            cvxpy Objective or None
        """
        pass

    @abstractmethod
    def initialize_trajectory(self, X: np.ndarray, U: np.ndarray):
        """
        Provide initial guess for state and control trajectories.
        Args:
            X: numpy array for states (modified in-place)
            U: numpy array for inputs (modified in-place)
        Returns:
            Tuple of initialized (X, U)
        """
        pass

    def nondimensionalize(self):
        """Optional: Scale model parameters and bounds to nondimensional units."""
        return

    def redimensionalize(self):
        """Optional: Revert scaling to original dimensional units."""
        return

    def x_nondim(self, x: np.ndarray) -> np.ndarray:
        """Optional: Nondimensionalize a state vector."""
        return x

    def u_nondim(self, u: np.ndarray) -> np.ndarray:
        """Optional: Nondimensionalize an input vector."""
        return u

    def x_redim(self, X: np.ndarray) -> np.ndarray:
        """Optional: Redimensionalize the state trajectory."""
        return X

    def u_redim(self, U: np.ndarray) -> np.ndarray:
        """Optional: Redimensionalize the input trajectory."""
        return U
