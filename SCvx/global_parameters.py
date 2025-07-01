# Global parameters for Successive Convexification (SCvx)

# Number of discretization points
K = 50

# Maximum number of SCvx iterations
MAX_ITER = 30

# Initial trust region radius
TRUST_RADIUS0 = 17.0

# Tolerance for convergence (e.g., change in cost or trajectory)
CONV_TOL = 1e-3

# Weights for defect and slack variables
WEIGHT_NU = 1e4
WEIGHT_SLACK = 1e6
WEIGHT_SIGMA = 100.0
