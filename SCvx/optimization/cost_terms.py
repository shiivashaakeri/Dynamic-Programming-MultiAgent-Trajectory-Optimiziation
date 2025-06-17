import cvxpy as cvx


def control_effort(U, weight):
    return weight * cvx.sum_squares(U)

def control_rate(U, weight):
    dU = U[:, 1:] - U[:, :-1]
    return weight * cvx.sum_squares(dU)

def curvature(X, weight):
    dtheta = X[2, 1:] - X[2, :-1]
    return weight * cvx.sum_squares(dtheta)

def collision_penalty(p_i, neighbors, radius, weight):
    constraints = []
    slack_vars = []
    cost = 0
    for p_j in neighbors:
        s = cvx.Variable(p_i.shape[1], nonneg=True)
        constraints.append(cvx.norm(p_i - p_j, 2, axis=0) >= radius - s)
        cost += weight * cvx.norm(s, 1)
        slack_vars.append(s)
    return cost, constraints, slack_vars
