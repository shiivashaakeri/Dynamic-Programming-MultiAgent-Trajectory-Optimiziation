import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp
from numpy import linalg as LA
from jax.numpy import linalg as la
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import jax
import jax.numpy as jnp
from scipy import signal


## dynamics
def descete_f(dt):
    A = np.array([[0, 0, 1, 0],
                  [0, 0, 0, 1],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]])
    B = np.array([[0, 0],
                  [0, 0],
                  [1, 0],
                  [0, 1]])
    C = np.eye(n)
    D = np.zeros((n, m))
    sys = signal.StateSpace(A, B, C, D)
    sysd = sys.to_discrete(dt)
    Ad = sysd.A
    Bd = sysd.B
    return [Ad, Bd]


def x_traj_opt(X_traj, trust_region):
    tol = 0.1
    s_val = {}
    s_bar_val = {}
    s_bar_val_new = {}
    diff = 0
    r_all = {}  ## initialize dual for all agents
    ## Initialize the perturbation variables
    for name in robots_name:
        r_all[name] = np.ones((T * 2,1)) * 10
        s_val[name] = np.zeros((T, n + m))
        s_bar_val[name] = np.zeros((T, 2))
        s_bar_val_new[name] = np.zeros((T, 2)) + 1
        diff += LA.norm(s_bar_val[name] - s_bar_val_new[name], 2)
    ## ADMM loop
    rho = 100000  ## Lagrangian penalty
    # while diff > tol:
    for iter in range(5):
        ##########################################################
        ## solve d - minimization individually (primal variables)
        for name in robots_name:
            r_i = r_all[name]  ## dual vector
            X_des_i = x_des[name]
            x_des_i = X_des_i[0:n]
            X_traj_i = X_traj[name]
            x_traj_i = X_traj_i[0:T, 0:n]
            u_traj_i = X_traj_i[0:T - 1, n:n + m]  # extract the control
            # Primary variables
            s_i = cp.Variable((T, n + m))
            S_i = cp.Variable(T)
            d_i = s_i[0:T, 0:n]
            x_i = d_i[:, 0:2]  ## only position
            s_i_vec = cp.reshape(x_i, (T * 2, 1), order="C")  ## vectorized primal variables
            w_i = s_i[0:T - 1, n:n + m]
            # Duplicate variables
            s_bar_val_i = s_bar_val[name][:,0:2]
            s_bar_val_i_vec = np.reshape(s_bar_val_i, (T * 2, 1), order="C")  ## vectorized duplicated variables
            ##########################################################
            ## s - minimization (primal variables)
            # Construct the augmented Lagrangian
            L_rho = 1 * cp.sum_squares(u_traj_i + w_i) + 1 * r_i.T @ (s_i_vec - s_bar_val_i_vec) + rho / 2 * cp.square(
                cp.norm(s_i_vec - s_bar_val_i_vec, 2))
            # L_rho = 1*cp.sum_squares(u_traj_i + w_i) + 10000 * cp.norm(S_i,2)
            constraints_s = [d_i[0, :] == np.zeros(n)]
            constraints_s.append(d_i[T - 1, :] + x_traj_i[T - 1, :] == x_des_i)
            for t in range(T - 1):
                x_traj_t = x_traj_i[t, :]
                x_traj_tp1 = x_traj_i[t + 1, :]
                u_traj_t = u_traj_i[t, :]
                d_t = d_i[t, :]
                d_tp1 = d_i[t + 1, :]
                w_t = w_i[t, :]
                f_t = Ad @ x_traj_t + Bd @ u_traj_t
                constraints_s.append(x_traj_tp1 + d_tp1 == f_t + Ad @ d_t + Bd @ w_t)
                constraints_s.append(cp.norm(w_t, 1) <= trust_region)
                # if name == "robot02":
                #     constraints_s.append(cp.norm(u_traj_t - w_t, 1) <= 0)
                ## boundary constraints
                constraints_s.append(x_traj_t[0] + d_t[0] <= 22)
                constraints_s.append(x_traj_t[0] + d_t[0] >= -1)
                constraints_s.append(x_traj_t[1] + d_t[1] >= -1)
                constraints_s.append(x_traj_t[1] + d_t[1] <= 20)
                # L_rho += 2 * u_traj_t.T @ w_t

            problem = cp.Problem(cp.Minimize(L_rho), constraints_s)
            problem.solve(solver=cp.CLARABEL)
            s_val[name] = s_i.value

            cc = 2
        ##########################################################
        ## sbar - minimization (centralized with duplicated variables)
        s_bar = {}
        S = {}
        for name in robots_name:
            s_bar[name] = cp.Variable((T, 2))
            S[name] = cp.Variable(T)
        L_rho_bar = 0
        # Construct the augmented Lagrangian
        for name in robots_name:
            s_val_i = s_val[name][:,0:2]
            s_val_i_vec = np.reshape(s_val_i, (T * 2, 1), order="C")  ## vectorized primal variables
            s_bar_i = s_bar[name]
            s_bar_i_vec = cp.reshape(s_bar_i, (T * 2, 1), order="C")  ## vectorized duplicated variables
            r_i = r_all[name]
            S_i = S[name]
            L_rho_bar += 1 * r_i.T @ (s_val_i_vec - s_bar_i_vec) + rho / 2 * cp.square(
                cp.norm(s_val_i_vec - s_bar_i_vec, 2)) + 1000000 * cp.norm(S_i, 1)
        constraints_bar = []
        for t in range(T):
            for name in robots_name:
                X_traj_i = X_traj[name]
                x_traj_i = X_traj_i[0:T, 0:n]
                x_traj_i_t = x_traj_i[t, :]
                d_bar_i = s_bar[name][:, 0:n]
                d_bar_i_t = d_bar_i[t, :]
                S_i = S[name]
                S_i_t = S_i[t]
                ## loop through obstacles
                for obs_name in robots_name:
                    X_traj_j = X_traj[obs_name]
                    x_traj_j = X_traj_j[0:T, 0:n]
                    x_traj_j_t = x_traj_j[t, :]
                    if obs_name != name:  ## exclude itself
                        S_fun = 2 * R - LA.norm(x_traj_i_t[0:2] - x_traj_j_t[0:2], 2)
                        S_grad = (x_traj_i_t[0:2] - x_traj_j_t[0:2]).T / cp.norm(x_traj_i_t[0:2] - x_traj_j_t[0:2], 2)
                        constraints_bar.append(
                            S_fun - S_grad @ d_bar_i_t[0:2] <= S_i_t
                        )
                        constraints_bar.append(S_i_t >= 0)
        problem = cp.Problem(cp.Minimize(L_rho_bar), constraints_bar)
        problem.solve(solver=cp.CLARABEL)
        print("cost:    ", problem.value)
        ## retrieve values
        for name in robots_name:
            s_bar_val_new[name] = s_bar[name].value

        ## dual - maximization (constraints with the duplicated variables)
        for name in robots_name:
            r_i = r_all[name]
            s_val_i = s_val[name][:,0:2]
            s_val_i_vec = np.reshape(s_val_i, (T * 2, 1), order="C")  ## vectorized primal variables
            s_bar_val_i = s_bar_val[name][:,0:2]
            s_bar_val_i_vec = np.reshape(s_bar_val_i, (T * 2, 1), order="C")  ## vectorized duplicated variables
            r_i += rho * (s_val_i_vec - s_bar_val_i_vec)
            r_all[name] = r_i  ## Update the dual variables

        ## Find the difference
        diff = 0
        for name in robots_name:
            diff += LA.norm(s_bar_val[name] - s_val[name][:,0:2], 2)
        print("Difference:  ", diff)
        ## replace the current s_bar with s_bar_new
        for name in robots_name:
            s_bar_val[name] = s_bar[name].value
    print("update X")
    for name in robots_name:
        if s_val[name].all() != None:
            X_traj[name] += np.array(s_val[name])

    return X_traj


## Initializatoin
def x_initial(x_ini, x_des):
    x_traj = {}
    for name in robots_name:
        x_traj[name] = np.linspace(x_ini[name], x_des[name], T)
        # for t in range(T):
        #     x_traj[name][t,:] = x_ini[name]
    return x_traj


## Plotting
def plot_traj(x_traj):
    fig, ax = plt.subplots()
    ax.clear()
    for t in range(T):
        ax.clear()
        ax.set_xlim(-1, 21)
        ax.set_ylim(-1, 21)
        for subname in robots_name:
            x_traj_i = x_traj[subname]
            ax.plot(x_traj_i[:, 0], x_traj_i[:, 1])
            x_traj_t = x_traj_i[t, :]
            if subname == "robot01":
                # plt.plot(x_traj_t[0], x_traj_t[1], "r.")
                ax.plot(x_traj_t[0], x_traj_t[1], "r.")
                circ = plt.Circle((x_traj_t[0], x_traj_t[1]), radius=R, color='r', fill=False)
                ax.add_patch(circ)
            elif subname == "robot02":
                plt.plot(x_traj_t[0], x_traj_t[1], "g.")
                circ = plt.Circle((x_traj_t[0], x_traj_t[1]), radius=R, color='g', fill=False)
                ax.add_patch(circ)
            else:
                plt.plot(x_traj_t[0], x_traj_t[1], "b.")
                circ = plt.Circle((x_traj_t[0], x_traj_t[1]), radius=R, color='b', fill=False)
                ax.add_patch(circ)
        plt.pause(0.01)

    plt.pause(0.1)
    plt.close(fig)


## Global constants
Tf = 20
T0 = 0
T = 81
t_traj = np.linspace(T0, Tf, T)
dt = t_traj[1] - t_traj[0]
n = 4  ## number of states
m = 2  ## number of controls
trust_region = 0.25
max_iter = 1000
N_agents = 3  # Number of agents
robots_name = ["robot01", "robot02", "robot03"]
R = 2.5  # agent radius
## Specify the desired states
x_ini = {}
x_des = {}
count = 0
for name in robots_name:
    x_ini[name] = np.array([0, count * 5.1, 0, 0, 0, 0])  ## positions, velocities, and controls (acceleration)
    x_des[name] = np.array([14, (N_agents - count - 1) * 5, 0, 0, 0, 0])
    # x_des[name] = np.array([20, count * 5 , 0, 0, 0, 0])
    count += 1
# x_ini["robot02"] = np.array([5,3, 0, 0, 0, 0])
x_des["robot02"] = np.array([14.1, 5, 0, 0, 0, 0])
x_des["robot03"] = np.array([13.5, 0, 0, 0, 0, 0])
# x_des["robot01"] = np.array([10.1,8, 0, 0, 0, 0])

## get descrete LTI
[Ad, Bd] = descete_f(dt)

if __name__ == "__main__":
    ## Initialization (straight line)
    X_traj = x_initial(x_ini, x_des)

    ## Plotting initial traj
    # plot_traj(X_traj)

    ## Begin optimization loop
    for iter in range(max_iter):
        print(trust_region)
        X_traj = x_traj_opt(X_traj, trust_region)
        if iter % 1 == 0:
            plot_traj(X_traj)
        trust_region = trust_region / 3
