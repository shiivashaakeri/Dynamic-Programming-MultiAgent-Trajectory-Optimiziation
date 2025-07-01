import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp
from numpy import linalg as LA
from scipy import signal


## dynamics
def descete_f(dt):
    A = np.array([[0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 1],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0]])
    B = np.array([[0, 0, 0],
                  [0, 0, 0],
                  [0, 0, 0],
                  [1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]])
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
        r_all[name] = np.ones((T * (n + m), 1)) * 0.5
        s_val[name] = np.zeros((T, n + m))
        s_bar_val[name] = np.zeros((T, n + m))
        s_bar_val_new[name] = np.zeros((T, n + m)) + 1
        diff += LA.norm(s_bar_val[name] - s_bar_val_new[name], 2)
    ## ADMM loop
    rho = 1000  ## Lagrangian penalty
    # while diff > tol:
    for iter in range(1):
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
            s_i_vec = cp.reshape(s_i, (T * (n + m), 1), order="C")  ## vectorized primal variables
            d_i = s_i[0:T, 0:n]
            w_i = s_i[0:T - 1, n:n + m]
            # Duplicate variables
            s_bar_val_i = s_bar_val[name]
            s_bar_val_i_vec = np.reshape(s_bar_val_i, (T * (n + m), 1), order="C")  ## vectorized duplicated variables
            ##########################################################
            ## s - minimization (primal variables)
            # Construct the augmented Lagrangian
            # L_rho = 1*cp.sum_squares(u_traj_i + w_i) + r_i.T @ (s_i_vec - s_bar_val_i_vec) + rho / 2 * cp.square(
            #     cp.norm(s_i_vec - s_bar_val_i_vec, 2))
            L_rho = 1 * cp.sum_squares(u_traj_i + w_i) + 10000 * cp.norm(S_i, 1)
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

                ## boundary constraints
                constraints_s.append(x_traj_t[0] + d_t[0] <= 22)
                constraints_s.append(x_traj_t[0] + d_t[0] >= -1)
                constraints_s.append(x_traj_t[1] + d_t[1] >= -1)
                constraints_s.append(x_traj_t[1] + d_t[1] <= 20)

                # loop through obstacles (test use)
                S_t = S_i[t]
                for obs_name in robots_name:
                    X_traj_j = X_traj[obs_name]
                    x_traj_j = X_traj_j[0:T, 0:n]
                    x_traj_j_t = x_traj_j[t, :]
                    if obs_name != name:  ## exclude itself
                        S = 2 * R - LA.norm(x_traj_t[0:3] - x_traj_j_t[0:3], 2)
                        S_grad = (x_traj_t[0:3] - x_traj_j_t[0:3]).T / cp.norm(x_traj_t[0:3] - x_traj_j_t[0:3], 2)

                        # S = R ** 2 -  LA.norm(x_traj_t[0:2] - x_traj_j_t[0:2], 2) ** 2
                        # S_grad = 2 * (x_traj_t[0:2] - x_traj_j_t[0:2]).T
                        constraints_s.append(
                            S - S_grad @ d_t[0:3] <= S_t
                        )
                        constraints_s.append(S_t >= 0)

            problem = cp.Problem(cp.Minimize(L_rho), constraints_s)
            problem.solve(solver=cp.CLARABEL)
            s_val[name] = s_i.value

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


def cost_fcn(X_traj):
    cost_iter = 0
    for name in robots_name:
        X_traj_i = X_traj[name]
        u_traj_i = X_traj_i[0:T - 1, n:n + m]
        for t in range(T - 1):
            cost_iter += LA.norm(u_traj_i[t, :], 2) ** 2
    return cost_iter


## Plotting
def plot_traj(X_traj):

    # Create a sphere parameters
    phi, theta = np.linspace(0, np.pi, 20), np.linspace(0, 2 * np.pi, 20)
    phi, theta = np.meshgrid(phi, theta)

    # Create the 3D plot
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ## plotting the time traj
    for t in range(T):
        ax.clear()
        for name in robots_name:
            X_traj_i = X_traj[name]
            ax.plot(X_traj_i[:, 0], X_traj_i[:, 1], X_traj_i[:, 2])  ## plotting entire trajectories
            x_i = R * np.sin(phi) * np.cos(theta) + X_traj_i[t, 0]
            y_i = R * np.sin(phi) * np.sin(theta) + X_traj_i[t, 1]
            z_i = R * np.cos(phi) + X_traj_i[t, 2]
            # Plot the sphere
            ax.plot_surface(x_i, y_i, z_i, color='cyan', alpha=0.3, edgecolor='none')

        # Set the limits
        ax.set_xlim([-5, 20])
        ax.set_ylim([-5, 20])
        ax.set_zlim([0, 20])
        plt.pause(0.1)
    plt.close(fig)


    # fig, ax = plt.subplots()
    # ax.clear()
    # for t in range(T):
    #     ax.clear()
    #     ax.set_xlim(-1, 21)
    #     ax.set_ylim(-1, 21)
    #     for subname in robots_name:
    #         x_traj_i = x_traj[subname]
    #         ax.plot(x_traj_i[:, 0], x_traj_i[:, 1])
    #         x_traj_t = x_traj_i[t, :]
    #         if subname == "robot01":
    #             # plt.plot(x_traj_t[0], x_traj_t[1], "r.")
    #             ax.plot(x_traj_t[0], x_traj_t[1], "r.")
    #             circ = plt.Circle((x_traj_t[0], x_traj_t[1]), radius=R, color='r', fill=False)
    #             ax.add_patch(circ)
    #         elif subname == "robot02":
    #             plt.plot(x_traj_t[0], x_traj_t[1], "g.")
    #             circ = plt.Circle((x_traj_t[0], x_traj_t[1]), radius=R, color='g', fill=False)
    #             ax.add_patch(circ)
    #         else:
    #             ax.plot(x_traj_t[0], x_traj_t[1], "b.")
    #             circ = plt.Circle((x_traj_t[0], x_traj_t[1]), radius=R, color='b', fill=False)
    #             ax.add_patch(circ)
    #     plt.pause(0.01)
    # plt.pause(0.1)
    # plt.close(fig)


## Global constants
Tf = 30
T0 = 0
T = 51
t_traj = np.linspace(T0, Tf, T)
dt = t_traj[1] - t_traj[0]
n = 6  ## number of states
m = 3  ## number of controls
trust_region = 0.25
max_iter = 1000
N_agents = 3  # Number of agents
robots_name = ["robot01", "robot02", "robot03"]
R = 2.3  # agent radius
## Specify the desired states
x_ini = {}
x_des = {}
count = 0
for name in robots_name:
    x_ini[name] = np.array([0, count * 5.1, 10,
                            0, 0, 0,
                            0, 0, 0])  ## positions(x,y,z), velocities(u,v,w), and controls(ax,ay,az) (acceleration)
    x_des[name] = np.array([14, (N_agents - count - 1) * 5, 10 + count * 1,
                            0, 0, 0,
                            0, 0, 0])
    # x_des[name] = np.array([20, count * 5 , 0, 0, 0, 0])
    count += 1
# x_ini["robot02"] = np.array([5,3, 0, 0, 0, 0])
# x_des["robot02"] = np.array([14.1, 5, 0, 0, 0, 0])
# x_des["robot03"] = np.array([13.5, 0, 0, 0, 0, 0])
# x_des["robot01"] = np.array([10.1,8, 0, 0, 0, 0])

## get descrete LTI
[Ad, Bd] = descete_f(dt)
## Cost list
cost_list = np.zeros(max_iter)
if __name__ == "__main__":
    ## Initialization (straight line)
    X_traj = x_initial(x_ini, x_des)

    ## Plotting initial traj
    plot_traj(X_traj)

    ## Begin optimization loop
    for iter in range(max_iter):
        print(trust_region)
        X_traj = x_traj_opt(X_traj, trust_region)
        if iter % 1 == 0:
            plot_traj(X_traj)
        # trust_region = trust_region / 2
        cost_list[iter] = cost_fcn(X_traj)
        print("Actual cost: ", cost_list[iter])
        if iter >= 1:
            if cost_list[iter] > cost_list[iter - 1]:
                trust_region = trust_region / 2
