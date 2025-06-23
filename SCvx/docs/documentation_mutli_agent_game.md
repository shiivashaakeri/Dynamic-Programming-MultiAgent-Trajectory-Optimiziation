# Theoretical and Mathematical Foundations: Multi-Agent Trajectory Planning with Free Time and Non-Convex Problems Solved by Non-Cooperative Nash Game Wrapped Around Successive Convexification

## 1. Introduction to Multi-Agent Trajectory Planning
### 1.1. Problem Statement: Navigating Multiple Agents in a Shared Environment

The problem is that the trajectory generation for each agent, modeled as a unicycle, navigates from a specified initial state to a desired final state while adhering to its individual dynamics and control limits and avoiding both static obstacles and dynamic collisions with other agents.

### 1.2. Challenges: Non-Convexity and Free-Time Optimization

This problem presents significant computational challenges primarily due to:

**Non-Convexity**: Collision avoidance constraints (both with static obstacles and other agents) are inherently non-convex and define a non-convex exclusion zone.

**Free-Time Optimization**: The optimal duration for each agent's trajectory ($\sigma$) is often not fixed a priori, which adds another layer of non-linearity and complexity to the problem.

### 1.3. Proposed Approach: Non-Cooperative Nash Game with Successive Convexification

To overcome these challenges, a two-layer iterative optimization framework is used:

**Outer Loop**: Non-Cooperative Nash Game via Iterative Best Response: The multi-agent problem is formulated as a non-cooperative game. Each agent $i$ wants to minimize its cost function $\mathcal{J}_i$ , which depends on its trajectory and also the trajectories of all other agents $j\neq i$. A Nash Equilibrium is a state where no agent can unilaterally change its strategy (trajectory) to improve its own cost, given the strategies of all other agents. The Iterative Best Response algorithm is used to seek this equilibrium. In each iteration, each agent calculates its 'best response' (optimal trajectory), assuming the trajectories of all other agents are fixed from the previous iteration. This process repeats until the trajectories of all agents converge.

**Inner Loop**: Successive Convexification (SCvx): Each individual agent's best-response problem is a non-convex optimal control problem. SCvx is an iterative technique designed to solve such problems by sequentially solving a series of convex approximations. In each SCvx iteration, the non-linear dynamics and non-convex constraints are linearized around a current reference trajectory, transforming the problem into a convex Quadratic Program (QP). A trust-region constraint is added to ensure that the solution remains within the region where the linear approximations are valid.

The 'free-time' aspect is handled by introducing a time-scaling factor $\sigma$, which dictates the total duration of the trajectory. While a fully coupled free-time Nash game is complex, this implementation handles it by fixing $\sigma$ at a reference value within the Nash game's outer loop, effectively solving a fixed-time optimal control problem for each agent's best response subproblem.

## 2. Agent Dynamics and Control Model
### 2.1. Unicycle Kinematic Model

Each agent is modeled as a 2D unicycle, a common non-holonomic vehicle model.

#### 2.1.1. State Variables: $x,y,\theta$

The state of agent $i$ at time $t$ is defined by a vector $\mathbf{x}_i(t)\in \mathbb{R}^3$:

```math
\mathbf{x}_i(t) = \begin{bmatrix}
x_i(t) \\ y_i(t) \\ \theta_i(t)
\end{bmatrix},
```
where:

- $x_i(t),\, y_i(t)$ are the coordinates of the agent's position.

- $\theta_i(t)$ is the agent's heading (orientation) relative to the positive $x$-axis.

#### 2.1.2. Control Inputs: Linear Velocity ($v$), Angular Velocity ($\omega$)

The control input vector for agent $i$ at time $t$ is $\mathbf{u}_i(t)\in \mathbb{R}^2$:

```math
\mathbf{u}_i(t) = \begin{bmatrix}
v_i(t) \\ \omega_i(t)
\end{bmatrix},
```

where:

- $v_i(t)$ is the linear velocity of the agent.

- $\omega_i (t)$ is the angular velocity (rate of change of heading).

#### 2.1.3. Governing Equations of Motion

The continuous-time non-linear kinematic equations for the unicycle model are:

```math
\dot{\mathbf{x}}_i(t) = f(\mathbf{x}_i(t), \mathbf{u}_i(t)) = \begin{bmatrix} v_i(t)\,\cos{(\theta_i(t))}\\ v_i(t)\,\sin{(\theta_i(t))} \\ \omega_i(t) \end{bmatrix}
```

The UnicycleModel class uses symbolic computation (via sympy) to derive the Jacobian matrices of these dynamics with respect to the state and control inputs:
​	
```math
A(\mathbf{x},\mathbf{u})= \frac{\partial f}{\partial \mathbf{x}}=\begin{bmatrix} 0&0& -v\sin{(\theta)}\\ 0&0&v\cos{(\theta)} \\ 0&0&0 \end{bmatrix}
```

```math
B(\mathbf{x},\mathbf{u})= \frac{\partial f}{\partial \mathbf{u}}=\begin{bmatrix} v\cos{(\theta)}&0\\ v\cos{(\theta)}&0 \\ 0&1 \end{bmatrix}
```


### 2.2. Discretization Method: First-Order Hold

The continuous-time optimal control problem is discretized into a finite-dimensional optimization problem over $K$ discrete time points. FOH approximates the control inputs $u_k$ and $u_{k+1}$ as linearly varying over each time interval $[t_k t_{k+1}]$, and the state is integrated according to the non-linear dynamics.

For a general non-linear system $\dot{\mathbf{x}}=f(\mathbf{x},\mathbf{u})$, the FOH method linearizes the dynamics around a reference trajectory $(\mathbf{x}_{\text{ref},k}, \mathbf{u}_{\text{ref},k})$ to obtain the following discrete-time dynamics:

```math
x_{k+1} = \bar{\mathbf{A}}_{k} \mathbf{x}_k+ \bar{\mathbf{B}}_{k} \mathbf{u}_k + \bar{\mathbf{C}}_{k} \mathbf{u}_{k+1}+ \bar{\mathbf{S}}_{k} \sigma + \bar{\mathbf{z}}_k + \nu_k
```

where:

- $x_k$ and $u_k$ are the state and control variables at time step $k$.

- $\sigma$ is the total trajectory duration (time scaling factor).

- $\nu_k \in \mathbb{R}^3$ is the *defect variable* at time step $k$. This variable is introduced in SCvx to turn the equality constraint into an inequality that penalizes deviations from the linearized dynamics, enabling the problem to remain convex even if the linearization is imperfect. In the context of the SCProblem, it represents a "virtual control" that allows for violations of the linearized dynamics, which are then penalized in the objective.

- $\bar{\mathbf{A}}_{k}, \bar{\mathbf{B}}_{k}, \bar{\mathbf{C}}_{k}, \bar{\mathbf{S}}_{k}, \bar{\mathbf{z}}_{k}$ are the discretization matrices.

## 3. Optimal Control Problem Formulation for a Single Agent
Within the SCvx framework, each agent solves a convex optimal control problem. This problem is defined by an objective function and a set of constraints.

### 3.1. Objective Function (Cost Functional)

The objective for each agent $i$ is to minimize a sum of weighted terms. The total objective for agent $i$ in a given SCvx iteration can be expressed as:

```math
\min_{\mathbf{X}_i,\mathbf{U}_i, \nu_i, \mathbf{s}'_i} \, \mathcal{J}_i(\mathbf{X}_i,\mathbf{U}_i, \nu_i, \mathbf{s}'_i, \sigma\, \mid \, \mathbf{X}_{\text{ref},i}, \mathbf{U}_{\text{ref},i}, \mathbf{X}_{\text{prev},i}, \mathbf{X}_{\text{neigh},\text{curr}}, \mathbf{X}_{\text{neigh},\text{prev}})
```

#### 3.1.1. Control Effort Minimization

This term penalizes the magnitude of the control inputs, promoting energy efficiency and smoother control actions.

```math
\mathcal{J}_{\text{ctrl}} = w_{\text{ctrl}} \sum_{k=0}^{K-1} ||\mathbf{u}_{i,k}||_2^2
```

#### 3.1.2. Control Rate Smoothing

This term penalizes large changes in control inputs between consecutive time steps, leading to smoother, less "jerky" control profiles.
​	
```math
\mathcal{J}_{\text{ctrl\_rate}} \sum_{k=0}^{K-2} ||\mathbf{u}_{i,k+1} - \mathbf{u}_{i,k}||_2^2
```

#### 3.1.3. Curvature Smoothing

This term penalizes sharp turns by minimizing changes in the agent's heading angle.

```math
\mathcal{J}_{\text{curv}} = w_{\text{curv}} \sum_{k=0}^{K-2} (\theta_{i,k+1} - \theta_{i,k})^2
```

#### 3.1.4. Inertia Regularization

This term penalizes deviations of the current trajectory from the reference trajectory used for linearization (specifically, the trajectory from the previous outer Nash iteration). 
​	
```math
\mathcal{J}_{\text{inertia}} = w_{\text{inertia}} \sum_{k=0}^{K-1} ||\mathbf{x}_{i,k} - \mathbf{x}_{\text{prev}, i,k}||_2^2
```


#### 3.1.5. Obstacle Slack Penalty

For linearized static obstacle avoidance, slack variables $s'_{j,k}$ are introduced to relax the hard constraints and ensure feasibility in cases where strict avoidance might be impossible or overly restrictive. These slack variables are penalized in the objective.

```math
\mathcal{J}_{\text{slack}} = w_\text{slack} \sum_{j=1}^{N_{\text{obs}}} \sum_{k=0}^{K-1} s'_{j,k}
```

#### 3.1.6. Defect Penalty

The defect variables $\nu_k$ are penalized to encourage the linearized dynamics to accurately represent the true non-linear dynamics.

```math
\mathcal{J}_{\text{defect}} = w_{\nu} \sum_{k=0}^{K-2} ||\nu_k||_1
```

#### 3.1.7. Time Penalty

The time scaling factor $\sigma$ is penalized to encourage faster trajectories, subject to other constraints.

```math
\mathcal{J}_{\text{time}} = w_{\sigma} \sigma
```

#### 3.1.8. Overall Cost Functional

The total objective function for agent $i$'s best-response problem within the SCvx framework is the sum of these terms:


```math
\mathcal{J}_i^{\text{total}} = \mathcal{J}_{\text{ctrl}}+\mathcal{J}_{\text{ctrl\_rate}} + \mathcal{J}_{\text{curv}} + \mathcal{J}_{\text{inertia}}+ \mathcal{J}_{\text{slack}}+ \mathcal{J}_{\text{defect}}+ \mathcal{J}_{\text{time}}
```
​	
 
### 3.2. Constraints

The optimal control problem for each agent is subject to several constraints:

#### 3.2.1. Initial and Final State Constraints

The agent's trajectory must start at a specified initial state and end at a desired final state:

```math
\mathbf{x}_{i,0} = \mathbf{x}_{\text{init},i} \\ \mathbf{x}_{i,K-1} = \mathbf{x}_{\text{final},i}
```
Additionally, initial and final controls are set to zero for smooth start/stop: 
```math
\mathbf{u}_{i,0} = \mathbf{0} \\ \mathbf{u}_{i,K-1} = \mathbf{0}
```


#### 3.2.2. Dynamics Constraints (Equality Constraints)

The discrete-time dynamics derived from the FOH linearization must be satisfied at each time step, with the inclusion of the defect variable $\nu$ 

```math
\mathbf{x}_{i,k+1} = \bar{\mathbf{A}}_{k} \mathbf{x}_{i,k}+ \bar{\mathbf{B}}_{k} \mathbf{u}_{i,k} + \bar{\mathbf{C}}_{k} \mathbf{u}_{i,{k+1}}+ \bar{\mathbf{S}}_{k} \sigma + \bar{\mathbf{z}}_k + \nu_{i,k}, \quad  \text{for } k=0,\dots,K−2
```

#### 3.2.3. Control Input Limits

The control inputs (linear and angular velocities) are bounded:

```math
0 \leq v_{i,k} \leq v_{\text{max},i}\\
-\omega_{\text{max},i} \leq \omega_{i,k} \leq \omega_{\text{max},i}
```

#### 3.2.4. State Bounds (Spatial)

The agent's position must remain within defined spatial boundaries:

```math
\text{lb} + R_{\text{robot}} \leq x_{i,k} \leq \text{ub} - R_{\text{robot}}\\
\text{lb} + R_{\text{robot}} \leq y_{i,k} \leq \text{ub} - R_{\text{robot}}
```
is the agent's physical radius.

#### 3.2.5. Static Obstacle Avoidance (Linearized)

Static obstacles are circular. The non-convex constraint for avoiding obstacle $j$ with center $\textbf{p}_{\text{obs},j}$
and radius $R_{\text{obs},j}$ is:

```math
||\mathbf{p}_{i,k} - \mathbf{p}_{\text{obs},k}||_2 \geq R_{\text{obs},j} + R_{\text{robot},i}
```

where $\textbf{p}_{i,k} = [x_{i,k},\, y_{i,k}]^\top$
 is the center of the agent.
This non-convex constraint is linearized around the reference position $\textbf{x}_{\text{ref}, i,k}$ from the previous SCvx iteration and includes a non-negative slack variable $s'_{j,k}$

```math
\hat{\textbf{n}}_{j,k}^\top (\textbf{p}_{i,k} - \textbf{p}_{\text{obs},j}) \geq (R_{\text{obs},j} +  R_{\text{robot},i}) - s'_{j,k}
```

where  
$\hat{\textbf{n}}_{j,k}$ is the normalized vector from the obstacle center to the reference position of agent $i$:
​
```math
\hat{\textbf{n}}_{j,k}=\frac{\textbf{x}_{\text{ref}, i,k}[0:2] - \mathbf{p}_{\text{obs},j}}{||\textbf{x}_{\text{ref}, i,k}[0:2] - \mathbf{p}_{\text{obs},j}||_2 + \epsilon}
```

The $\epsilon$ term prevents division by zero.


## 4. Successive Convexification (SCvx) Algorithm
The SCProblem encapsulates the convex optimization problem solved in each iteration of the SCvx algorithm.

### 4.1. Iterative Approximation of Non-Convex Problems

SCvx is an iterative technique that approximates a non-linear, non-convex optimal control problem by a sequence of convex Quadratic Programs (QPs). Each QP is formed by:

1. Linearizing the non-linear dynamics and non-convex constraints around a current reference trajectory.

2. Adding a trust-region constraint to ensure that the solution remains close to the linearization point, where the approximations are valid.

3. Penalizing the "defect" in the linearized dynamics.

### 4.2. Convex Quadratic Program (QP) Structure

The SCProblem constructs a QP for agent $i$ in each SCvx iteration. The decision variables are the state trajectory $\mathbf{X}_i$, control trajectory $\mathbf{U}_i$, defect variables $\nu_i$, and the time scaling factor $\sigma_i$.

#### 4.2.1. Objective Function in QP Form

The overall objective function, as described in Section 3.1.8, is convex and composed of quadratic terms and linear terms. This structure ensures the problem is a QP.

#### 4.2.2. Linear Equality and Inequality Constraints

The constraints defined in Section 3.2 are all formulated as linear equality or inequality constraints on the decision variables.

- Dynamics Constraints: The discrete-time dynamics (Section 3.2.2) are linear equality constraints in $\mathbf{X}_i, \mathbf{U}_i, \nu_i, \sigma$

- Boundary Conditions: Initial/final states and controls (Section 3.2.1) are linear equality constraints.

- Input/State Bounds: These (Sections 3.2.3, 3.2.4) are linear inequality constraints.

- Static Obstacle Constraints: The linearized obstacle avoidance (Section 3.2.5) with slack variables forms linear inequality constraints.

- Inter-Agent Collision Constraints: The linearized inter-agent collision avoidance (Section 4.2) also forms linear inequality constraints.

### 4.3. Trust Region Mechanism

A crucial component of SCvx is the trust region constraint, which limits the allowable deviation of the current solution from the linearization reference point. This ensures that the linear approximations remain accurate.
The trust region is typically an $L_1$ norm constraint on the deviation of states, controls, and time scaling from their reference values:

```math
||\mathbf{X}_i - \mathbf{X}_{\text{ref},i}||_1+||\mathbf{U}_i - \mathbf{U}_{\text{ref},i}||_1 + |\sigma_i - \sigma_{\text{ref},i}| \leq \Delta 
```
where
​	
- $\mathbf{X}_{\text{ref},i}, \mathbf{U}_{\text{ref},i}, \sigma_{\text{ref},i}$ are the state, control, and time scaling reference from the previous SCvx iteration.

- $\Delta$ is the trust region radius.


## 5. Non-Cooperative Nash Game Framework
The highest level of the algorithm is the non-cooperative Nash game, solved using an Iterative Best Response strategy.

### 5.1. Fundamentals of Game Theory and Nash Equilibrium

A non-cooperative game involves multiple players, each making decisions to optimize their own objective function, given the decisions of others. A Nash Equilibrium is a set of strategies such that no player can improve its own objective by unilaterally changing its strategy, assuming all other players' strategies remain fixed.

### 5.2. Iterative Best Response Algorithm

The NashSolver class implements the Iterative Best Response algorithm to find a Nash Equilibrium. The process is as follows:

#### 5.2.1. Definition of Best Response for Agent $i$

For agent $i$, its best response $(\mathbf{X}_i^*,\mathbf{U}_i^*)$ is the solution to its optimal control problem:

```math
(\mathbf{X}_i^*,\mathbf{U}_i^*) = \arg\min_{\mathbf{X}_i,\mathbf{U}_i} \mathcal{J}_i (\mathbf{X}_i,\mathbf{U}_i \mid \{\mathbf{X}_j,\mathbf{U}_j\}_{j\neq i, \text{fixed}})
```

This means agent $i$ finds its optimal trajectory while the trajectories of all other agents $j\neq i$ are fixed parameters.

#### 5.2.2. Sequential Update Strategy

The $\texttt{NashSolver.solve()}$ method uses a sequential update strategy:

In each outer Nash iteration, agents are processed one by one (from $i=0 $to $N−1$).

When agent $i$ solves its best response, it uses the most recently updated trajectories of its neighbors. This means if agent $j$ (where $j<i$) has already updated its trajectory in the current outer iteration, agent $i$ will react to agent $j$'s new trajectory. If agent $l$ (where $l>i$) has not yet updated, agent $i$ will react to agent $l$'s trajectory from the previous outer iteration.

After agent $i$ computes its new best response $(\mathbf{X}_{\text{new},i},\mathbf{U}_{\text{new},i})$, its current trajectories are immediately updated.

#### 5.2.3. Role of Other Agents' Fixed Trajectories

For agent $i$ to solve its best response, the trajectories of other agents $j\neq i$ must be treated as known. 

### 5.3. Handling Free-Time Optimization via Fixed $\sigma$

The concept of "free time" implies that the total duration of the maneuver is also optimized. In this specific implementation, $\texttt{sigma\_ref}$ is passed into $\texttt{NashSolver.solve()}$ as a fixed initial value for the entire Nash game.
Crucially, within each agent's best-response problem (in $\texttt{AgentBestResponse.setup}$), the decision variable $\texttt{self.scp.var["sigma"]}$ is explicitly pinned to this sigma_ref value via the constraint:

```math
\sigma_i = \sigma_{\text{ref}}
```

This simplifies each best-response subproblem to a fixed-time optimal control problem. While the code provides the mechanism for a time-scaling variable, the overall NashSolver itself does not adapt sigma iteratively to find an "optimal" free time across the game; it finds a Nash Equilibrium for the given sigma_ref. This implies that the overall free-time optimization would need to be handled by an even outer-most loop not present in the provided code, or sigma_ref is considered a hyperparameter chosen externally.

## 6. Overall Algorithm: Nash Game Wrapped SCvx
The complete algorithm operates as a nested iterative process:

### 6.1. Outer Loop: Iterative Best Response for Nash Equilibrium


**Inputs**: Initial guess trajectories for all agents $(\mathbf{X}_{\text{refs}}, \mathbf{U}_{\text{refs}})$, and a reference total time $\sigma_{\text{ref}}$
​	

**Process**:

1. Initialization: Set $\mathbf{X}_{\text{curr}}= \mathbf{X}_{\text{refs}}$, $\mathbf{U}_{\text{curr}}= \mathbf{U}_{\text{refs}}$

2. Iterate for $\text{it}=0,…,\text{max\_iter}−1$:

  - Record $\mathbf{X}_{\text{prev\_all}}$ as a copy of $\mathbf{X}_{\text{curr}}$ from the beginning of this outer iteration. This is used for collision linearization.

 - $\Delta$ is initialized to $0$.

 - For each agent $i=0,\dots ,N−1$ (sequential update):

    - Retrieve discretized dynamics matrices $(\bar{\mathbf{A}}, \dots,\bar{\mathbf{z}})$ for agent $i$ based on its current $\mathbf{X}_{\text{curr}}[i], \mathbf{U}_{\text{curr}}[i]$

    - Prepare neighbor references: current neighbor ($\mathbf{X}_{\text{curr}}$ for $j\neq i$) and previous neighbor ($\mathbf{X}_{\text{prev}}$ for $j\neq i$).

    - Call $\texttt{AgentBestResponse.setup()}$: Formulate agent $i$'s convex QP using the current references, the fixed $\sigma_{\texttt{ref}}$, the linearized dynamics, and the current/previous neighbor trajectories. This step sets up the SCvx problem for agent $i$.

    - Call $\texttt{AgentBestResponse.setup()}$: Solve the convex QP.

    - Update $\mathbf{X}_{\text{curr}}, \mathbf{U}_{\text{curr}}$ with the newly found best response.

    - Calculate the change: $\Delta = ||\mathbf{X}_{\text{new},i}- \mathbf{X}_{\text{old},i}||$, update $\Delta$

    - Convergence Check: If $\Delta < \text{tol}$, break the loop.

### 6.2. Inner Loop: Successive Convexification for Each Agent's Best Response

Each call to $\texttt{AgentBestResponse.solve()}$ internally triggers the SCvx iterations (controlled by the $\texttt{SCProblem}$ class). This inner loop is abstracted away in the provided $\texttt{AgentBestResponse.solve()}$ code, which assumes $\texttt{SCProblem.solve()}$ handles the full SCvx process. In a complete SCvx implementation, $\texttt{SCProblem.solve()}$ would be called iteratively, with $X_\text{ref}, U_\text{ref}$, and $\Delta$ being updated based on the solution quality, until the inner SCvx problem converges. For this code, it appears $\texttt{SCProblem.solve()}$ refers to a single pass through the convex solver with a fixed $\Delta$, which is a common pattern when SCvx is handled by an external loop, but here it's implicitly handled by the cvxpy solver itself once parameters are set.

### 6.3. Convergence Criteria and Termination Conditions

The outer Nash game loop terminates when one of two conditions is met:

- Convergence: The maximum Euclidean norm of the difference between the current and previous trajectories across all agents falls below a specified tolerance.

- Maximum Iterations: The number of outer Nash iterations reaches $\text{max\_iter}$.


## 7. Experimental Results

This section presents the setup and outcomes of a Nash-game-based trajectory optimization involving three non-cooperative agents in a shared workspace with static obstacles. Each agent solves a convexified local problem using SCvx, coordinated through an iterative best-response loop.

---

### 7.1 Agent Definitions (Init/Goal, Weights)

The table below summarizes the agent settings:

| Agent | Initial State (x, y, θ) | Final State (x, y, θ) | Control Weight \($w_u$\) | Collision Weight \($w_{\text{coll}}$\) | Radius \($r_{\text{coll}}$\) | Control Rate \($w_{\dot{u}}$\) | Curvature \($w_{\kappa}$\) |
|-------|--------------------------|------------------------|------------------------|------------------------------|------------------------|-----------------------------|----------------------|
| 0     | (0.0, 0.0, 0.0)          | (2.0, 2.0, 0.0)        | 100.0                 | 10.0                         | 0.5                    | 5.0                         | 5.0                  |
| 1     | (2.0, 0.0, 0.0)          | (0.0, 2.0, 0.0)        | 100.0                 | 10.0                         | 0.5                    | 5.0                         | 5.0                  |
| 2     | (1.0, 0.1, 0.0)          | (1.0, 1.9, 0.0)        | 100.0                 | 10.0                         | 0.5                    | 5.0                         | 5.0                  |

- These parameters are injected into each agent's model via `GameUnicycleModel`.
- The third agent moves vertically, navigating the obstacle from below to above.

---

### 7.2 Obstacle Settings and Clearance

- A circular obstacle is placed at \((1.0, 1.0)\) with a radius of \(0.25\).
- A clearance buffer of \(\varepsilon = 0.05\) is added around obstacles during warm-start path generation.
- Minimum inter-agent distance is enforced at \(D_{\min} = 0.5\, \text{m}\) via hard linearized constraints.

---

### 7.3 Warm-Start Trajectory Generation

- Each agent is initialized with a **piecewise linear path** from start to goal.
- These paths are lifted into full state-space trajectories using constant heading and evenly distributed waypoints.
- Obstacle-aware inflation helps guide the initial guess outside infeasible zones.

---

### 7.4 Final Trajectory Results

<p align="center">
  <img src="img/Figure_trajectory_game.png" width="500"/>
  <br/>
  <em>Figure 7.1: Final trajectories of three agents in the Nash-game scenario. Trajectories respect clearance and reach their goals without collision.</em>
</p>

---

### 7.5 Convergence of Nash Solver

<p align="center">
  <img src="img/Figure_cnvrg_game.png" width="500"/>
  <br/>
  <em>Figure 7.2: Convergence plot of the Nash iterative best-response solver. The norm of the change in state trajectory \(\|\Delta X\|\) drops below threshold.</em>
</p>

---

### 7.6 Animation of Agent Motions

<p align="center">
  <img src="img/game_trajectory_animation.gif" width="500"/>
  <br/>
  <em>Figure 7.3: Animated agent motions through obstacle-laden workspace, generated from the final SCvx solution trajectories.</em>
</p>

## 8. Comparison

### Summary Table

| Method | Iter | Time (s)  | Min-sep (m) | Effort   | Length  | CtrlSmooth | CurvSmooth | SlackSum |
|:-------|-----:|----------:|------------:|---------:|--------:|-----------:|-----------:|---------:|
| ADMM   |   20 |    26.4579 |      0.5000 |   91.0586 |   9.5391 |     6.4686 |     0.0059 |     0.0 |
| NASH   |    6 |     7.5691 |      0.5665 |    2.9906 |   9.6735 |     0.0940 |     0.0000 |     0.0 |

---

### Convergence Profiles

<p align="center">
  <img src="img/compare_admm_nash_convergence.png" width="500"/>
  <br/>
  <em>Figure 8.1: Convergence profiles—ADMM primal/dual residuals vs. iteration and Nash ‖ΔX‖ vs. iteration.</em>
</p>

### Control Effort & Path Length

<p align="center">
  <img src="img/compare_admm_nash_control_pathlength.png" width="500"/>
  <br/>
  <em>Figure 8.2: Total control effort and path length for ADMM vs. Nash.</em>
</p>

### Trajectory Comparison

<p align="center">
  <img src="img/compare_admm_nash_trajectories.png" width="500"/>
  <br/>
  <em>Figure 8.3: Overlaid agent trajectories (solid = Nash, dashed = ADMM; same color per agent).</em>
</p>

### Inter‐Agent Separation Over Time

<p align="center">
  <img src="img/compare_admm_game_interseperation.png" width="500"/>
  <br/>
  <em>Figure 8.4: Minimum inter‐agent distance d<sub>min</sub>(k) vs. timestep for ADMM and Nash.</em>
</p>