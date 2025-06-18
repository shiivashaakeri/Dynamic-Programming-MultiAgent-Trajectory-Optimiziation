# 1. Introduction

## 1.1 Problem Motivation

In multi-agent systems such as autonomous drone swarms, mobile robots, or satellite constellations, each agent must plan a trajectory to reach its goal without collisions and while respecting its own dynamics. Centralized methods often scale poorly with the number of agents due to:
- The curse of dimensionality in the joint state space.
- The communication bottleneck between agents and a central planner.
- Lack of privacy or autonomy for independently governed agents.

A more scalable and decentralized alternative is to treat the problem as a non-cooperative game, where each agent solves its own trajectory optimization by considering the others’ strategies as fixed.

⸻

## 1.2 Nash Games in Multi-Agent Motion Planning

We model the interaction as a Nash game over the space of control trajectories:
- Let agent $i \in \{1, \dots, N\}$ have control input $u_i(t)$ and state $x_i(t)$.
- Each agent minimizes its own cost functional:
```math
J_i(x_i, u_i; \{x_j\}_{j \neq i}) = \int_0^T \ell_i(x_i(t), u_i(t)) \, dt + \Phi_i(x_i(T))
```

subject to dynamics:
$\dot{x}_i = f_i(x_i, u_i), \quad x_i(0) = x_i^0$
and collision constraints:
```math
\|x_i(t) - x_j(t)\| \geq d_{\min}, \quad \forall j \neq i, \; \forall t \in [0, T]
```

In this formulation, agents simultaneously minimize their cost while reacting to the others’ planned trajectories, which leads to a Nash equilibrium:
```math
\forall i,\quad (x_i^*, u_i^*) = \arg\min_{x_i, u_i} J_i(x_i, u_i; \{x_j^*\}_{j \neq i})
```
⸻

## 1.3 Challenges with Decentralization and Coupling

While Nash games offer decentralization, they introduce several mathematical and algorithmic challenges:
- Coupling through constraints: Collision avoidance inherently couples agents’ decision spaces.
- Non-convexity: The dynamics and constraints make the optimization landscape non-convex.
- Convergence to equilibria: Iterative best-response updates do not always converge and can oscillate or diverge.

To address these, we embed each agent’s optimization within a Sequential Convex Programming (SCvx) framework and coordinate agents through Iterative Best Response (IBR) updates. Our method leverages convexification of nonlinear constraints and trust regions to ensure stability and convergence.

# 2. Mathematical Formulation

## 2.1 Agent Dynamics (Unicycle Model)

Each agent $i \in \{1, \dots, N\}$ is modeled using the unicycle dynamics:

```math
\dot{x}_i = v_i \cos(\theta_i), \quad
\dot{y}_i = v_i \sin(\theta_i), \quad
\dot{\theta}_i = \omega_i
```

or in vector form:

```math
\dot{r}_i = f(r_i, u_i), \quad
\text{where } r_i = \begin{bmatrix} x_i \\ y_i \\ \theta_i \end{bmatrix}, \quad
u_i = \begin{bmatrix} v_i \\ \omega_i \end{bmatrix}
```

This is discretized using First-Order Hold (FOH) over K time steps for use in trajectory optimization.

## 2.2 Cost Terms per Agent

Each agent i minimizes a convexified objective composed of weighted cost terms:

```math
J_i = J_{\text{ctrl}} + J_{\text{rate}} + J_{\text{curv}} + J_{\text{inertia}}
```

### 2.2.1 Control Effort

```math
J_{\text{ctrl}} = w_{\text{ctrl}} \sum_{k=0}^{K-1} \| u_i(k) \|^2
```

This penalizes aggressive control inputs (both linear and angular velocity).

### 2.2.2 Control Rate Smoothing

```math
J_{\text{rate}} = w_{\text{rate}} \sum_{k=0}^{K-2} \| u_i(k+1) - u_i(k) \|^2
```

Promotes smooth changes in control inputs across the trajectory.

### 2.2.3 Heading Curvature

```math
J_{\text{curv}} = w_{\text{curv}} \sum_{k=0}^{K-2} \left( \theta_i(k+1) - \theta_i(k) \right)^2
```

Reduces rapid heading changes and encourages natural turning behavior.

### 2.2.4 Inertia Regularization

```math
J_{\text{inertia}} = w_{\text{inertia}} \sum_{k=0}^{K-1} \| r_i(k) - r_i^{\text{prev}}(k) \|^2
```

Regularizes trajectory updates by discouraging large deviations from previous iterations.

## 2.3 Hard Collision Constraints

To ensure safe distances between agents i and j, we linearize the collision constraint around previous trajectories:

```math
\hat{n}_{ij}^{(k)} = \frac{r_i^{\text{prev}}(k) - r_j^{\text{prev}}(k)}{\|r_i^{\text{prev}}(k) - r_j^{\text{prev}}(k)\|}
```

```math
\hat{n}_{ij}^{(k)}{}^\top \left( r_i(k) - r_j(k) \right) \geq d_{\text{min}}, \quad \forall k = 0, \dots, K-1
```

This constraint is added as a hard constraint to each agent’s optimization problem, ensuring minimum separation $d_{\text{min}}$.

## 2.4 Nash Equilibrium Definition (Best-Response Fixed Point)

We define the Nash equilibrium as a fixed point of iterative best responses:

```math
\left\{ (r_i^*, u_i^*) \right\}_{i=1}^N \quad \text{such that} \quad
(r_i^*, u_i^*) = \arg \min_{r_i, u_i} J_i(r_i, u_i; \{r_j^*\}_{j \ne i})
```

Each agent optimizes its own cost, treating other agents’ trajectories as fixed. Iterating this best-response process leads to convergence under suitable conditions (e.g., convex cost, smooth constraints, trust regions).

# 3. Game-Specific Model Extensions

This section describes how agent-specific modeling and interaction terms are incorporated into the mathematical formulation of a non-cooperative multi-agent trajectory game.

⸻

## 3.1 Per-Agent Cost Structure

Each agent $i \in \{1, \dots, N\}$ is modeled with its own dynamics and a personalized cost function, leading to a non-cooperative game formulation. The decision variables for agent $i$ are its state $X_i \in \mathbb{R}^{n \times K}$ and control$ U_i \in \mathbb{R}^{m \times K}$, over a time horizon of $K$ steps.

The cost function for agent $i$ is defined as:
```math
J_i(X_i, U_i; X_{-i}) = J_{\text{ctrl},i} + J_{\text{rate},i} + J_{\text{curv},i} + J_{\text{inertia},i}
```

where each term is:
- Control Effort:
```math
J_{\text{ctrl},i} = w_{\text{ctrl}}^i \sum_{k=0}^{K-1} \| u_i(k) \|^2
```

- Control Rate Smoothing:
```math
J_{\text{rate},i} = w_{\text{rate}}^i \sum_{k=0}^{K-2} \| u_i(k+1) - u_i(k) \|^2
```

- Heading Curvature (for orientation $\theta_i$):
```math
J_{\text{curv},i} = w_{\text{curv}}^i \sum_{k=0}^{K-2} \left( \theta_i(k+1) - \theta_i(k) \right)^2
```

- Inertia Regularization (distance to previous trajectory):
```math
J_{\text{inertia},i} = w_{\text{inertia}}^i \sum_{k=0}^{K-1} \| X_i(k) - X_i^{\text{prev}}(k) \|^2
```

Here, $w_{\text{ctrl}}^i, w_{\text{rate}}^i, \ldots$ are agent-specific weights, and $X_i^{\text{prev}}$ is the previous iterate of the agent’s trajectory.

## 3.2 Game-Theoretic Coupling via Constraints

Agents are coupled through collision avoidance constraints, which are applied as hard constraints in the optimization problem.

For every pair of distinct agents $i \ne j$, a minimum separation constraint is enforced. Let $p_i(k) \in \mathbb{R}^2$ denote the position of agent $i$ at timestep $k$. The linearized constraint for each timestep $k$ is:
1. Relative position (frozen at previous iterates):
$d_{ij}^{(k)} = p_i^{\text{prev}}(k) - p_j^{\text{prev}}(k)$
2. Unit normal vector:
$\hat{n}_{ij}^{(k)} = \frac{d{ij}^{(k)}}{\| d_{ij}^{(k)} \|}$
3. Linearized constraint:
$\hat{n}_{ij}^{(k)\top} \left( p_i(k) - p_j(k) \right) \ge d_{\text{min}}^i$

This enforces a convexified buffer zone around each agent at every timestep, maintaining distance without introducing non-convex constraints.

## 3.3 Agent-Level Optimization Problem

Each agent solves the following constrained optimization problem:

```math
\begin{aligned}
\min_{X_i, U_i} \quad & J_i(X_i, U_i; X_{-i}) \\
\text{s.t.} \quad
& X_i(k+1) = f_d(X_i(k), U_i(k)), \quad k = 0, \dots, K-1 \\
& X_i(0) = x_{i,\text{init}}, \quad X_i(K) = x_{i,\text{final}} \\
& \hat{n}_{ij}^{(k)\top} \left( p_i(k) - p_j(k) \right) \ge d_{\text{min}}^i, \quad \forall j \ne i, \, \forall k
\end{aligned}
```

where f_d denotes the discretized unicycle dynamics. The linearized constraints depend on the neighbors’ previous trajectories, yielding a best-response structure.