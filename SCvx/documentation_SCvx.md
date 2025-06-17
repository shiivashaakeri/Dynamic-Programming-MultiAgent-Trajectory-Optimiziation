# 1. Introduction

## 1.1 Motivation for Trajectory Optimization

Trajectory optimization plays a central role in planning and control for autonomous systems, particularly when operating in environments with constraints, nonlinear dynamics, or objectives involving efficiency, safety, or agility. In robotic motion planning, achieving smooth, dynamically feasible trajectories that avoid obstacles and respect physical limitations is essential for real-world deployment. Traditional optimal control methods often struggle with convergence and scalability when dealing with nonlinear dynamics and nonconvex constraints. This motivates the use of structured iterative algorithms that can leverage convex approximations to reliably solve such problems.

## 1.2 Overview of Successive Convexification (SCvx)

Successive Convexification (SCvx) is an iterative optimization method designed to solve nonconvex trajectory optimization problems by sequentially solving convex subproblems. At each iteration, the original nonlinear and nonconvex dynamics are linearized around a reference trajectory using first-order approximations. The resulting linear dynamics, along with convexified constraints, form a subproblem that can be efficiently solved using convex optimization techniques. SCvx incorporates trust region mechanisms to ensure global convergence and robustness against poor linearization. It penalizes defects (i.e., violations of dynamics due to linearization errors) and slack variables introduced to handle constraint relaxation, guiding the solution toward feasible, optimal trajectories.

## 1.3 Application to Unicycle Model

In this work, we apply the SCvx framework to the trajectory optimization problem for a unicycle model, a commonly used abstraction for mobile robots. The unicycle system is characterized by nonlinear, nonholonomic dynamics that make direct optimization challenging. Our implementation demonstrates how SCvx can be used to generate smooth, obstacle-avoiding trajectories from an initial state to a target pose, while satisfying bounds on velocity and angular rate. The optimization incorporates discretization via First-Order Hold (FOH), trust region adaptation, and symbolic Jacobian computation, and is structured to support modular extension to more complex models.

# 2. Problem Formulation

We consider a unicycle model where the system’s state vector is defined as
```math
x = \begin{bmatrix} x \\ y \\ \theta \end{bmatrix} \in \mathbb{R}^3,
```

representing the 2D position $(x, y)$ and heading angle $\theta$. The control input is
$$
u = \begin{bmatrix} v \\ \omega \end{bmatrix} \in \mathbb{R}^2,
$$
where $v$ is the forward linear velocity and $\omega$ is the angular velocity.