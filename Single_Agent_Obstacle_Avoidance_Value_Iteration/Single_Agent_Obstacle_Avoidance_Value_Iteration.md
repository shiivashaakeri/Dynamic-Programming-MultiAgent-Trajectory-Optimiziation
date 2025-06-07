
## Environment
We defined a 2D grid world $\mathcal{G}\in \mathbb{Z}^2$ of size $(20\times 20)$.

Each state $s\in \mathcal{S}$ represents a grid cell, where

```math
\mathcal{S} = \{ (i,j)\mid 0\leq i<20,\, 0\leq j<20\}\, \cup \,\{s_T\},
```

where $s=(i,j)$ denotes the cell at row $i$ and column $j$, and $s_T = (\text{None, None})$ is a terminal state.

## Action Space
The set of allowed actions is defined as:

```math
\mathcal{A} = \{a_1, a_2, a_3,a_4\} = \{\text{up, down, left, right}\}
```

Each action:
- up: $(i,j)\rightarrow (i+1, j)$
- down: $(i,j)\rightarrow (i-1, j)$
- right: $(i,j)\rightarrow (i, j+1)$
- left: $(i,j)\rightarrow (i, j-1)$

Movements are clamped at the grid boundries:
- if $i+1 \geq 20$, then $(i,j)\rightarrow (19,j)$
- if $i-1<0$, then $(i,j) \rightarrow (0,j)$
- similarly for horizontal movement in $j$

## Terminal, Goal, Fire States
We define goal states as $\mathcal{S}_G$, fire states (obstacles) as $\mathcal{S}_F$, and terminal state as $s_T$.

The agent transitions to $s_T$, if it reaches a goal state or is explicitly terminated.

## Reward
The reward function $R:\mathcal{S}\rightarrow \mathbb{R}$ is defined as:

```latex
R(s) = \begin{cases}
d & \text{if something} \\
e & \text{otherwise}
\end{cases}
```

