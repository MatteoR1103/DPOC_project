# Flappy Bird: Dynamic Programming & Optimal Control

This repository contains the implementation of the optional project for the **Dynamic Programming and Optimal Control** course. The project involves simulating the classic "Flappy Bird" game and formulating it as an **Infinite Horizon Problem** to derive an optimal control policy.



## 📝 Project Overview

The objective is to navigate an agent (the bird) through a series of obstacles (pipes) by determining the optimal action at each discrete time step. By modeling the environment as a **Markov Decision Process (MDP)**, we move beyond simple heuristics to a mathematically proven optimal strategy.

### Problem Formulation
* **State Space ($S$):** Characterized by the bird's vertical position ($y$), vertical velocity ($v$), and the relative horizontal and vertical distance to the next goal post.
* **Action Space ($A$):** A discrete set $\{0, 1\}$, where $0$ represents "do nothing" and $1$ represents "flap."
* **Cost Function ($g$):** A terminal cost is applied upon collision, while a running cost encourages the bird to maintain a trajectory centered within the pipe gaps.

---

## 🚀 Key Features

* **Infinite Horizon Formulation:** Solved using Value Iteration to find a stationary policy $\mu^*(x)$ that maximizes survival probability over an indefinite period.
* **State Discretization:** Implementation of a grid-based approach to handle the continuous nature of the game's physics while mitigating the "curse of dimensionality."
* **Simulation & Visualization:** A Python-based simulation engine to test the policy in real-time and visualize the bird's decision-making process.


---



https://github.com/user-attachments/assets/cea06e38-7f24-4fba-b68c-0583b8767508





