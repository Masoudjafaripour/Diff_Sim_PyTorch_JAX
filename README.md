# Differentiable Simulation and Reinforcement Learning with JAX

This repository explores **differentiable simulation of physical systems** and **learning-based control** using **JAX** for high-performance, end-to-end gradient-based optimization. The focus is on pendulum-like dynamical systems as minimal yet representative examples of robotics and control problems.

## Scope

The core idea is to model physical dynamics as **fully differentiable programs**, enabling:

* Gradient-based system analysis and optimization
* Policy learning through **deep reinforcement learning** directly on the simulator
* Efficient acceleration on CPU/GPU via **JAX XLA compilation**

The repository compares and connects:

* Classical physics-based simulation
* Neural-network-based policy learning
* Differentiable simulation + policy gradient methods

## Key Components

* **Differentiable dynamics in JAX**
  Single-DoF and double-DoF pendulum systems implemented using pure JAX, compatible with `jit`, `grad`, and `lax.scan`.

* **Policy Gradient / REINFORCE**
  End-to-end training of stochastic policies where gradients flow through the simulator, demonstrating learning-based control of physical systems.

* **JAX vs PyTorch vs CUDA**
  Side-by-side implementations highlighting trade-offs between:

  * JAX (XLA-compiled, functional, differentiable)
  * PyTorch (imperative, flexible autograd)
  * Custom CUDA kernels (maximum control, lowest-level)

* **Neural control of physical systems**
  Neural networks act as controllers, optimized directly against physical objectives (stabilization, regulation) using deep RL.

## Structure

* `Diff_Sim_JAX.py`, `JAX_Diff_Sim.py`
  Core differentiable simulators in JAX

* `pend_JAX_RL.py`, `pend_JAX_RL.ipynb`
  Reinforcement learning (REINFORCE) with differentiable physics

* `InvPend_JAX.py`
  Inverted pendulum dynamics and control

* `Diff_PG.ipynb`
  Policy gradient experiments and analysis

* `Cuda_Diff_Sim.cpp`
  Low-level CUDA-based differentiable simulation (comparison baseline)

* `Dis_Comp_JAX.py`
  Computational and performance comparisons

## Motivation

Differentiable simulation bridges classical control, robotics, and modern machine learning. By combining **physics priors** with **neural policies** and **automatic differentiation**, the same framework can be used for:

* System identification
* Control learning
* Sensitivity analysis
* Sim-to-real research pipelines

This repository serves as a compact experimental testbed for these ideas, emphasizing clarity, performance, and physical correctness.
