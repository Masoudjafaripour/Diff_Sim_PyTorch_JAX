# https://chatgpt.com/share/68867266-3aa8-8000-ae15-75f805cdf67f

import brax
from brax import jumpy as jp
from brax.envs import create
from brax.io import html
from jax import grad

env = create(env_name='ant')
state = env.reset(rng=jp.random_prngkey(seed=0))

def loss_fn(actions):
    trajectory = simulate(env, state, actions)  # forward sim
    return -trajectory.reward.sum()  # negative reward = loss

gradients = grad(loss_fn)(actions)


import torch
from tds.torch_sim import SimModel  # example wrapper

sim = SimModel()
x0 = torch.tensor([0.0, 0.0], requires_grad=True)
actions = torch.tensor([...], requires_grad=True)

final_pos = sim.simulate(x0, actions)
loss = (final_pos - target_pos).pow(2).sum()
loss.backward()  # gradients w.r.t actions and initial state

loss = loss_fn(simulation_output)  # e.g., shape matching
loss.backward()  # gradients flow through simulation
