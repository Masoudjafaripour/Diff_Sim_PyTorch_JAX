import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Constants
g = 9.81   # Gravity (m/s^2)
l = 1.0    # Pendulum length (m)
dt = 0.02  # Time step (s)

# Pendulum Dynamics (Theta, Omega)
def pendulum_dynamics(state, torque=0.0):
    theta, omega = state
    dtheta = omega
    domega = (-g / l) * jnp.sin(theta) + torque  # Equation of motion
    return jnp.array([theta + dtheta * dt, omega + domega * dt])

# Loss Function: Minimize Deviation from Upright Position (theta = 0)
def loss(state):
    return jnp.sum(state**2)  # Simple quadratic loss

# Compute Gradient w.r.t. Initial State
grad_loss = jax.grad(loss)

# Simulate Inverted Pendulum for N Steps
def simulate_pendulum(initial_state, steps=100, torque=0.0):
    states = [initial_state]
    for _ in range(steps):
        new_state = pendulum_dynamics(states[-1], torque)
        states.append(new_state)
    return jnp.stack(states)

# Initial Conditions: Small Perturbation from Upright
initial_state = jnp.array([0.1, 0.0])  # Theta = 0.1 rad, Omega = 0

# Run Simulation
states = simulate_pendulum(initial_state, steps=200)

# Plot Results
time = jnp.arange(len(states)) * dt
plt.figure(figsize=(10, 5))
plt.plot(time, states[:, 0], label="Theta (rad)")
plt.plot(time, states[:, 1], label="Omega (rad)")
plt.xlabel("Time (s)")
plt.ylabel("State")
plt.legend()
plt.title("Inverted Pendulum Simulation (JAX)")
plt.show()
