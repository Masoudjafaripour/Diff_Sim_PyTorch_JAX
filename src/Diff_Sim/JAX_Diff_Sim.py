import jax
import jax.numpy as jnp

# System Dynamics: x'' = -k*x (Hooke's Law)
def dynamics(state, k=1.0, dt=0.1):
    x, v = state
    a = -k * x  # Acceleration
    v_new = v + a * dt
    x_new = x + v_new * dt
    return jnp.array([x_new, v_new])

# Differentiate w.r.t. Initial Conditions
def loss(state):
    final_state = dynamics(state)
    return jnp.sum(final_state**2)  # Minimize final energy

grad_loss = jax.grad(loss)

# Test Gradient Computation
initial_state = jnp.array([1.0, 0.0])  # Initial position and velocity
print("Gradient:", grad_loss(initial_state))
