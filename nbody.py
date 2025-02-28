import jax
import jax.numpy as jnp
from jax import jit, lax
import matplotlib.pyplot as plt
from xmap import xmap

# -------------------------------
# Simulation Parameters (constants)
# -------------------------------
N = 50           # Number of bodies
dim = 2          # Dimension (2D)
steps = 1000     # Total simulation steps
dt = 0.01        # Time step
G = 1.0          # Gravitational constant
softening = 0.1  # Softening factor

# -------------------------------
# Initialization Function (with mass generation)
# -------------------------------
def initialize_state(key, N, dim):
    """
    Initialize positions, velocities, and masses for N bodies in dim dimensions.
    Positions are uniformly random in [-1, 1]^dim, velocities are small random normals,
    and masses are uniformly random between 0.5 and 1.5.
    """
    key1, key2, key3 = jax.random.split(key, 3)
    pos = jax.random.uniform(key1, (N, dim), minval=-1.0, maxval=1.0)
    vel = jax.random.normal(key2, (N, dim)) * 0.1
    mass = jax.random.uniform(key3, (N,), minval=0.5, maxval=1.5)
    return pos, vel, mass

# JIT-compile initialize_state with N and dim as static arguments.
initialize_state = jit(initialize_state, static_argnames=("N", "dim"))

# -------------------------------
# Acceleration Computation using double vmap and jnp.fill_diagonal
# -------------------------------
def body_force(pos_i, pos_j, mass_j, G, softening):
    """Compute the gravitational force on one body at pos_i due to a body at pos_j."""
    diff = pos_i - pos_j
    r_sq = jnp.dot(diff, diff) + softening**2
    inv_r3 = r_sq ** (-1.5)
    return -G * mass_j * diff * inv_r3

def compute_accelerations(pos, mass, G, softening):
    """
    Compute the net acceleration for each body via pairwise gravitational forces,
    using xmap for clearer named-axis vectorization.

    Args:
      pos: Array of shape (N, d) containing positions of N bodies, with axes named 'i' and 'd'.
      mass: Array of shape (N,) containing the masses, with axis 'i'.
      G: Gravitational constant.
      softening: Softening parameter to avoid singularities.

    Returns:
      An array of shape (N, d) representing the net acceleration on each body.
    """
    # Compute the full pairwise force matrix using xmap.
    # The xmap call vectorizes body_force over two axes:
    #   - 'i' (for the source body: pos_i)
    #   - 'j' (for the target body: pos_j and mass_j)
    # The ellipsis (...) denotes any remaining unnamed axes (here, the spatial axis 'd').
    force_matrix = xmap(
        body_force,
        in_axes={
            "pos_i": ["i", ...],
            "pos_j": ["j", ...],
            "mass_j": ["j"],
            "G": float,
            "softening": float,
        },
        out_axes=["i", "j", ...]
    )(pos, pos, mass, G, softening)
    # At this point, force_matrix has shape (N, N, d) where force_matrix[i, j, :]
    # is the force on body i due to body j.
    
    # Sum over axis 'j' (all bodies contributing force) to get the net acceleration on each body (minus its own).
    net_acc = xmap(
        lambda forces, diag_index: jnp.sum(forces) - forces[diag_index],
        in_axes={
            "forces": ["i", ..., "d"],
            "diag_index": ["i"],
        },
        out_axes=["i", "d"]
    )(force_matrix, jnp.arange(mass.size))
    return net_acc

# -------------------------------
# Simulation Step Function
# -------------------------------
def simulation_step(state, dt, params):
    """
    Update state (positions, velocities, masses) for one time step using Euler integration.
    Mass remains constant.
    """
    pos, vel, mass = state
    G, softening = params
    acc = compute_accelerations(pos, mass, G, softening)
    new_vel = vel + dt * acc
    new_pos = pos + dt * new_vel
    return new_pos, new_vel, mass

# -------------------------------
# Simulation Loop Using lax.fori_loop
# -------------------------------
def simulate(state, dt, steps, params):
    """
    Run the simulation for a fixed number of steps.
    The loop is entirely inside this function so that it can be jitted as a whole.
    """
    def body_fun(i, state):
        return simulation_step(state, dt, params)
    return lax.fori_loop(0, steps, body_fun, state)

# JIT-compile simulate with static arguments dt, steps, and params.
# Donate only the state (an array) for efficiency.
simulate = jit(simulate, static_argnames=("dt", "steps", "params"), donate_argnames=("state",))

# -------------------------------
# Energy Computation (jitted)
# -------------------------------
def total_energy(pos, vel, mass, G, softening):
    """
    Compute the total energy (kinetic + potential) of the system.
    """
    kinetic = 0.5 * jnp.sum(mass * jnp.sum(vel**2, axis=-1))
    diff = pos[:, None, :] - pos[None, :, :]
    dist = jnp.sqrt(jnp.sum(diff**2, axis=-1) + softening**2)
    potential = -0.5 * G * jnp.sum((mass[:, None] * mass[None, :]) / dist)
    return kinetic + potential

# JIT-compile total_energy with G and softening as static.
total_energy = jit(total_energy, static_argnames=("G", "softening"))

# -------------------------------
# Main Execution with Prints
# -------------------------------
print("Initializing state...")
key = jax.random.PRNGKey(42)
state0 = initialize_state(key, N, dim)
print("State initialized.")

print("Starting simulation...")
params = (G, softening)  # parameters to pass to simulation
final_state = simulate(state0, dt, steps, params)
final_pos, final_vel, final_mass = final_state
print("Simulation complete.")

print("Computing total energy...")
energy = total_energy(final_pos, final_vel, final_mass, G, softening)
print("Final total energy:", energy)

print("Saving plot to nbody_plot.png...")
final_pos_np = jax.device_get(final_pos)
plt.figure(figsize=(6, 6))
plt.scatter(final_pos_np[:, 0], final_pos_np[:, 1])
plt.xlabel("x")
plt.ylabel("y")
plt.title("Final Positions of Bodies")
plt.savefig("nbody_plot.png")
plt.close()
print("Plot saved.")
