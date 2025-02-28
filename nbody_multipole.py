import time
import jax
import jax.numpy as jnp
from jax import jit, lax
from pathlib import Path
import matplotlib.pyplot as plt

# -------------------------------
# Simulation Parameters (constants)
# -------------------------------
N = 20000           # Number of bodies
dim = 2             # Dimension (2D)
steps = 1000        # Total simulation steps
dt = 0.01           # Time step
G = 1.0             # Gravitational constant
softening = 0.1     # Softening factor

# Multipole-specific parameters:
grid_size = 16      # Regular grid resolution along each dimension
theta = 0.5         # Threshold for using multipole approximation vs. direct summation

output_folder = Path("./outputs")
plot_name = "multipole_plot.png"

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

initialize_state = jit(initialize_state, static_argnames=("N", "dim"))

# -------------------------------
# Grid Aggregation Utilities
# -------------------------------
def compute_cell_indices(pos, grid_size):
    """
    Given positions in [-1,1]^2, compute integer cell indices for a regular grid of size grid_size.
    Returns an array of shape (N, 2) with indices in {0,...,grid_size-1}.
    """
    # Map positions from [-1,1] to [0,1], then scale to grid indices.
    indices = jnp.floor((pos + 1.0) / 2.0 * grid_size).astype(jnp.int32)
    indices = jnp.clip(indices, 0, grid_size - 1)
    return indices

def aggregate_cells(pos, mass, grid_size):
    """
    Bin the particles into a regular grid. For each cell (flattened index) compute:
      - Total mass in the cell.
      - Sum of mass-weighted positions.
    Then compute the cell’s center-of-mass; if a cell is empty, use its geometric center.
    Returns:
      cell_mass: (grid_size*grid_size,) array of total mass per cell.
      cell_com: (grid_size*grid_size, 2) array of cell center-of-mass.
      particle_cells: (N, 2) cell indices for each particle.
    """
    indices = compute_cell_indices(pos, grid_size)  # shape (N,2)
    flat_indices = indices[:, 0] * grid_size + indices[:, 1]  # shape (N,)
    num_cells = grid_size * grid_size

    # Sum mass and mass*position per cell using segment_sum.
    cell_mass = jax.ops.segment_sum(mass, flat_indices, num_segments=num_cells)
    cell_mass_pos = jax.ops.segment_sum(mass[:, None] * pos, flat_indices, num_segments=num_cells)

    # Compute geometric center of each cell.
    cell_ids = jnp.arange(num_cells)
    cell_i = cell_ids // grid_size
    cell_j = cell_ids % grid_size
    cell_width = 2.0 / grid_size
    cell_center_x = -1.0 + cell_width / 2.0 + cell_i * cell_width
    cell_center_y = -1.0 + cell_width / 2.0 + cell_j * cell_width
    geom_center = jnp.stack([cell_center_x, cell_center_y], axis=1)

    # Where cell_mass > 0, use aggregated center; otherwise use geometric center.
    com = jnp.where(cell_mass[:, None] > 0, cell_mass_pos / cell_mass[:, None], geom_center)
    return cell_mass, com, indices

# -------------------------------
# Multipole Force Computation
# -------------------------------
def compute_accelerations_multipole(pos, mass, G, softening, grid_size, theta):
    """
    Compute net acceleration for each particle using a multipole-like method.
    
    Steps:
      1. Bin particles into a regular grid; compute each cell's total mass and center-of-mass.
      2. For each particle, treat cells that are well-separated (not in or adjacent to the particle's cell)
         using a multipole (monopole) approximation.
      3. For cells that are nearby (same or adjacent cells), compute the forces directly.
    
    Returns:
      acc: Array of shape (N, 2) representing the net acceleration on each particle.
    """
    # Aggregate cell data.
    cell_mass, cell_com, particle_cell_indices = aggregate_cells(pos, mass, grid_size)
    num_cells = grid_size * grid_size

    # Precompute cell grid indices for all cells.
    cell_ids = jnp.arange(num_cells)
    cell_i = cell_ids // grid_size
    cell_j = cell_ids % grid_size
    cell_grid = jnp.stack([cell_i, cell_j], axis=1)  # shape (num_cells, 2)

    # Determine for each particle which cells are "near" (within one cell in any direction).
    # particle_cell_indices has shape (N,2); compare with cell_grid.
    diff_cells = particle_cell_indices[:, None, :] - cell_grid[None, :, :]  # (N, num_cells, 2)
    near_mask = jnp.max(jnp.abs(diff_cells), axis=2) <= 1  # (N, num_cells) boolean

    # For far cells, use the multipole (monopole) approximation.
    diff_vec = pos[:, None, :] - cell_com[None, :, :]  # (N, num_cells, 2)
    r2 = jnp.sum(diff_vec**2, axis=2) + softening**2   # (N, num_cells)
    inv_r3 = r2 ** (-1.5)
    # Force from each cell: F = -G * m_particle * cell_mass * diff / r^3.
    force_cell = -G * (mass[:, None] * cell_mass[None, :])[:, :, None] * diff_vec * inv_r3[:, :, None]
    # Zero out contributions from near cells.
    far_force = jnp.where(near_mask[:, :, None], 0.0, force_cell)
    multipole_force = jnp.sum(far_force, axis=1)  # (N, 2)

    # For near cells, compute direct interactions between particles.
    # Compute each particle's cell index (already computed above).
    # We now compute a full pairwise interaction—but only keep interactions where the particles are in cells
    # that are near each other.
    diff_particles = pos[:, None, :] - pos[None, :, :]  # (N, N, 2)
    r2_particles = jnp.sum(diff_particles**2, axis=2) + softening**2  # (N, N)
    inv_r3_particles = r2_particles ** (-1.5)
    force_pair = -G * (mass[:, None] * mass[None, :])[:, :, None] * diff_particles * inv_r3_particles[:, :, None]
    # Build near-particle mask using cell indices.
    diff_particle_cell = particle_cell_indices[:, None, :] - particle_cell_indices[None, :, :]  # (N, N, 2)
    near_particle_mask = jnp.max(jnp.abs(diff_particle_cell), axis=2) <= 1
    # Exclude self-interaction.
    near_particle_mask = near_particle_mask & (~jnp.eye(pos.shape[0], dtype=bool))
    direct_force = jnp.sum(jnp.where(near_particle_mask[:, :, None], force_pair, 0.0), axis=1)  # (N, 2)

    # Total force is the sum of far-field (multipole) and near-field (direct) contributions.
    net_force = multipole_force + direct_force
    acc = net_force / mass[:, None]
    return acc

# -------------------------------
# Simulation Step and Loop (using JAX jit and lax.fori_loop)
# -------------------------------
def simulation_step(state, dt, params, grid_size, theta):
    """
    Update state (positions, velocities, masses) for one time step using Euler integration.
    """
    pos, vel, mass = state
    G, softening = params
    acc = compute_accelerations_multipole(pos, mass, G, softening, grid_size, theta)
    new_vel = vel + dt * acc
    new_pos = pos + dt * new_vel
    return new_pos, new_vel, mass

def simulate(state, dt, steps, params, grid_size, theta):
    """
    Run the simulation for a fixed number of steps.
    The loop is implemented with lax.fori_loop so the whole simulation is jitted.
    """
    def body_fun(i, state):
        return simulation_step(state, dt, params, grid_size, theta)
    return lax.fori_loop(0, steps, body_fun, state)

simulate = jit(simulate, static_argnames=("dt", "steps", "params", "grid_size", "theta"),
               donate_argnames=("state",))

# -------------------------------
# Energy Computation (jitted)
# -------------------------------
def total_energy(pos, vel, mass, G, softening):
    kinetic = 0.5 * jnp.sum(mass * jnp.sum(vel**2, axis=-1))
    diff = pos[:, None, :] - pos[None, :, :]
    dist = jnp.sqrt(jnp.sum(diff**2, axis=-1) + softening**2)
    potential = -0.5 * G * jnp.sum((mass[:, None] * mass[None, :]) / dist)
    return kinetic + potential

total_energy = jit(total_energy, static_argnames=("G", "softening"))

# -------------------------------
# Main Execution: Initialization, Simulation, and Output
# -------------------------------
print("Initializing state...")
key = jax.random.PRNGKey(42)
state0 = initialize_state(key, N, dim)
print("State initialized.")

print("Starting simulation with multipole method...")
params = (G, softening)
start_time = time.time()
final_state = simulate(state0, dt, steps, params, grid_size, theta)
final_state = jax.block_until_ready(final_state)
end_time = time.time()
runtime = end_time - start_time
print(f"Simulation completed in {runtime:.3f} seconds.")

final_pos, final_vel, final_mass = final_state

print("Computing total energy...")
energy = total_energy(final_pos, final_vel, final_mass, G, softening)
print("Final total energy:", energy)

print(f"Saving plot to {plot_name}...")
final_pos_np = jax.device_get(final_pos)
plt.figure(figsize=(6, 6))
plt.scatter(final_pos_np[:, 0], final_pos_np[:, 1], s=1)
plt.xlabel("x")
plt.ylabel("y")
plt.title(f"Final Positions (energy: {energy:.2f}, time: {runtime:.3f}s)")
plt.savefig(output_folder / plot_name)
plt.close()
print("Plot saved.")
