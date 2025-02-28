import time
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -------------------------------
# Simulation Parameters (constants)
# -------------------------------
N = 20000           # Number of bodies
dim = 2             # Dimension (2D)
steps = 1000        # Total simulation steps
dt = 0.01           # Time step
G = 1.0             # Gravitational constant
softening = 0.1     # Softening factor
theta = 0.5         # Barnes–Hut threshold parameter
output_folder = Path("./outputs")  # where to store outputs
plot_name = "barneshut_plot.png"        # Plot file name

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

initialize_state = jax.jit(initialize_state, static_argnames=("N", "dim"))

# -------------------------------
# Barnes–Hut Tree Data Structures and Functions (Python, non-JIT)
# -------------------------------
class Node:
    def __init__(self, center, size, mass, com, particle_index=None, children=None):
        """
        center: 2D center of the cell (numpy array)
        size: half-width of the cell (scalar)
        mass: total mass contained in the cell (scalar)
        com: center-of-mass of the cell (numpy array)
        particle_index: if a leaf with a single particle, record its index (int); otherwise None.
        children: list of four child Node objects (or None for a leaf)
        """
        self.center = center  
        self.size = size      
        self.mass = mass      
        self.com = com        
        self.particle_index = particle_index  
        self.children = children  

def build_tree(positions, masses, indices, center, size, capacity=1):
    """
    Recursively build a quadtree over the particles.
    
    positions: np.array of shape (N,2)
    masses: np.array of shape (N,)
    indices: list of particle indices (ints) for particles in this cell
    center: 2D center of the current cell (np.array)
    size: half-width of the cell (scalar)
    capacity: maximum number of particles per leaf (here, 1)
    """
    if len(indices) == 0:
        return None
    # If number of particles is <= capacity, create a leaf node.
    if len(indices) <= capacity:
        total_mass = masses[indices].sum()
        com = np.average(positions[indices], axis=0, weights=masses[indices])
        # If exactly one particle, record its index.
        p_index = indices[0] if len(indices) == 1 else None
        return Node(center=center, size=size, mass=total_mass, com=com, particle_index=p_index, children=None)
    
    # Otherwise, subdivide into 4 quadrants.
    children = []
    new_size = size / 2.0
    # Offsets for quadrants: top-left, top-right, bottom-left, bottom-right.
    offsets = [np.array([-1, 1]), np.array([1, 1]),
               np.array([-1, -1]), np.array([1, -1])]
    for offset in offsets:
        child_center = center + offset * new_size * 0.5  # shift by a quarter of the cell width
        # Select particles that lie within the quadrant.
        pos_sub = positions[indices]
        cond_x = (pos_sub[:, 0] >= child_center[0] - new_size) & (pos_sub[:, 0] < child_center[0] + new_size)
        cond_y = (pos_sub[:, 1] >= child_center[1] - new_size) & (pos_sub[:, 1] < child_center[1] + new_size)
        cond = cond_x & cond_y
        child_indices = [indices[i] for i, flag in enumerate(cond) if flag]
        child_node = build_tree(positions, masses, child_indices, child_center, new_size, capacity)
        children.append(child_node)
        
    # Compute total mass and center-of-mass (weighted average) from children.
    total_mass = 0.0
    com = np.array([0.0, 0.0])
    for child in children:
        if child is not None:
            total_mass += child.mass
            com += child.mass * child.com
    if total_mass > 0:
        com /= total_mass
    else:
        com = center
    return Node(center=center, size=size, mass=total_mass, com=com, particle_index=None, children=children)

def compute_force_on_particle(i, pos_i, tree, theta, G, softening):
    """
    Recursively traverse the Barnes–Hut tree to compute the acceleration (force/mass)
    on particle i located at pos_i.
    """
    force = np.array([0.0, 0.0])
    
    def traverse(node, force):
        if node is None:
            return force
        # If node is a leaf:
        if node.children is None:
            # Skip self-interaction.
            if node.particle_index == i:
                return force
            # Compute direct force contribution.
            r = pos_i - node.com
            dist_sq = np.dot(r, r) + softening**2
            inv_dist3 = dist_sq**(-1.5)
            return force - G * node.mass * r * inv_dist3
        else:
            # For an internal node, decide whether to approximate.
            r = pos_i - node.com
            d = np.sqrt(np.dot(r, r)) + 1e-10
            if node.size / d < theta:
                # Use multipole (monopole) approximation.
                dist_sq = np.dot(r, r) + softening**2
                inv_dist3 = dist_sq**(-1.5)
                return force - G * node.mass * r * inv_dist3
            else:
                # Otherwise, traverse the children.
                for child in node.children:
                    force = traverse(child, force)
                return force
    force = traverse(tree, force)
    return force

def compute_all_accelerations(pos, mass, G, softening, theta):
    """
    Compute the acceleration on every particle using the Barnes–Hut algorithm.
    Since tree building and traversal are dynamic, we build the tree (on the CPU)
    outside of JIT and then compute the force on each particle via a Python loop.
    """
    # Convert JAX arrays to NumPy arrays.
    pos_np = np.array(jax.device_get(pos))
    mass_np = np.array(jax.device_get(mass))
    N_particles = pos_np.shape[0]
    # Compute the bounding box of all particles.
    xmin, xmax = pos_np[:,0].min(), pos_np[:,0].max()
    ymin, ymax = pos_np[:,1].min(), pos_np[:,1].max()
    center = np.array([(xmin+xmax)/2, (ymin+ymax)/2])
    half_size = max(xmax - xmin, ymax - ymin)/2.0 + 1e-5
    # Build the tree over all particles.
    indices = list(range(N_particles))
    tree = build_tree(pos_np, mass_np, indices, center, half_size, capacity=1)
    # For each particle, traverse the tree to compute acceleration.
    acc_list = []
    for i in range(N_particles):
        a = compute_force_on_particle(i, pos_np[i], tree, theta, G, softening)
        acc_list.append(a)
    return jnp.stack(acc_list)

# -------------------------------
# Simulation Step Function (Euler integration)
# -------------------------------
def simulation_step(state, dt, params):
    """
    Update state (positions, velocities, masses) for one time step using Euler integration.
    Uses the Barnes–Hut tree to compute the accelerations.
    """
    pos, vel, mass = state
    G, softening, theta = params
    acc = compute_all_accelerations(pos, mass, G, softening, theta)
    new_vel = vel + dt * acc
    new_pos = pos + dt * new_vel
    return new_pos, new_vel, mass

# -------------------------------
# Energy Computation (jitted)
# -------------------------------
def total_energy(pos, vel, mass, G, softening):
    kinetic = 0.5 * jnp.sum(mass * jnp.sum(vel**2, axis=-1))
    diff = pos[:, None, :] - pos[None, :, :]
    dist = jnp.sqrt(jnp.sum(diff**2, axis=-1) + softening**2)
    potential = -0.5 * G * jnp.sum((mass[:, None] * mass[None, :]) / dist)
    return kinetic + potential

total_energy = jax.jit(total_energy, static_argnames=("G", "softening"))

# -------------------------------
# Simulation Loop (Python loop)
# -------------------------------
def simulate(state, dt, steps, params):
    """
    Run the simulation for a fixed number of steps.
    (Note: due to the dynamic Barnes–Hut tree construction and recursion,
    this loop is implemented in Python rather than jitted.)
    """
    for i in range(steps):
        state = simulation_step(state, dt, params)
    return state

# -------------------------------
# Main Execution with Prints and Timing
# -------------------------------
print("Initializing state...")
key = jax.random.PRNGKey(42)
state0 = initialize_state(key, N, dim)
print("State initialized.")

print("Starting simulation...")
params = (G, softening, theta)
start_time = time.time()
final_state = simulate(state0, dt, steps, params)
final_state = jax.device_get(final_state)  # ensure state is moved back from device if needed
end_time = time.time()
runtime = end_time - start_time
print(f"Simulation completed in {runtime:.3f} seconds.")

final_pos, final_vel, final_mass = final_state

print("Computing total energy...")
energy = total_energy(final_pos, final_vel, final_mass, G, softening)
print("Final total energy:", energy)

print(f"Saving plot to {plot_name}...")
final_pos_np = np.array(final_pos)
plt.figure(figsize=(6, 6))
plt.scatter(final_pos_np[:, 0], final_pos_np[:, 1], s=1)
plt.xlabel("x")
plt.ylabel("y")
plt.title(f"Final Positions (energy:{energy:.2f}, time:{runtime:.3f}s)")
plt.savefig(output_folder / plot_name)
plt.close()
print("Plot saved.")
