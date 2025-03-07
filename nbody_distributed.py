import time
import jax
from jax import random, block_until_ready
from jax.experimental import multihost_utils
from jax.sharding import Mesh, PartitionSpec
from pathlib import Path
import matplotlib.pyplot as plt

# -------------------------------
# Distributed Initialization
# -------------------------------
# NOTE: this needs to be called BEFORE anything else.
jax.distributed.initialize()

# We assume that N is evenly divisible by the number of processes.
proc = jax.process_index()
num_procs = jax.process_count()
print(f"Process {proc} out of {num_procs}")

if proc == 0:
    print(f"Global devices: {jax.devices()}")
print(f"[{proc}]: Local devices: {jax.local_devices()}")

# -------------------------------
# Simulation Parameters (constants)
# -------------------------------
# Import simulation functions from nbody.py.
from nbody import initialize_state, simulate, total_energy

# Simulation parameters: adjust these values as needed.
N = 20000           # Number of bodies
dim = 2             # Spatial dimensions (2D)
steps = 1000        # Number of simulation steps
dt = 0.01           # Time step
G = 1.0             # Gravitational constant
softening = 0.1     # Softening factor
output_folder = Path("./outputs")
plot_name = "naive_distributed_plot.png"

# -------------------------------
# Sharding the Initial State
# -------------------------------
local_N = N // num_procs
start = proc * local_N
end = start + local_N

# Compute the full initial state on every process.
key = random.PRNGKey(42)
full_state = initialize_state(key, N, dim)
pos, vel, mass = full_state

# Each process extracts its local shard (a contiguous slice along the bodies dimension).
local_pos = pos[start:end, :]
local_vel = vel[start:end, :]
local_mass = mass[start:end]

# Create a global mesh over all available devices.
global_mesh = Mesh(jax.devices(), ('devices',))

# Define partition specs to shard the "bodies" (first) axis.
# These specs are used for constructing the global arrays.
pos_pspec = PartitionSpec('devices', None)   # for pos of shape (N, dim)
vel_pspec = PartitionSpec('devices', None)     # for vel of shape (N, dim)
mass_pspec = PartitionSpec('devices')          # for mass of shape (N,)

# Convert each local array to a global distributed array.
global_pos = multihost_utils.host_local_array_to_global_array(local_pos, global_mesh, pos_pspec)
global_vel = multihost_utils.host_local_array_to_global_array(local_vel, global_mesh, vel_pspec)
global_mass = multihost_utils.host_local_array_to_global_array(local_mass, global_mesh, mass_pspec)

# Combine the sharded arrays into a single global state.
global_state = (global_pos, global_vel, global_mass)

# Print out sharding information.
print("Done sharding:")
print("global_pos.shape:", global_pos.shape)
print("global_vel.shape:", global_vel.shape)
print("global_mass.shape:", global_mass.shape)

# -------------------------------
# Run the Simulation
# -------------------------------
params = (G, softening)
print("Starting simulation...")
start_time = time.time()
final_state = simulate(global_state, dt, steps, params)
final_state = block_until_ready(final_state)
end_time = time.time()
runtime = end_time - start_time
print(f"Simulation completed in {runtime:.3f} seconds.")

final_pos, final_vel, final_mass = final_state

# -------------------------------
# Compute Energy and Plot Results
# -------------------------------
print("Computing total energy...")
energy = total_energy(final_pos, final_vel, final_mass, G, softening)
print("Final total energy:", energy)

# To gather the full array each process
final_pos_full = multihost_utils.process_allgather(final_pos)
print(f"Final positions shape: {final_pos_full.shape}")

# Only process 0 gathers the full global final positions to host-local memory.
if proc == 0:
    print(f"Saving plot to {plot_name}...")
    plt.figure(figsize=(6, 6))
    plt.scatter(final_pos_full[:, 0], final_pos_full[:, 1], s=1)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Final Positions (energy:{energy:.2f} time:{runtime:.3f}s nbProcess:{num_procs})")
    plt.savefig(output_folder / plot_name)
    plt.close()
    print("Plot saved.")