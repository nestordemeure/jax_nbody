# Jax n-body Simulation

JAX nbody implementation.

## Content

* [`nbody.py`](./nbody.py): naive (n^2) simulation.
* [`nbody_distributed.py`](./nbody_distributed.py): naive (n^2) simulation, distributed over available GPUs (this might scale with GPU provided).
* [`nbody_multipole.py`](./nbody_multipole.py): simulation using the multipole method over a fixed grid (non working).
* [`nbody_barneshut.py`](./nbody_barneshut.py): simulation using the Barnes-Hut method (non working).

## Install

```sh
module load python

# Create a new virtual environment
python -m venv jax-venv

# Activate the virtual environment
source jax-venv/bin/activate

# Install some common dependencies
pip install numpy scipy rich matplotlib

# Install the GPU-enabled version of JAX (for CUDA 12)
pip install --upgrade "jax[cuda12]==0.4.37"
```

## Usage

Use the following to run the code locally:

```sh
module load python
source jax-venv/bin/activate

python3 ./nbody.py
```

Run the distributed code with the following:

```sh
sbatch run_distributed.slurm
```

## TODO

algorithms:

* multipole: no idea if the prototype is correct (but it is slow)
* barnes hut: deadlocking?
