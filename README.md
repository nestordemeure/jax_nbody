# Jax n-body Simulation

JAX nbody implementation.

## Content

* [`nbody.py`](./nbody.py): naive (n^2) simulation.
* [`nbody_distributed.py`](./nbody_distributed.py): naive (n^2) simulation, distributed over 4 GPUs.
* [`nbody_multipole.py`](./nbody_multipole.py): simulation using the multipole method over a fixed grid.

## Install

```sh
module load python

# Create a new virtual environment
python -m venv jax-venv

# Activate the virtual environment
source jax-venv/bin/activate

# Upgrade pip
pip install --upgrade pip

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

On an interactive node, you would run (for a distributed example):

```sh
salloc --exclusive --account=nstaff -N 1 -n 4 -C gpu -G 4 -q interactive --gpus-per-task=1 -t 01:00:00

module load python
source jax-venv/bin/activate

srun --ntasks=4 --gpus-per-task=1 ython3 nbody_distributed.py
```

And, obviously, you can run the `.slurm` scripts provided.

## TODO

distributed code:

* local process should create their data slice, not the full data
* the end result should only be gathered to process 0, not all to all

algorithms:

* multipole: no idea if the prototype is correct (but it is slow)
* barnes hut: deadlocking?
