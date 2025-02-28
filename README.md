# Jax n-body Simulation

JAX nbody implementation.

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

## TODO

* ?
