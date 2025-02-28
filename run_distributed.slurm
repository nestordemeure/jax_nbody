#!/bin/bash
#SBATCH --job-name=nbody_naive_distributed
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --qos=debug
#SBATCH --constraint=gpu
#SBATCH --gpus-per-node=4
#SBATCH --account=nstaff
#SBATCH --output=./outputs/nbody_distributed.out

# load our dependencies
module load python
source jax-venv/bin/activate

# runs the model
srun python3 nbody_distributed.py