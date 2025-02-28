#!/bin/bash
#SBATCH --image=ghcr.io/nvidia/jax:jax
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --qos=debug
#SBATCH --constraint=gpu
#SBATCH --gpus-per-node=4
#SBATCH --module=gpu,nccl-plugin

srun shifter python3 ./nbody.py
