#!/bin/bash
#SBATCH -n 1  # number of tasks (e.g. MPI)
#SBATCH -t 3-00:00:00
#SBATCH -p gpu
#SBATCH --cpus-per-task=16
#SBATCH --job-name=hdc
#SBATCH --mem=128GB
#SBATCH --hint=nomultithread    # don't use hyperthreading

module purge
module load apptainer
apptainer exec ../apptainer.sif make
