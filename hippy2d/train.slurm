#!/bin/bash
#SBATCH --job-name=hippy2d_train
#SBATCH --output=train_output_%j.out
#SBATCH --error=train_error_%j.err
#SBATCH --time=24:00:00            # Job duration
#SBATCH --partition=gpu            # GPU partition name
#SBATCH --gres=gpu:1               # Number of GPUs
#SBATCH --cpus-per-task=16          # CPUs per task
#SBATCH --mem=64G                  # RAM

echo "Starting training job on $(hostname) with job ID: $SLURM_JOB_ID, partition: $SLURM_JOB_PARTITION"
echo "Running in directory: $(pwd)"
nvidia-smi
echo "Loading conda environment..."
module load miniconda3
