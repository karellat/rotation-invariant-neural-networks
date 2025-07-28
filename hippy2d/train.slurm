#!/bin/bash
#SBATCH --job-name=hippy2d_train
#SBATCH --output=./logs/%j_hippy2d_train.out
#SBATCH --error=./logs/%j_hippy2d_train.err
#SBATCH --time=10:00:00            # Job duration
#SBATCH --partition=volta-x86      # GPU partition name
#SBATCH --gres=gpu:1               # Number of GPUs
#SBATCH --cpus-per-task=16         # CPUs per task
#SBATCH --mem=64G                  # RAM

# Set single NVIDIA GPU
echo "Starting training job on $(hostname) with job ID: $SLURM_JOB_ID, partition: $SLURM_JOB_PARTITION"
echo "Running in directory: $(pwd)"
nvidia-smi
echo "Loading conda environment..."
module load miniconda3

# Check if conda environment exists
if conda info --envs | grep -q "hippy2d"; then
    echo "Activating conda environment 'hippy2d'..."
    source activate hippy2d
else
    echo "Conda environment 'hippy2d' not found. Please create it first"
    exit 1
fi

# Run the training script
echo "Running training script..."
# Set PYTHONPATH to the project root
export PYTHONPATH=$(pwd)
CUDA_VISIBLE_DEVICES=0 python ./scripts/training.py --debug