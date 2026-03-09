#!/bin/bash
#SBATCH --job-name=hippy2d_train
#SBATCH --output=./logs/%A_%a_hippy2d_train.out
#SBATCH --time=10:00:00            # Job duration
#SBATCH --partition=volta-x86      # GPU partition name
#SBATCH --constraint=gpu1_model:Tesla_V100-PCIE-32GB
#SBATCH --cpus-per-task=16         # CPUs per task
#SBATCH --mem=64G                  # RAM
#SBATCH --array=0-3                # Run jobs oversplit 

# Environment
SWEEPID="$1"

if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters"
    exit 1
fi

# Getting the environment ready
echo "Starting training job on $(hostname) with job ID: $SLURM_JOB_ID, partition: $SLURM_JOB_PARTITION"
echo "Running in directory: $(pwd)"
nvidia-smi
echo "Loading conda environment..."
module load miniconda3

# Check if conda environment exists
if conda info --envs | grep -q "hippy2d"; then
    source ~/.bashrc
    echo "Activating conda environment 'hippy2d'..."
    conda activate hippy2d
else
    echo "Conda environment 'hippy2d' not found. Please create it first"
    exit 1
fi


# Run the training script
echo "Running training script. Python=`which python`"

NVIDIA_CUDA_VISIBLE=0 PYTHONPATH="`pwd`" wandb agent --entity "karella" --project "hippy2d"  "$SWEEPID"
