#!/bin/bash
#SBATCH --job-name=hippy2d_train
#SBATCH --output=./logs/%j_hippy2d_train.out
#SBATCH --time=10:00:00            # Job duration
#SBATCH --partition=volta-x86      # GPU partition name
#SBATCH --gres=gpu:2               # Number of GPUs
#SBATCH --cpus-per-task=16         # CPUs per task
#SBATCH --mem=64G                  # RAM

# Check the number of parameters 
CONFIG_PATH="$1"

if [ "$#" -gt 1 ]; then
    EPOCHS="$2"
else
    EPOCHS=10
fi
if [ "$#" -gt 2 ]; then
    NUM_SAMPLES="$3"
else
    NUM_SAMPLES=20
fi
if [ "$#" -gt 3 ]; then
    GRACE_PERIOD="$4"
else
    GRACE_PERIOD=5
fi

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
python ./scripts/training_ray.py --search_space "$CONFIG_PATH" --num_epochs "$EPOCHS" --num_samples "$NUM_SAMPLES" --grace_period "$GRACE_PERIOD"