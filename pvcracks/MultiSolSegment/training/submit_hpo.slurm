#!/bin/bash
#SBATCH --job-name=pv_hpo
#SBATCH --output=logs/hpo_%j.out
#SBATCH --error=logs/hpo_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --partition=gpu  # Change this to your cluster's GPU partition name

# --- Setup ---
# Load necessary modules (Uncomment and adjust if needed)
# module load cuda/11.8
# module load python/3.10

# Activate your conda environment
source ~/.bashrc  # Or path to your shell init
conda activate pvcracks

# Create logs directory if it doesn't exist
mkdir -p logs

# --- Info ---
echo "Job running on node: $SLURM_JOB_NODELIST"
echo "GPUs available: $CUDA_VISIBLE_DEVICES"

# --- Run ---
# Ray will automatically detect the 4 GPUs allocated by Slurm
python pvcracks/multisolsegment/training/train_channeled_unet_cv_hpo.py
