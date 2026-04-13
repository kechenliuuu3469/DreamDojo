#!/bin/bash
#SBATCH --job-name=lam_eval
#SBATCH --nodes=1
#SBATCH --partition=ailab
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=6:00:00
#SBATCH --output=slurm_outputs/%x/out_%x_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kl0820@princeton.edu

# ============================================
# Environment setup
# ============================================
source "$(conda info --base)/etc/profile.d/conda.sh"
export CONDA_ENVS_PATH=/scratch/gpfs/AM43/users/kl0820/envs/conda/envs
export CONDA_PKGS_DIRS=/scratch/gpfs/AM43/users/kl0820/envs/conda/pkgs
conda activate dreamdojo_lam

# ============================================
# Thread limits
# ============================================
export OMP_NUM_THREADS=4
export OPENCV_NUM_THREADS=2
export MKL_NUM_THREADS=4

# ============================================
# Project setup
# ============================================
cd /scratch/gpfs/AM43/users/kl0820/projects/DreamDojo/external/lam_project
export PYTHONPATH=$(pwd):$PYTHONPATH

mkdir -p slurm_outputs/lam_eval

# ============================================
# Run evaluation
# ============================================
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Start time: $(date)"
echo "=========================================="

python eval_suite/eval_lam_full.py \
    --num_samples 10 \
    --device cuda:0 \
    --output_dir eval_results_test

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
