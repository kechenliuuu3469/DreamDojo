#!/bin/bash
#SBATCH --job-name=lam_bridge_test
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --gres=gpu:1                      # Number of GPUs
#SBATCH --ntasks=1                        # One process (adjust if your script is multi-proc)
#SBATCH --cpus-per-task=8                 # CPU cores
#SBATCH --mem=80G                         # Memory
#SBATCH --time=24:00:00                   # Time limit (hh:mm:ss)
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
# Thread limits (fixes OpenCV thread errors)
# ============================================
export OMP_NUM_THREADS=4
export OPENCV_NUM_THREADS=2
export MKL_NUM_THREADS=4

# ============================================
# Project setup
# ============================================
cd /scratch/gpfs/AM43/users/kl0820/projects/DreamDojo/external/lam_project
export PYTHONPATH=$(pwd):$PYTHONPATH

mkdir -p slurm_logs

# ============================================
# Run training
# ============================================
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Start time: $(date)"
echo "Config: config/lam_bridge_test.yaml"
echo "=========================================="

python infer_lam.py \
    --ckpt_path /n/fs/geniemodel/DreamDojo/checkpoints/LAM/LAM_400k.ckpt \
    --video_dir /n/fs/geniemodel/DreamDojo/datasets/train \
    --num_videos 100 \
    --extract_full \
    --save_dir lam_inference_output

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
