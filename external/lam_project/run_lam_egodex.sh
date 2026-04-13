#!/bin/bash
#SBATCH --job-name=lam_egodex
#SBATCH --nodes=1
#SBATCH --partition=ailab
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00
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
module load proxy/default
export WANDB_API_KEY=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

mkdir -p slurm_outputs/lam_egodex
mkdir -p exp_ckpts_egodex
mkdir -p exp_imgs_egodex

# ============================================
# Run training
# ============================================
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Num GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
echo "GPU type: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Start time: $(date)"
echo "Config: config/lam_egodex.yaml"
echo "Mode: FINE-TUNING from LAM_400k.ckpt"
echo "=========================================="

torchrun \
    --nnodes=1 \
    --nproc_per_node=2 \
    --master_port=$((29500 + RANDOM % 1000)) \
    main.py fit \
    --config config/lam_egodex.yaml \
    --data.num_workers=4 \
    --pretrained_ckpt /scratch/gpfs/AM43/users/kl0820/projects/DreamDojo/checkpoints/LAM_400k.ckpt

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
