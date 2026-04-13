#!/bin/bash
#SBATCH --job-name=lam_droid_full
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=72:00:00
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

export WANDB_API_KEY=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


mkdir -p slurm_outputs/lam_bridge_full
mkdir -p exp_ckpts_bridge_full
mkdir -p exp_imgs_bridge_full

# ============================================
# Run training
# ============================================
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Num GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
echo "GPU type: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Start time: $(date)"
echo "Config: config/lam_bridge_full.yaml"
echo "Mode: FINE-TUNING from LAM_400k.ckpt on full Bridge V2"
echo "=========================================="

# rm -rf wandb/

# export WANDB_RESUME=never
# export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")

torchrun \
    --nnodes=1 \
    --nproc_per_node=8 \
    main.py fit \
    --config config/lam_droid_full_dreamzero.yaml \
    --data.num_workers=4 \
    --ckpt_path /n/fs/geniemodel/DreamDojo/external/lam_project/exp_ckpts_bridge_droid_full_dreamzero/last.ckpt

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="