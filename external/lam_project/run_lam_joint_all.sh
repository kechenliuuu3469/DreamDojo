#!/bin/bash
#SBATCH --job-name=lam_joint_all
#SBATCH --nodes=1
#SBATCH --partition=ailab
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=12:00:00
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

mkdir -p slurm_outputs/lam_joint_all
mkdir -p exp_ckpts_joint_all3
mkdir -p exp_imgs_joint_all3

# ============================================
# Run training
# ============================================
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Num GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
echo "GPU type: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Start time: $(date)"
echo "Config: config/lam_joint_all.yaml"
echo "Mode: FINE-TUNING from LAM_400k.ckpt on all 10 datasets"
echo "=========================================="

# torchrun \
#     --nnodes=1 \
#     --nproc_per_node=4 \
#     --master_port=$((29500 + RANDOM % 1000)) \
#     main.py fit \
#     --config config/lam_joint_all.yaml \
#     --data.num_workers=4 \
#     --pretrained_ckpt /scratch/gpfs/AM43/users/kl0820/projects/DreamDojo/checkpoints/LAM_400k.ckpt

torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    --master_port=$((29500 + RANDOM % 1000)) \
    main.py fit \
    --config config/lam_joint_all.yaml \
    --data.num_workers=4 \
    --ckpt_path /scratch/gpfs/AM43/users/kl0820/projects/DreamDojo/external/lam_project/exp_ckpts_joint_all3/last-v1.ckpt


echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
