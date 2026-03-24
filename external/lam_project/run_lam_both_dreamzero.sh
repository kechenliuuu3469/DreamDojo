#!/bin/bash
#SBATCH --job-name=lam_both
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=72:00:00
#SBATCH --output=slurm_outputs/%x/out_%x_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kl0820@princeton.edu
#SBATCH --exclude=neu301,neu306,neu309,neu312

# ============================================
# Environment setup
# ============================================
module purge
module load anaconda3/2024.02
source "$(conda info --base)/etc/profile.d/conda.sh"

BIG=/n/fs/geniemodel
export CONDA_ENVS_PATH=$BIG/conda/envs
export CONDA_PKGS_DIRS=$BIG/conda/pkgs

if [ ! -d "$BIG/conda/envs/dreamdojo_lam" ]; then
    echo "Creating conda environment dreamdojo_lam..."
    conda create -n dreamdojo_lam python=3.10 -y
    conda activate dreamdojo_lam
    pip install torch torchvision
    pip install lightning einops opencv-python piq numpy Pillow tensorboard
    echo "Environment created successfully."
else
    echo "Environment dreamdojo_lam already exists, activating..."
fi

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
cd $BIG/DreamDojo/external/lam_project
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
echo "=========================================="

rm -rf wandb/

export WANDB_RESUME=never
export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")

torchrun \
    --nnodes=1 \
    --nproc_per_node=8 \
    main.py fit \
    --config config/lam_bridge_droid_full_dreamzero.yaml \
    --data.num_workers=4 \
    --ckpt_path /n/fs/geniemodel/DreamDojo/external/lam_project/exp_ckpts_bridge_droid_full_dreamzero/last.ckpt

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="