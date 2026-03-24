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
