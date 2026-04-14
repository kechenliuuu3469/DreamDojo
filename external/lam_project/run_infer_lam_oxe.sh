#!/bin/bash
#SBATCH --job-name=lam_infer_oxe
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_outputs/%x/out_%x_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kl0820@princeton.edu

# Usage:
#   sbatch run_infer_lam_oxe.sh [PERCENT] [DATASETS...]
#   ./run_infer_lam_oxe.sh 10 bridge droid    # interactive, first 10%
# Defaults: PERCENT=100, all 7 datasets.

set -euo pipefail

PERCENT="${1:-100}"
shift || true
if [ "$#" -gt 0 ]; then
    DATASETS=("$@")
else
    DATASETS=(bc_z bridge droid fmb fractal furniture_bench taco_play)
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
export CONDA_ENVS_PATH=/scratch/gpfs/AM43/users/kl0820/envs/conda/envs
export CONDA_PKGS_DIRS=/scratch/gpfs/AM43/users/kl0820/envs/conda/pkgs
conda activate dreamdojo_lam

export OMP_NUM_THREADS=4
export OPENCV_NUM_THREADS=2
export MKL_NUM_THREADS=4

cd /scratch/gpfs/AM43/users/kl0820/projects/DreamDojo/external/lam_project
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

mkdir -p slurm_outputs/lam_infer_oxe

CKPT=/scratch/gpfs/AM43/users/kl0820/projects/DreamDojo/external/lam_project/exp_ckpts_joint_all3/last-v1.ckpt
DATASET_ROOT=/scratch/gpfs/AM43/users/kl0820/datasets/oxe_mp4

echo "=========================================="
echo "Start:    $(date)"
echo "Ckpt:     $CKPT"
echo "Percent:  $PERCENT"
echo "Datasets: ${DATASETS[*]}"
echo "=========================================="

python infer_lam_oxe.py \
    --ckpt_path  "$CKPT" \
    --dataset_root "$DATASET_ROOT" \
    --datasets   "${DATASETS[@]}" \
    --percent    "$PERCENT" \
    --batch_size 64 \
    --out_subdir latent_actions_lam

echo "=========================================="
echo "End: $(date)"
echo "=========================================="
