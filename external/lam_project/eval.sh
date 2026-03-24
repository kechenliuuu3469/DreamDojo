#!/bin/bash
#SBATCH --job-name=eval_results
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --gres=gpu:1                      # Number of GPUs
#SBATCH --ntasks=1                        # One process (adjust if your script is multi-proc)
#SBATCH --cpus-per-task=64                 # CPU cores
#SBATCH --mem=80G                         # Memory
#SBATCH --time=24:00:00                   # Time limit (hh:mm:ss)
#SBATCH --output=slurm_outputs/%x/out_%x_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kl0820@princeton.edu
#SBATCH --exclude=neu301,neu306,neu309,neu312


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

cd /n/fs/geniemodel/DreamDojo/external/lam_project
export PYTHONPATH=$(pwd):$PYTHONPATH
export OMP_NUM_THREADS=4
export OPENCV_NUM_THREADS=2

python /n/fs/geniemodel/DreamDojo/external/lam_project/vidwm/metrics/perceptual_metrics.py --lam_eval --num_samples 100 

 