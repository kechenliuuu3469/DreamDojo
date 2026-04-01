#!/bin/bash
# ============================================
# LAM Environment Setup
# ============================================
# Usage:
#   source setup_env.sh                    # Use defaults
#   source setup_env.sh /scratch/user      # Custom base dir for conda envs
#
# This script:
#   1. Loads conda (via module or existing install)
#   2. Creates/activates the dreamdojo_lam conda environment
#   3. Installs all dependencies
#   4. Sets recommended thread limits
#   5. Sets PYTHONPATH
#
# After first run, subsequent runs just activate the existing env.
# ============================================

set -e

ENV_NAME="dreamdojo_lam"
PYTHON_VERSION="3.10"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Base directory for conda envs/pkgs (override with $1 or $CONDA_BASE_DIR)
CONDA_BASE="${1:-${CONDA_BASE_DIR:-}}"

# --- Load conda ---
if command -v module &>/dev/null; then
    module purge 2>/dev/null || true
    module load anaconda3/2024.02 2>/dev/null || module load anaconda3 2>/dev/null || true
fi

if ! command -v conda &>/dev/null; then
    echo "ERROR: conda not found. Install Miniconda or load an anaconda module."
    return 1 2>/dev/null || exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

# --- Optional: redirect conda envs to a larger filesystem ---
if [ -n "$CONDA_BASE" ]; then
    export CONDA_ENVS_PATH="$CONDA_BASE/conda/envs"
    export CONDA_PKGS_DIRS="$CONDA_BASE/conda/pkgs"
    mkdir -p "$CONDA_ENVS_PATH" "$CONDA_PKGS_DIRS"
    echo "Conda envs dir: $CONDA_ENVS_PATH"
fi

# --- Create or activate environment ---
if ! conda env list | grep -q "^${ENV_NAME} "; then
    echo "Creating conda environment ${ENV_NAME} (python=${PYTHON_VERSION})..."
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y

    conda activate "$ENV_NAME"

    # Install PyTorch (CUDA 12.8 by default; change URL for other CUDA versions)
    # See https://pytorch.org/get-started/locally/ for options
    echo "Installing PyTorch..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

    # Install remaining dependencies
    echo "Installing LAM dependencies..."
    pip install -r "${SCRIPT_DIR}/requirements.txt"

    echo "Environment ${ENV_NAME} created successfully."
else
    conda activate "$ENV_NAME"
    echo "Activated existing environment: ${ENV_NAME}"
fi

# --- Thread limits (prevents OpenCV/MKL thread explosion on SLURM) ---
export OMP_NUM_THREADS=4
export OPENCV_NUM_THREADS=2
export MKL_NUM_THREADS=4

# --- Project path ---
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

echo "Ready. Python: $(python --version), PyTorch: $(python -c 'import torch; print(torch.__version__)')"
