#!/bin/bash
#SBATCH --job-name=gnlse_tutorial
#SBATCH --partition=gpu,gpu_h200,gpu_rtx6000,gpu_b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=logs/%j_tutorial.out

set -euo pipefail
export PYTHONUNBUFFERED=1

module purge
module load miniconda
conda activate pytorch_env

REPO_DIR="${HOME}/project_pi_lgw23/jz845/rl-physical-control/code/3p1DGNLSE_jax"
cd "${REPO_DIR}"
mkdir -p logs runs

OUT_NB="runs/tutorial_output_${SLURM_JOB_ID}.ipynb"

echo "=== JAX devices ==="
python -c "import jax; print(jax.devices())"

echo "=== Running GNLSE_tutorial.ipynb -> ${OUT_NB} ==="
papermill GNLSE_tutorial.ipynb "${OUT_NB}" --log-output --progress-bar

echo "=== Done ==="
