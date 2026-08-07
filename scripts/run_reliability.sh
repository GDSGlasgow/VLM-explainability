#!/bin/bash

#SBATCH --job-name=vlm_reliability
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --time=02-00:00:00
#SBATCH --mail-user=joseph.shingleton@glasgow.ac.uk
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/reliability_job%j.log
#SBATCH --error=logs/reliability_job%j.err
#SBATCH --ntasks=1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vlm-explainability

# SLURM runs a copied script from /var/spool/slurmd, so use the directory from
# which `sbatch` was invoked. Retain a fallback for running this file directly.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO_ROOT="$SLURM_SUBMIT_DIR"
else
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$REPO_ROOT"

python src/reliability_analysis.py \
    --image-root images \
    --prompt prompts/prompt.txt \
    --output results/reliability_analysis.csv \
    --runs 100 \
    --seed 42 \
    "$@"
