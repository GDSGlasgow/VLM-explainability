#!/bin/bash 

#SBATCH --job-name=vlm_explainability 
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1 
#SBATCH --nodes=1 
#SBATCH --mem=16G 
#SBATCH --time 00-06:00:00 
#SBATCH --mail-user joseph.shingleton@glasgow.ac.uk 
#SBATCH --mail-type END,FAIL 
#SBATCH --output=logs/vlm_job%j.log
#SBATCH --error=logs/vlm_job%j.err
#SBATCH --ntasks=1

#cd ~/sharedscratch

# load env
source ~/miniconda3/etc/profile.d/conda.sh
conda activate img_segmentation

# SLURM executes a spool copy, so recover the original submission directory.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO_ROOT="$SLURM_SUBMIT_DIR"
else
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$REPO_ROOT"

# Run the Python script
python src/vlm_analysis.py -s gwW1UsKJs_JNiU9Qoa8mpQ -p prompts/prompt.txt
