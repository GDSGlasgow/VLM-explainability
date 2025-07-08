#!/bin/bash 

#SBATCH --job-name=vlm_explainability 
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1 
#SBATCH --nodes=1 
#SBATCH --mem=16G 
#SBATCH --time 00-01:00:00 
#SBATCH --mail-user joseph.shingleton@glasgow.ac.uk 
#SBATCH --mail-type END,FAIL 
#SBATCH --output=vlm_job%j.log
#SBATCH --error=vlm_job%j.err
#SBATCH --ntasks=1

#cd ~/sharedscratch

# load env
source ~/miniconda3/etc/profile.d/conda.sh
conda activate img_segmentation

# Run the Python script
 python src/masking.py -s "images/glasgow/gwW1UsKJs_JNiU9Qoa8mpQ" -c "config_files/config.json" 