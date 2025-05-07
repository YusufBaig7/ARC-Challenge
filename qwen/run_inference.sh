#!/bin/bash
#SBATCH --account=cs_ga_3033_102-2025sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:00:00
#SBATCH --job-name=qwen_inference

source /home/yb2510/miniconda3/etc/profile.d/conda.sh 
conda activate ARC
python /scratch/yb2510/ARC/qwen_inference.py
