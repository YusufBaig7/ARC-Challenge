#!/bin/bash
#SBATCH --account=cs_ga_3033_102-2025sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=8th_attempt
#SBATCH --requeue

source /home/yb2510/miniconda3/etc/profile.d/conda.sh 
conda activate ARC
python /scratch/yb2510/ARC/deepseek_first_finetune_3_epoch/complete_script.py
