#!/bin/bash
#SBATCH --account=cs_ga_3033_102-2025sp
#SBATCH --partition=c24m170-a100-2
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:a100:2
#SBATCH --job-name=transduction
#SBATCH --requeue

source /home/yb2510/miniconda3/etc/profile.d/conda.sh 
conda activate ARC
python /scratch/yb2510/ARC/transduction/global_script.py
