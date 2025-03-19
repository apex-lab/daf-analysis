#!/bin/bash
#SBATCH --job-name=daf
#SBATCH --output=out/daf_%A_%a.out
#SBATCH --error=out/daf_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --partition=caslake
#SBATCH --account=pi-hcn1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --array=1-32

cd /project/hcn1/daf-analysis
source activate daf
python trf.py $SLURM_ARRAY_TASK_ID
