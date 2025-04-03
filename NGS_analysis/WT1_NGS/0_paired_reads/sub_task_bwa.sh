#!/bin/bash
#SBATCH -p cpu
#SBATCH --mem=16g
#SBATCH -c 1
#SBATCH -a 1-30
#SBATCH -o task_bwa.out
#SBATCH -e task_bwa.err
#SBATCH -J sub_bwa_coms
#SBATCH -t 03:00:00


CMD=$(sed -n "${SLURM_ARRAY_TASK_ID}p" task_bwa)
echo "Running job $SLURM_ARRAY_TASK_ID with command: $CMD"
eval "$CMD"

