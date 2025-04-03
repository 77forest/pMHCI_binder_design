#!/bin/bash
#SBATCH -p cpu
#SBATCH --mem=16g
#SBATCH -c 1
#SBATCH -a 1-30
#SBATCH -o task_counts.out
#SBATCH -e task_counts.err
#SBATCH -J sub_ngs_coms
#SBATCH -t 03:00:00


CMD=$(sed -n "${SLURM_ARRAY_TASK_ID}p" task_counts)
echo "Running job $SLURM_ARRAY_TASK_ID with command: $CMD"
eval "$CMD"

