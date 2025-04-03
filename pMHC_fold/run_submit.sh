#!/bin/bash
sbatch -p gpu -t 04:00:00 --gres=gpu:a4000:1 -a 0-0 pMHC_fold_commands/pMHC_fold_array.submit
