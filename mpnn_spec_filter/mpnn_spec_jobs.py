#README: this script is used to generate slurm jobs for ProteinMPNN peptide scanning log probability screening; 
#imputs required: pdbs and sequences of designs being evaluated (in this case coming from collected.silent), peptide sequence (9-mer)
#USAGE:
#mkdir mpnn_spec; cd mpnn_spec; mkdir pdbs pepscan
#silentsequence ../collected.silent > collected.seq
#cd pdbs; silentextract ../../collected.silent
#cd ../..; cp mpnn_spec_jobs.py . #this script--change peptide
#python mpnn_spec_jobs.py; sbatch submit_ala_scan.sh 


import sys
import os
import glob
import csv
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

base_dir = './mpnn_spec/pepscan/'
pdbDir = './mpnn_spec/pdbs/'
seqs = './mpnn_spec/collected.seq'

seqs_df = pd.read_csv(seqs, header=None, names=['binder', 'pmhc', 'pdb_name'], delim_whitespace=True)

#change peptide sequence - this script works for 9-mer peptides
peptide = 'TLMSAMTNL' #example peptide (PAP)

# Function to find the start position of the peptide in the sequence
def find_start_site(sequence, peptide):
    start_site = sequence.find(peptide)
    return start_site if start_site != -1 else None

# Create a list to store the data
data = []

# Process each sequence
for idx, row in seqs_df.iterrows():
    binder = row['binder']
    pmhc = row['pmhc']
    sequence = binder + pmhc
    pdb_name = row['pdb_name']

    # Find the start site of the peptide in the sequence
    start_site = find_start_site(sequence, peptide)

    if start_site is not None:
        # Create mutants based on the start site
        mutants = [peptide[i] + str(start_site + i + 1) + 'A' for i in range(len(peptide))]

        for mutant in mutants:
            is_WT = mutant == peptide
            mutant_chain = 'B'  # Assuming the mutant chain is 'B' for all
            data.append({'mutant': mutant, 'is_WT': is_WT, 'wt_pdb': pdb_name, 'mutant_chain': mutant_chain})
    else:
        print(f"Peptide {peptide} not found in sequence for PDB: {pdb_name}")

# Create a DataFrame from the data
df = pd.DataFrame(data)

# Determine the number of CSV files needed
csv_num = math.ceil(len(df) / 180)

# Save the DataFrame to CSV files
for i in range(csv_num):
    df.iloc[i*180:(i+1)*180].to_csv(base_dir + 'pepscan_' + str(i) + '.csv', index=False, columns=df.columns)

csv_files = glob.glob(base_dir + 'pepscan_*.csv')

jobs = []

for csv_file in csv_files:
    csv_name = os.path.basename(csv_file).split('.')[0]
    job = f'../software/mlfold-20230409.sif -u ../software/mpnn/zero_shot_mutation_effect_prediction.py '+\
                    f'--csv_file {csv_file} ' +\
                    f'--folder_with_pdbs {pdbDir} ' +\
                    f'--output_dir {base_dir}proteinmpnn{csv_name} ' +\
                    f'--use_mt_structure 0 ' +\
                    f'--num_seq_per_target 10 '+\
                    f'--batch_size 10 '+\
                    f'--mutant_column mutant '+\
                    f'--mutant_chain_column mutant_chain'
    jobs.append(job)
# with open(base_dir + 'jobs.sh', 'w') as f:
#     f.write('\n'.join(jobs))
#     # os.system('sbatch ' + base_dir + 'jobs.sh')
# jobs

# Write the jobs to the run_order_refiner file
with open(base_dir + 'ala_scan_mpnn_tasks', 'w') as f:
    f.write('\n'.join(jobs))

# Write the SLURM submission script
slurm_script = f"""#!/bin/bash
#SBATCH -p cpu
#SBATCH --mem=16g
#SBATCH -c 1
#SBATCH -a 0-{len(jobs) + 1}
#SBATCH -o o.out
#SBATCH -e e.err
#SBATCH -J ala_scan_mpnn
#SBATCH -t 06:00:00

GROUP_SIZE=1

for I in $(seq 1 $GROUP_SIZE)
do
    J=$(($SLURM_ARRAY_TASK_ID * $GROUP_SIZE + $I - $GROUP_SIZE))
    CMD=$(sed -n "${{J}}p" {base_dir}ala_scan_mpnn_tasks)
    echo "${{CMD}}" | bash
done
"""

# Write the SLURM script to a file
with open('submit_ala_scan.sh', 'w') as f:
    f.write(slurm_script)


