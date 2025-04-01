#README: This script is used to collect the scores from ProteinMPNN peptide log probability screening and generate a score file by which binder designs can be filtered on; 
#run from directory above '/mpnn_spec/' as 'python mpnn_spec_scores.py'
#how to filter:
#adjust column numbers for filtering by 'log_p_wt' or 'log_p_mt__minus__log_p_wt' and which residues to filter on from mpnn_spec_scores_combined.sc
# cat mpnn_spec_scores_combined.sc | awk '{if ($12 < -0.5 && $13 < -0.5 && $14 < -0.5 && $15 < -0.5 && $16 < -0.5 && $17 < -0.5 && $18 < -0.5 && $19 < -0.5) {print $1}}' > mpnn_spec_tags.list
# cat mpnn_spec_tags.list | silentslice collected.silent > mpnnscan_filter05.silent
#where 'collected.silent' is the silent file of imput designs for ProteinMPNN peptide screening

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
result_csvs = glob.glob(base_dir + 'proteinmpnnpepscan_*/zero_shot_predictions/*.csv')
df_results = pd.concat([pd.read_csv(result_csv) for result_csv in result_csvs])

# Define the peptide sequence - this script works for 9-mer peptides
peptide = 'TLMSAMTNL' #example peptide (PAP)

# Ensure the dataframe is ordered correctly
df_results = df_results.sort_values(by=['wt_pdb', 'mutant']).reset_index(drop=True)

# Generate condensed mutants based on peptide sequence
condensed_mutants = [f"{peptide[i]}{i + 1}A" for i in range(len(peptide))]
condensed_mutants.sort()

print("Condensed Mutants:", condensed_mutants)

# Function to assign condensed mutants in groups of 9 rows
def assign_condensed_mutants(df, condensed_mutants):
    # Ensure df length is multiple of 9 for correct grouping
    assert len(df) % len(condensed_mutants) == 0, "Number of rows must be multiple of 9"

    # Assign condensed mutants in chunks of 9 rows
    chunk_size = len(condensed_mutants)
    for i in range(0, len(df), chunk_size):
        for j in range(chunk_size):
            df.at[i + j, 'condensed_mutant'] = condensed_mutants[j]
    return df

# Assign condensed mutants to df_results
df_results = assign_condensed_mutants(df_results, condensed_mutants)

# Verify the 'condensed_mutant' column
print(df_results[['mutant', 'condensed_mutant']])

# Calculate average score
df_results['avg_score'] = 0.0
unique_wt_pdb = df_results['wt_pdb'].unique()
mutants = df_results['condensed_mutant'].unique()

# Create a dictionary for fast lookup of log_p_mt__minus__log_p_wt values
df_dict = df_results.set_index(['wt_pdb', 'condensed_mutant'])['log_p_mt__minus__log_p_wt'].to_dict()

# Create the NumPy matrix
matrix = np.zeros((len(unique_wt_pdb), len(mutants)))

# Print the dimensions of the matrix
print("Matrix Dimensions:", matrix.shape)

# Initialize a dictionary to store scores for averaging
avg_scores = {wt_pdb: [] for wt_pdb in unique_wt_pdb}

# Populate the matrix and calculate average scores
for i, wt_pdb in enumerate(unique_wt_pdb):
    for j, mutant in enumerate(mutants):
        key = (wt_pdb, mutant)
        if key in df_dict:
            value = df_dict[key]
            matrix[i, j] = value
            avg_scores[wt_pdb].append(value)

# Calculate the average scores and update the DataFrame in one go
avg_score_updates = pd.Series({wt_pdb: np.mean(scores) for wt_pdb, scores in avg_scores.items() if scores})
df_results['avg_score'] = df_results['wt_pdb'].map(avg_score_updates)


# Restructure the DataFrame using pivot_table
pivot_df = df_results.pivot_table(index='wt_pdb',
                                  columns='condensed_mutant',
                                  values=['log_p_mt', 'log_p_wt', 'log_p_mt__minus__log_p_wt'])

# Flatten the multi-index columns
pivot_df.columns = ['_'.join(col) for col in pivot_df.columns.values]

# Reset index to make 'wt_pdb' a regular column
pivot_df = pivot_df.reset_index()

pivot_df.to_csv('./mpnn_spec_scores_combined.sc', index=False, sep='\t')

#how to filter:
#adjust column numbers for filtering by 'log_p_wt' or 'log_p_mt__minus__log_p_wt' and which residues to filter on
# cat alascan_scores_combined.sc | awk '{if ($12 < -0.5 && $13 < -0.5 && $14 < -0.5 && $15 < -0.5 && $16 < -0.5 && $17 < -0.5 && $18 < -0.5 && $19 < -0.5) {print $1}}' > ala_tags.list
# cat ala_tags.list | silentslice collected.silent > alascan_filter05.silent
