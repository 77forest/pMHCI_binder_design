#README: script for merging all contact residue scores into one .sc file
#Change range of residues and residue numbers throughout script if not 181-190
#Usage: run from head directory (above "/contact" directory) as "python contact_residue_filter.py"
#After running, can filter as:
#cat contact_merged_scores.sc | awk '{if ($1 > 0 && $3 > 0 && $4 > 0 && $5 > 0 && $6 > 0 && $7 > 0 && $8 > 0) {print $NF}}' > good_tags.list
#cat good_tags.list | silentslice collected.silent > rescontact_0.silent
#where "collected.silent" is the silent file of the pdbs originally used to run the contact residue script

import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)

dic = {i: pd.DataFrame() for i in range(181, 190)} #change for peptide residues

# Iterate over the range
for i in range(181, 190): #change for peptide residues
    # Iterate over the directories in the specified path
    for direc in os.listdir('./contact/pep'):
        if direc.startswith('x'):
            file_path = f'./contact/pep/{direc}/contact_{i}/score.sc'
            if os.path.exists(file_path):
                # Read the CSV file, skipping bad lines
                temp = pd.read_csv(file_path, delim_whitespace=True, on_bad_lines='skip', skiprows=[0])
                temp = temp.drop(columns=['SCORE:', 'total_score', 'score', 
                                       'dslf_fa13', 'fa_atr', 'fa_dun_dev', 'fa_dun_rot', 'fa_dun_semi',
                                       'fa_elec', 'fa_intra_atr_xover4', 'fa_intra_elec',
                                       'fa_intra_rep_xover4', 'fa_intra_sol_xover4', 'fa_rep', 'fa_sol',
                                       'hbond_bb_sc', 'hbond_lr_bb', 'hbond_sc', 'hbond_sr_bb', 'hxl_tors',
                                       'lk_ball', 'lk_ball_bridge', 'lk_ball_bridge_uncpl',
                                       'lk_ball_iso', 'omega', 'p_aa_pp','rama_prepro', 'pro_close', 'ref', 'time'])
                temp = temp.rename(columns={'contact_patch': f'contact_patch_{i}'})
                # Concatenate the data
                dic[i] = pd.concat([dic[i], temp], ignore_index=True)

# Optionally, you can print the dictionary to see the results
for key, value in dic.items():
    print(f'Key: {key}, Number of rows: {value.shape[0]}')


dataframes_list = [df for df in dic.values()]
merged_df = dataframes_list[0]
for df in dataframes_list[1:]:
    merged_df = pd.merge(merged_df, df, on='description')

merged_df['description'] = merged_df['description'].apply(lambda x: x[:-5])
#change to range of residue numbers representing your peptide
merged_df = merged_df[['contact_patch_181', 'contact_patch_182',
       'contact_patch_183', 'contact_patch_184', 'contact_patch_185',
       'contact_patch_186', 'contact_patch_187', 'contact_patch_188','contact_patch_189',
     'description']]

merged_df.to_csv('./contact_merged_scores.sc',index=False, sep='\t')
