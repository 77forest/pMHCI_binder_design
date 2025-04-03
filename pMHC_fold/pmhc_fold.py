"""A file for folding pMHCs and de novo proteins.

"""

import dataclasses
import datetime
import glob
import os
import re
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, List
import sys
import pickle
import json

sys.path.insert( 0, '../../af2exp')

import numpy as np
import pandas as pd

from alphafold.data.parsers import TemplateHit, parse_hhr

from alphafold.data.tools.hhsearch import HHSearch
from alphafold.data.tools.hhblits import HHBlits
from alphafold.data import templates
import util

import random

import collections
from timeit import default_timer as timer
import argparse

import io
from Bio import PDB
from Bio.PDB.Polypeptide import PPBuilder
from Bio.PDB import PDBParser
from Bio.PDB.mmcifio import MMCIFIO
from Bio import SeqIO
from Bio.SeqUtils import seq1

from Bio import pairwise2
from Bio.SubsMat import MatrixInfo as matlist

from absl import logging
from alphafold.common import residue_constants
from alphafold.data import mmcif_parsing
from alphafold.data import parsers
from alphafold.data.tools import kalign
from alphafold.common import protein
from alphafold.data import pipeline
from alphafold.data import templates
from alphafold.model import data
from alphafold.model import config
from alphafold.model import model
from alphafold.data.tools import hhsearch
from alphafold.relax import relax

from alphafold.common.protein import from_pdb_string
from alphafold.common.residue_constants import restypes

from jax import random
from jax import jit
import jax
import jax.numpy as jnp
import haiku as hk


def get_template_features(
    mmcif_object: mmcif_parsing.MmcifObject,
    pdb_id: str,
    template_chain_id: str,
    kalign_binary_path: str):
    ''''''
    
    # since we are giving exact template
    template_sequence = query_sequence = original_query_sequence = mmcif_object.chain_to_seqres[template_chain_id]
    indices = make_trivial_indices(query_sequence, 0, [])

    mapping = templates._build_query_to_hit_index_mapping(
        query_sequence, template_sequence, indices, indices,
        original_query_sequence)

    features, realign_warning = templates._extract_template_features(
        mmcif_object=mmcif_object,
        pdb_id=pdb_id,
        mapping=mapping,
        template_sequence=template_sequence,
        query_sequence=query_sequence,
        template_chain_id=template_chain_id,
        kalign_binary_path=kalign_binary_path)
    
    
    return features, realign_warning


def make_trivial_indices(sequence: str, start_index: int, indices_list: List[int]):
    """Computes the relative indices for each residue with respect to the original sequence."""
    counter = start_index
    for symbol in sequence:
        if symbol == '-':
            indices_list.append(-1)
        else:
            indices_list.append(counter)
        counter += 1
        
    return indices_list


def combine_chains(features_list, new_name):
    all_chain_features = {}
    all_chain_features['template_all_atom_positions'] = np.concatenate(
        [feature_dict['template_all_atom_positions'] for feature_dict in features_list],
        axis=0)
    
    all_chain_features['template_all_atom_masks'] = np.concatenate(
        [feature_dict['template_all_atom_masks'] for feature_dict in features_list],
        axis=0)
    
    all_chain_features['template_sequence'] = ''.encode().join(
        [feature_dict['template_sequence'] for feature_dict in features_list])
    
    all_chain_features['template_aatype'] = np.concatenate(
        [feature_dict['template_aatype'] for feature_dict in features_list],
        axis=0)
    
    all_chain_features['template_domain_names'] = [new_name.encode()]
    
    return (all_chain_features)


def run_alphafold_prediction(pdb_name: str,
                             query_sequence: str,
                             msa: list,
                             deletion_matrix: list, 
                             chainbreak_sequence: str,
                             template_features: dict,
                             out_prefix: str):
    '''msa should be a list. If single seq is provided, it should be a list of str.'''
    # gather features for running with only template information
    feature_dict = {
        **pipeline.make_sequence_features(sequence=query_sequence,
                                          description="none",
                                          num_res=len(query_sequence)),
        **pipeline.make_msa_features(msas=[msa],
                                     deletion_matrices=[deletion_matrix]),
        **template_features
    }

    # add big enough number to residue index to indicate chain breaks

    # Ls: number of residues in each chain
    # Ls = [ len(split) for split in chainbreak_sequence.split('/') ]
    Ls = [ len(split) for split in chainbreak_sequence.split('/') ]
    idx_res = feature_dict['residue_index']
    L_prev = 0
    for L_i in Ls[:-1]:
        idx_res[L_prev+L_i:] += 200
        L_prev += L_i
    feature_dict['residue_index'] = idx_res
    
    print('Beginning prediction for', out_prefix.split('/')[-1])
    plddts, prediction_result_dict, time = predict_structure(out_prefix, feature_dict, do_relax=False)
    
    print('Done predicting', out_prefix.split('/')[-1])
    
    return plddts, prediction_result_dict, time


def predict_structure(prefix, feature_dict, do_relax=True, random_seed=0):  
    """Predicts structure using AlphaFold for the given sequence."""

    # Run the models.
    plddts = []
    unrelaxed_pdb_lines = []
    relaxed_pdb_lines = []
    
    prediction_result_dict = {}
    for model_name, params in model_params.items():
        start = timer()
        print(f"running {model_name}")
        # swap params to avoid recompiling
        # note: models 1,2 have diff number of params compared to models 3,4,5
        if any(str(m) in model_name for m in [1,2]): model_runner = model_runner_1
        if any(str(m) in model_name for m in [3,4,5]): model_runner = model_runner_3
        model_runner.params = params

        processed_feature_dict = model_runner.process_features(feature_dict, random_seed=random_seed)
        prediction_result = model_runner.predict(processed_feature_dict)
        prediction_result_dict[model_name] = prediction_result
        
        unrelaxed_protein = protein.from_prediction(processed_feature_dict,prediction_result)
        unrelaxed_pdb_lines.append(protein.to_pdb(unrelaxed_protein))
        plddts.append(prediction_result['plddt'])

        time = timer() - start
        print( f"{model_name} Time: {time} " )
        if do_relax:
            # Relax the prediction.
            amber_relaxer = relax.AmberRelaxation(max_iterations=0,tolerance=2.39,
                                                stiffness=10.0,exclude_residues=[],
                                                max_outer_iterations=20)      
            relaxed_pdb_str, _, _ = amber_relaxer.process(prot=unrelaxed_protein)
            relaxed_pdb_lines.append(relaxed_pdb_str)

    # rerank models based on predicted lddt
    lddt_rank = np.arange(len(plddts))#np.mean(plddts,-1).argsort()[::-1]
    plddts_ranked = {}
    for n,r in enumerate(lddt_rank):
#         print(f"model_{n+1} {np.mean(plddts[r])}")

        unrelaxed_pdb_path = f'{prefix}_unrelaxed_model_{n+1}.pdb'    
        with open(unrelaxed_pdb_path, 'w') as f: f.write(unrelaxed_pdb_lines[r])

        if do_relax:
            relaxed_pdb_path = f'{prefix}_relaxed_model_{n+1}.pdb'
            with open(relaxed_pdb_path, 'w') as f: f.write(relaxed_pdb_lines[r])

        plddts_ranked[f"model_{n+1}"] = plddts[r]
  
    return plddts_ranked, prediction_result_dict, time


def add_align_dict(algn_dict_mhc, algn_dict_peptide, len_pep_query, len_pep_target):
    algn_dict = {(k+len_pep_query):(v+len_pep_target) for k,v in algn_dict_mhc.items()}
    algn_dict = {**algn_dict_peptide, **algn_dict}
    return algn_dict

def aa_seq_from_aatype(aatype):
    return ''.join([restypes[i] for i in aatype])

def combine_protein_objects(protein_list):
    attr_list = ['atom_positions', 'atom_mask', 'aatype',]
    return [np.concatenate([getattr(obj, attr) for obj in protein_list], axis=0) for attr in attr_list ]

def protein_to_template_features(protein_obj, new_name):
    all_chain_features = {}
    all_chain_features['template_all_atom_positions'] = protein_obj[0]

    all_chain_features['template_all_atom_masks'] = protein_obj[1]
    all_chain_features['template_sequence'] = aa_seq_from_aatype(protein_obj[2])

    all_chain_features['template_aatype'] = protein_obj[2]

    all_chain_features['template_domain_names'] = [new_name.encode()]
    
    return all_chain_features

def extract_protein_sequences(pdb_file):
    parser = PDBParser()
    structure = parser.get_structure("protein", pdb_file)

    sequences = {}

    for model in structure:
        for chain in model:
            chain_id = chain.get_id()
            sequence = ''
            for residue in chain:
                if residue.get_id()[0] == ' ':  # Ignore hetero residues
                    sequence += seq1(residue.get_resname())  # Convert to one-letter code
            sequences[chain_id] = sequence

    return sequences

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-mb', '--minib_dir', type=str, help='Path to directory of template pdb files.')
    parser.add_argument('-qu', '--query', type=str, help='Path to query .seq file.')
    parser.add_argument('-mhc', '--mhc_dir', type=str, default='./template_pdbs/', help='Path to directory of mhc templates.')
    parser.add_argument('--alleles', nargs='+', help='List of alleles to fold, must match with peptides. Give in format of alleles separated by a space')
    parser.add_argument('--peptides', nargs='+', help='List of peptides to fold, must match with alleles. Give in format of peptides separated by a space')
    parser.add_argument('--df', type=str, help='Path to the .csv file containing the data to fold.')
    parser.add_argument('-od', '--out_dir', type=str, required=True, 
                        help='Path to directory in which predictions are saved.')
    parser.add_argument('-sm', '--seq_mode', type=str, default='single_seq', choices=['single_seq', 'msa'],   
                        help='Whether to use single sequence or msa as sequence input to the model.')

    parser.add_argument('-nr', '--num_recycle', type=int, default=3,
                        help='Number of recycle iterations to perform.')
    parser.add_argument('-mw', '--model_weights_path', type=str, 
                        help='path to model weights pkl file')

    parser.add_argument('--pretrain', action='store_true',
                        help='Set if desired to run the pre-trained model.')
    
    parser.add_argument('-mn', '--model_name', type=str, default='model_1_ptm', 
                        help='AF2 model name. Use model_1_ptm if using templates.')

    parser.add_argument('-bi', '--batch_index', type=int, default=None,
                        help='Split the pandas df on this index.')

    return parser

def get_args(parser, argument=None):
    if argument:
        args = parser.parse_args(argument.split())
    else:
        args = parser.parse_args()

    # resolving the cases when user forgets to provide msa_dir
    if args.seq_mode=='msa' and args.msa_dir==None:
        raise ValueError('Please provide msa directory when specifying msa as seq_mode.')

    return args


def parse_pdb(filename, **kwargs):
    '''extract xyz coords for all heavy atoms'''
    lines = open(filename,'r').readlines()
    return parse_pdb_lines(lines, **kwargs)

def parse_pdb_lines(lines, parse_hetatom=False, ignore_het_h=True, **kwargs):
    lines = [l for l in lines if l[:4]=="ATOM"]
    chain_ids = kwargs['chain_ids']
    combined_out = {}
    for chain_id in chain_ids:
        chain_lines = [l for l in lines if l[21] == chain_id]
        # indices of residues observed in the structure
        res = [(l[22:26],l[17:20]) for l in chain_lines if l[:4]=="ATOM" and l[12:16].strip()=="CA"]
        seq = [residue_constants.restype_3to1[r[1]] if r[1] in residue_constants.restype_3to1.keys() else 20 for r in res]
        pdb_idx = [( l[21:22].strip(), int(l[22:26].strip()) ) for l in chain_lines if l[:4]=="ATOM" and l[12:16].strip()=="CA"]  # chain letter, res num

        # 4 BB + up to 10 SC atoms
        xyz = np.full((len(res), 14, 3), np.nan, dtype=np.float32)
        for l in chain_lines:
            if l[:4] != "ATOM":
                continue
            chain, resNo, atom, aa = l[21:22], int(l[22:26]), l[12:16], l[17:20]
            idx = pdb_idx.index((chain,resNo))
            for i_atm, tgtatm in enumerate(util.aa2long[util.aa2num[aa]]):
                if tgtatm == atom:
                    xyz[idx,i_atm,:] = [float(l[30:38]), float(l[38:46]), float(l[46:54])]
                    break

        # save atom mask
        mask = np.logical_not(np.isnan(xyz[...,0]))
        #xyz[np.isnan(xyz[...,0])] = 0.0

        out = {'xyz':xyz, # cartesian coordinates, [Lx14]
                'mask':mask, # mask showing which atoms are present in the PDB file, [Lx14]
                'idx':np.array([i[1] for i in pdb_idx]), # residue numbers in the PDB file, [L]
                'seq':''.join(seq), # amino acid sequence, [L]
                'pdb_idx': pdb_idx,  # list of (chain letter, residue number) in the pdb file, [L]
               }
            
        combined_out[chain_id] = out

    return combined_out


def calc_U(xyz1, xyz2):
    """
    Calculate the rotation matrix that is used for superimposing two chains during rmsd computation.
    """
    # center to CA centroid  
    xyz1 = xyz1 - xyz1.mean(0)
    xyz2 = xyz2 - xyz2.mean(0)
    
    # Computation of the covariance matrix
    C = xyz2.T @ xyz1

    # Compute optimal rotation matrix using SVD
    V, S, W = np.linalg.svd(C)

    d = np.ones([3,3])
    d[:,-1] = np.sign(np.linalg.det(V)*np.linalg.det(W))

    # Rotation matrix U
    U = (d*V) @ W

    return U


def get_interface_pairs(xyz_chain1: np.array, xyz_chain2: np.array) -> list:
    dist = np.linalg.norm(xyz_chain1.reshape(-1, 3)[:, None] - xyz_chain2.reshape(-1, 3)[None, :], axis=-1)
    i,j = np.where(np.logical_and(dist <= 5.0, dist > 3.0))

    pairs = list(set([(a//xyz_chain1.shape[1], b//xyz_chain1.shape[1]) for a,b in zip(i,j)]))
    return pairs


parser = make_parser()
args = get_args(parser)


num_recycle = args.num_recycle


if args.pretrain:
    if "model_params" not in dir(): model_params = {}
else:
    model_params = {}
for model_name in [args.model_name]:
    if model_name not in model_params:
        model_config = config.model_config(model_name)
        model_config.data.eval.num_ensemble = 1

        model_config.data.common.num_recycle = num_recycle
        model_config.model.num_recycle = num_recycle

        if args.pretrain:
            model_params[model_name] = data.get_model_haiku_params(model_name=model_name, data_dir="./") # CHANGE THIS (directory where "params/ folder is")
        else:
            full_model_params = pickle.load(open(args.model_weights_path, "rb")) # CHANGE THIS (directory where "params/ folder is")
            binder_model_params, af2_model_params = hk.data_structures.partition(lambda m, n, p: m[:9] != "alphafold", full_model_params)
            model_params[args.model_name] = af2_model_params
            print(f'loaded model from {args.model_weights_path}')
        if model_name in ['model_1_ptm', 'model_1', 'model_2_ptm', 'model_2']:
            model_runner_1 = model.RunModel(model_config, model_params[model_name])
        if model_name in ['model_3_ptm', 'model_3', 'model_4_ptm', 'model_4', 'model_5_ptm', 'model_5']:
            model_runner_3 = model.RunModel(model_config, model_params[model_name])


out_dir = args.out_dir
print('out_dir:', out_dir)

alignment_df = pd.read_csv('../../alignments_all_alleles_vs_pdb_June29_2024.csv')


mhc_template_dir = args.mhc_dir
minib_query_structures = args.minib_dir
df_sample = pd.read_csv(args.df)

n_templates = 4

minb = True

metrics_df = pd.DataFrame(columns=['pae_total', 'plddt_total', 'ptm', 'ppi_pae_int', 'ppi_pae_int_peptide', 'ppi_pae_int_mhc',
                                   'pae_peptide', 'plddt_peptide', 'pae_mhc', 'plddt_mhc', 'pae_binder', 'plddt_binder', 'time', 'description'])


if os.path.exists(out_dir + 'metrics.sc'):
    temp_df = pd.read_csv(out_dir + 'metrics.sc')
    # filter sample_df by those tags that are not in temp_df
    df_sample = df_sample[~df_sample['description'].isin(temp_df['description'])]
    print(len(df_sample))
    print(f'Skipping {len(temp_df)} entries that have already been processed.')
    
if not os.path.exists(out_dir + 'metrics.sc'):
    metrics_df.to_csv(out_dir + 'metrics.sc', index=False)

TEMPLATE_FEATURES = {
    'template_aatype': np.float32,
    'template_all_atom_masks': np.float32,
    'template_all_atom_positions': np.float32,
    'template_domain_names': object,
    'template_sequence': object,
    'template_sum_probs': np.float32,
}

problemmatic = []
loop_count=0

print('Total queries:', len(df_sample))
for index, row in df_sample.iterrows():
    all_template_features = {}
    for template_feature_name in TEMPLATE_FEATURES:
        all_template_features[template_feature_name] = []
    
    templates_msa = []
    templates_deletion_matrix = []
    
    allele_name = row['allele_msa_file_names']   # fake pdbname
    
    peptide_seq_query = row['peptide_seq']
    mhc_seq = row['allele_sequence']
    minibinder_seq = row['target_chainseq'].split('/')[1]

    chainbreak_sequence = minibinder_seq + '/' + mhc_seq + '/' + peptide_seq_query

    print('Allele query:', allele_name)
    print('Query seq:', chainbreak_sequence)

    query_sequence = minibinder_seq + mhc_seq + peptide_seq_query


    num_res = len(query_sequence)
    entry_index = row['entry_index']
    
    description = row['description']
    
    pdb_name = row['pdb_name']

    cols = ['pdbid_template', 'alignstring_mhc', 'peptide_seq_template', 'mhc_seq_query', 'mhc_seq_template']
    

    subset = alignment_df[alignment_df['allele_name_query'] == allele_name]
    subset = subset[subset['peptide_seq_template'].apply(len) == len(peptide_seq_query)]
    blosum62 = matlist.blosum62
    alignment_scores = []
    for peptide in subset['peptide_seq_template']:
        alignments = pairwise2.align.globalds(peptide_seq_query, peptide, blosum62, -10, -0.5)
        best_alignment = alignments[0]
        score = best_alignment[2]  # Extract the alignment score
        alignment_scores.append(score)
    subset['peptide_alignment_score'] = alignment_scores
    subset['combined_alignment_score'] = (subset['mhc_alignment_score'] / len(mhc_seq)) + (subset['peptide_alignment_score'] / len(peptide_seq_query))
    subset = subset.sort_values(by='combined_alignment_score', ascending=False)
    subset = subset.iloc[0:n_templates, :]  

    count = 0
    for _, line in subset.iterrows():
        count += 1
        (template_pdbid, target_to_template_alignstring_mhc, 
         peptide_seq_template, mhc_seq_query, mhc_seq_template) = line[cols]
        target_len, template_len = len(query_sequence), len(peptide_seq_template + mhc_seq_template + minibinder_seq)
        print('Template:', template_pdbid, peptide_seq_template)

        # get align_dict for peptide on the fly
        try:
            if len(peptide_seq_query) == len(peptide_seq_template):
                algn_dict_peptide = {i:i for i in range(len(peptide_seq_query))}
            else:
                algn_dict_peptide = {}
                for i in range(3):
                    algn_dict_peptide[i] = i
                    algn_dict_peptide[len(peptide_seq_query)-1-i] = len(peptide_seq_template)-1-i
        except:
            problemmatic.append(row_pdb.pdbid)
            continue

        assert target_len == num_res
        assert query_sequence == minibinder_seq + mhc_seq + peptide_seq_query

        query_structure_path = minib_query_structures + pdb_name + '.pdb'
        mhc_structure_filename = [filename for filename in os.listdir('../../' + mhc_template_dir) if filename.startswith(template_pdbid) and filename.endswith('.pdb')][0]
        mhc_structure_path = os.path.join('../../' + mhc_template_dir, mhc_structure_filename)

        
        # Load in the query and template pdb structures
        query_pdb_list = []
        with open('../../' + minib_query_structures + pdb_name + '.pdb') as qf:
            query_pdbstr = qf.read()
            for chain in ['A', 'B']:
                query_pdb_list.append(from_pdb_string(query_pdbstr, chain))
                tests = from_pdb_string(query_pdbstr, chain)
            query_pmhc = combine_protein_objects([query_pdb_list[1]])
            query_pmhc_temp = protein_to_template_features(query_pmhc, 'query_mhc')
            query_minib = combine_protein_objects([query_pdb_list[0]])
            query_minib_temp = protein_to_template_features(query_minib, 'query_minib')
            query_all = combine_protein_objects([query_pdb_list[0], query_pdb_list[1]])
            query_all_temp = protein_to_template_features(query_all, 'query_all')


            all_positions_query_pmhc, all_positions_mask_query_pmhc = list(map(query_pmhc_temp.get, ['template_all_atom_positions', 'template_all_atom_masks']))
            all_positions_query_minib, all_positions_mask_query_minib = list(map(query_minib_temp.get, ['template_all_atom_positions', 'template_all_atom_masks']))
            all_positions_query_all, all_positions_mask_query_all = list(map(query_all_temp.get, ['template_all_atom_positions', 'template_all_atom_masks']))
        
            all_positions_query_mhc, all_positions_mask_query_mhc = all_positions_query_pmhc[:-len(peptide_seq_query)], all_positions_mask_query_pmhc[:-len(peptide_seq_query)]           
            
            
        template_pdb_list = []
        with open(mhc_structure_path) as tf:
            template_pdbstr = tf.read()
            for chain in ['A', 'B']:
                template_pdb_list.append(from_pdb_string(template_pdbstr, chain))
            template_mhc = combine_protein_objects([template_pdb_list[0]])
            template_mhc_temp = protein_to_template_features(template_mhc, 'template_mhc')
            template_pmhc = combine_protein_objects([template_pdb_list[0], template_pdb_list[1]])
            template_pmhc_temp = protein_to_template_features(template_pmhc, 'template_mhc')

            all_positions_template_mhc, all_positions_mask_template_mhc, template_mhc_seqs = list(map(template_mhc_temp.get, ['template_all_atom_positions', 'template_all_atom_masks', 'template_sequence']))
            all_positions_template_pmhc, all_positions_mask_template_pmhc, template_pmhc_seqs = list(map(template_pmhc_temp.get, ['template_all_atom_positions', 'template_all_atom_masks', 'template_sequence']))
            
                
        all_positions_query_mhc_trunc, all_positions_mask_query_mhc_trunc, mhc_seq_trunc = all_positions_query_mhc[:len(all_positions_template_mhc)], all_positions_mask_query_mhc[:len(all_positions_mask_template_mhc)], mhc_seq[:len(template_mhc_seqs)]         
        all_positions_query_pmhc_trunc, all_positions_mask_query_pmhc_trunc, query_pmhc_seq_trunc = np.concatenate([all_positions_query_mhc_trunc, all_positions_query_pmhc[-len(peptide_seq_query):]], axis=0), np.concatenate([all_positions_mask_query_mhc_trunc, all_positions_mask_query_pmhc[-len(peptide_seq_query):]], axis=0), mhc_seq_trunc + peptide_seq_query
        all_positions_template_pmhc_trunc, all_positions_mask_template_pmhc_trunc, template_pmhc_seqs_trunc = np.concatenate([all_positions_template_mhc[:len(all_positions_query_mhc_trunc)], all_positions_template_pmhc[-len(peptide_seq_query):]], axis=0), np.concatenate([all_positions_mask_template_mhc[:len(all_positions_query_mhc_trunc)], all_positions_mask_template_pmhc[-len(peptide_seq_query):]], axis=0), template_mhc_seqs[:len(mhc_seq_trunc)] + template_pmhc_seqs[-len(peptide_seq_query):]
        
        minib_xyz, mask_minib = all_positions_query_minib, all_positions_mask_query_minib
        query_xyz_pmhc, mask_q = all_positions_query_pmhc_trunc, all_positions_mask_query_pmhc_trunc
        template_xyz_pmhc, mask_t = all_positions_template_pmhc_trunc, all_positions_mask_template_pmhc_trunc
            
        n_res, n_atom, n_xyz = query_xyz_pmhc.shape
        assert (n_res, n_atom, n_xyz) == template_xyz_pmhc.shape and n_xyz==3

        mask_t = ~np.isnan(template_xyz_pmhc)
        mask_template = mask_t.reshape(-1,3).all(-1)

        mask_q = ~np.isnan(query_xyz_pmhc)
        mask_query = mask_q.reshape(-1,3).all(-1)
             
        xyz_template_nonan = template_xyz_pmhc.reshape(-1,3)[mask_template & mask_query]
        xyz_query_nonan = query_xyz_pmhc.reshape(-1,3)[mask_template & mask_query]

        U = calc_U(xyz_template_nonan,xyz_query_nonan) # structure in 2nd argument is moved onto the 1st
        
        minib_xyz_mean = minib_xyz - query_xyz_pmhc.mean(0)
        all_positions_template_pmhc_mean = all_positions_template_pmhc - all_positions_template_pmhc.mean(0)
        minib_xyz_rot = minib_xyz_mean @ U
        aligned_pmhc = all_positions_template_pmhc_mean
                

        aligned_minib_pmhc = np.concatenate((minib_xyz_rot, aligned_pmhc), axis=0)
        template_full_sequence = minibinder_seq + template_pmhc_seqs
  
        mask = np.concatenate([query_pdb_list[0].atom_mask, all_positions_mask_template_pmhc], axis=0)    

        all_positions_tmp, all_positions_mask_tmp = aligned_minib_pmhc, mask
    

        algn_dict_mhc = {int(x.split(':')[0]) : int(x.split(':')[1]) # 0-indexed
                         for x in target_to_template_alignstring_mhc.split(';')[:-1]}
        algn_dict_combined = add_align_dict(algn_dict_mhc, algn_dict_peptide, len(peptide_seq_query), len(peptide_seq_template))


        target_start = max(algn_dict_combined.keys())+1
        template_start = max(algn_dict_combined.values())+1
        minibinder_dict =  {target_start + i: template_start + i for i in range(len(minibinder_seq))}
        algn_dict_combined.update(minibinder_dict)

        
        target_to_template_alignstring = ''.join([f'{k}:{v};' for k,v in algn_dict_combined.items()])

        tmp2query = {int(x.split(':')[1]) : int(x.split(':')[0]) # 0-indexed
                     for x in target_to_template_alignstring.split(';')[:-1]}

        identities = sum(template_full_sequence[i] == query_sequence[j] for i, j in tmp2query.items())

        all_positions = np.zeros([num_res, residue_constants.atom_type_num, 3])
        all_positions_mask = np.zeros([num_res, residue_constants.atom_type_num],
                                      dtype=np.int64)

        template_alseq = ['-']*num_res
        for i,j in tmp2query.items(): # i=template, j=query
            template_alseq[j] = template_full_sequence[i]
            all_positions[j] = all_positions_tmp[i]
            all_positions_mask[j] = all_positions_mask_tmp[i]

        template_sequence = ''.join(template_alseq)
        assert len(template_sequence) == len(query_sequence) == num_res
        assert identities == sum(a==b for a,b in zip(template_sequence, query_sequence))

        template_aatype = residue_constants.sequence_to_onehot(
            template_sequence, residue_constants.HHBLITS_AA_TO_ID)

        all_template_features['template_all_atom_positions'].append(all_positions)
        all_template_features['template_all_atom_masks'].append(all_positions_mask)
        all_template_features['template_sequence'].append(template_sequence.encode())
        all_template_features['template_aatype'].append(template_aatype)
        all_template_features['template_domain_names'].append(f'{pdb_name}_{mhc_structure_filename}'.encode())
        all_template_features['template_sum_probs'].append([identities])


        query2tmp = {y:x for x,y in tmp2query.items()}
        deletion_vec = []
        alseq = ''
        last_alpos = -1
        for i in range(len(query_sequence)):
            j = query2tmp.get(i,None)
            if j is not None:
                alseq += template_full_sequence[j]
                deletion = j - (last_alpos+1)
                deletion_vec.append(deletion)
                last_alpos = j
            else:
                alseq += '-'
                deletion_vec.append(0)
        templates_msa.append(alseq)
        templates_deletion_matrix.append(deletion_vec)

        total_deletions = sum(deletion_vec)
        last_aligned_tmp = max(tmp2query.keys())
        naligned = len(tmp2query.keys())

        assert last_aligned_tmp+1 == naligned + total_deletions


    all_identities = np.array(all_template_features['template_sum_probs'])[:,0]
    reorder = np.arange(all_identities.shape[0])
    for name in all_template_features:
        all_template_features[name] = np.stack(
            [all_template_features[name][x] for x in reorder], axis=0).astype(
                templates.TEMPLATE_FEATURES[name])

    
    if args.seq_mode == 'single_seq':
        msa=[query_sequence]
        deletion_matrix=[[0]*len(query_sequence)]
    
        
    allele_name_no_colon = allele_name.replace(':', '')
    temp = f'{out_dir}/{pdb_name}_{allele_name_no_colon}_{peptide_seq_query}'
    out_prefix = temp.replace("*", "")
    print('Writing prediciton to', out_prefix)
    

    plddts, prediction_result_dict, time = run_alphafold_prediction(pdb_name=allele_name,
                                                            query_sequence=query_sequence,
                                                            msa=msa,
                                                            deletion_matrix=deletion_matrix,
                                                            chainbreak_sequence=chainbreak_sequence,
                                                            template_features=all_template_features,
                                                            out_prefix=out_prefix)
    

    metrics = {'pae_total': [], 'plddt_total': [], 'ptm': [],
               'pae_int': [], 'pae_int_peptide': [], 'pae_int_mhc': [], 'pae_peptide': [], 'plddt_peptide': [],
               'pae_mhc': [], 'plddt_mhc': [], 'pae_binder': [], 'plddt_binder': [], 'time': [], 'description': []}

    pdb = parse_pdb(out_prefix + '_unrelaxed_model_1.pdb', chain_ids=['A'])

    keys_to_remove = ['distogram', 'experimentally_resolved', 'masked_msa', 'predicted_lddt', 'structure_module', 'aligned_confidence_probs', 'max_predicted_aligned_error', 'pae_logits']


    for key in keys_to_remove:
        prediction_result_dict['model_2_ptm'].pop(key)    

  
    np.savez('{}_pred_results.npz'.format(out_prefix), prediction_result_dict)
    np.savez('{}_templates_array.npz'.format(out_prefix), all_template_features)
    
    xyz_a = pdb['A']['xyz']
    xyz_minib = xyz_a[:len(minibinder_seq)]
    xyz_pmhc = xyz_a[len(minibinder_seq):]
    xyz_peptide = xyz_a[-len(peptide_seq_query):]
    xyz_mhc = xyz_a[len(minibinder_seq):-len(peptide_seq_query)]
    
    pae = prediction_result_dict['model_2_ptm']['predicted_aligned_error']
    plddt = prediction_result_dict['model_2_ptm']['plddt']
    ptm = prediction_result_dict['model_2_ptm']['ptm'].item(0)

    pae_peptide = np.mean(pae[-len(peptide_seq_query):, -len(peptide_seq_query):])
    plddt_peptide = np.mean(plddt[-len(peptide_seq_query):])
    pae_mhc = np.mean(pae[len(minibinder_seq):-len(peptide_seq_query), len(minibinder_seq):-len(peptide_seq_query)])
    plddt_mhc = np.mean(plddt[len(minibinder_seq):-len(peptide_seq_query)])
    pae_binder = np.mean(pae[len(minibinder_seq):, len(minibinder_seq):])
    plddt_binder = np.mean(plddt[len(minibinder_seq):])



    ppi_pae_interact = (np.mean(pae[len(minibinder_seq):, :len(minibinder_seq)]) + np.mean(pae[:len(minibinder_seq), len(minibinder_seq):])) / 2
    ppi_pae_interact_pep = (np.mean(pae[-len(peptide_seq_query):, :len(minibinder_seq)]) + np.mean(pae[:len(minibinder_seq), -len(peptide_seq_query):])) / 2
    ppi_pae_interact_mhc = (np.mean(pae[len(minibinder_seq):-len(peptide_seq_query), :len(minibinder_seq)]) + np.mean(pae[:len(minibinder_seq), len(minibinder_seq):-len(peptide_seq_query)])) / 2


    metrics['pae_total'].append(np.mean(pae))
    metrics['plddt_total'].append(np.mean(plddt))
    metrics['ptm'].append(ptm)

    metrics['pae_int'].append(ppi_pae_interact)
    metrics['pae_int_peptide'].append(ppi_pae_interact_pep)
    metrics['pae_int_mhc'].append(ppi_pae_interact_mhc)

    metrics['pae_peptide'].append(pae_peptide)
    metrics['plddt_peptide'].append(plddt_peptide)
    metrics['pae_mhc'].append(pae_mhc)
    metrics['plddt_mhc'].append(plddt_mhc)
    metrics['pae_binder'].append(pae_binder)
    metrics['plddt_binder'].append(plddt_binder)
    metrics['time'].append(time)
    metrics['description'].append(out_prefix.split('/')[-1])

    
    
    metrics_to_df = pd.DataFrame(metrics)
    metrics_to_df = metrics_to_df.round(3)
    metrics_to_df.to_csv(out_dir + '/metrics.sc', mode='a', index=False, header=False)   

    loop_count += 1
    print(f'Done predicting {loop_count}/{len(df_sample)} sequences.\n')

    if problemmatic:
        print(f'problemmatics at this point: {problemmatic}')
