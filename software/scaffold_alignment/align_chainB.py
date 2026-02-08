"""Based off of a script from Isaac Sappington, refined by Nathan F Greenwood. Takes an input dir (of pdb files with a binder and a chain b that you want to swap), out dir, ref chain (chain B) and a reference pdb (the chain that you will replace the original chain B.

improved efficiency from a previous script due to better loading of pdbs.

usage:
python align_chainB.py --input-directory $dir --ref-pdb my_target_pMHC.pdb --ref-chain B --output-directory $dir
"""

import os
import pymol
from pymol import cmd
import sys
import argparse

def align_and_save(pdb_path, ref_object, ref_chain, out_dir):
    cmd.load(pdb_path, 'complex')
    cmd.align('complex and chain B', ref_object)
    cmd.create('complex_aligned', 'complex and chain A, ' + ref_object)

    aligned_pdb_path = os.path.join(out_dir, os.path.basename(pdb_path)[:-4] + '_aligned.pdb')
    cmd.save(aligned_pdb_path, 'complex_aligned')
    cmd.delete('complex')
    cmd.delete('complex_aligned')

def main(args):
    ref_object = 'receptor'
    ref_pdb = args.ref_pdb
    ref_chain = args.ref_chain
    output_directory = args.output_directory

    cmd.load(ref_pdb, 'receptor_pdb')
    cmd.create(ref_object, f'receptor_pdb and chain {ref_chain}')
    cmd.alter(f'{ref_object} and chain {ref_chain}', 'segi="B"; chain="B"')

    os.makedirs(output_directory, exist_ok=True)

    for filename in os.listdir(args.input_directory):
        if filename.endswith('.pdb'):
            pdb_path = os.path.join(args.input_directory, filename)
            align_and_save(pdb_path, ref_object, ref_chain, output_directory)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Align PDB structures to a reference and save aligned PDBs.")
    parser.add_argument("--input-directory", required=True, help="Path to input directory containing PDB files.")
    parser.add_argument("--ref-pdb", required=True, help="Path to the reference PDB structure.")
    parser.add_argument("--ref-chain", required=True, help="Chain identifier of the reference structure.")
    parser.add_argument("--output-directory", required=True, help="Path to output directory for aligned PDBs.")
    args = parser.parse_args()
    main(args)

