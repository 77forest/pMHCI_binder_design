#!/usr/bin/env python3
"""
All-by-all PDB alignment script for protein design analysis.
Aligns structures based on chain B, then computes RMSD for chain A.

For determining which scaffolds are most similar and could be dropped from initial input if desired.

usage:
pyhon all_by_all_align_onB_rmsd_onA.py /path/to/pdbs/ -o rmsd.csv
"""

import os
import numpy as np
from Bio import PDB
from Bio.PDB import Superimposer
import csv
from pathlib import Path
import argparse


def get_ca_atoms(chain):
    """Extract C-alpha atoms from a chain."""
    ca_atoms = []
    for residue in chain:
        if residue.has_id('CA'):
            ca_atoms.append(residue['CA'])
    return ca_atoms


def align_and_compute_rmsd(pdb1_path, pdb2_path):
    """
    Align two PDB structures based on chain B, compute RMSD for chain A.
    
    Returns:
        rmsd: RMSD value for chain A after chain B alignment
        None if alignment fails
    """
    parser = PDB.PDBParser(QUIET=True)
    
    try:
        # Load structures
        structure1 = parser.get_structure('struct1', pdb1_path)
        structure2 = parser.get_structure('struct2', pdb2_path)
        
        # Get chains
        chain_b1 = structure1[0]['B']
        chain_b2 = structure2[0]['B']
        chain_a1 = structure1[0]['A']
        chain_a2 = structure2[0]['A']
        
        # Get C-alpha atoms
        ca_b1 = get_ca_atoms(chain_b1)
        ca_b2 = get_ca_atoms(chain_b2)
        ca_a1 = get_ca_atoms(chain_a1)
        ca_a2 = get_ca_atoms(chain_a2)
        
        # Check if chains have same length
        if len(ca_b1) != len(ca_b2):
            print(f"Warning: Chain B length mismatch ({len(ca_b1)} vs {len(ca_b2)})")
            # Use minimum length
            min_len = min(len(ca_b1), len(ca_b2))
            ca_b1 = ca_b1[:min_len]
            ca_b2 = ca_b2[:min_len]
        
        if len(ca_a1) != len(ca_a2):
            print(f"Warning: Chain A length mismatch ({len(ca_a1)} vs {len(ca_a2)})")
            min_len = min(len(ca_a1), len(ca_a2))
            ca_a1 = ca_a1[:min_len]
            ca_a2 = ca_a2[:min_len]
        
        # Perform superimposition based on chain B
        super_imposer = Superimposer()
        super_imposer.set_atoms(ca_b1, ca_b2)
        
        # Apply the transformation to all atoms in structure2
        super_imposer.apply(structure2[0].get_atoms())
        
        # Now compute RMSD for chain A
        # After transformation, manually calculate RMSD
        coords_a1 = np.array([atom.coord for atom in ca_a1])
        coords_a2 = np.array([atom.coord for atom in ca_a2])
        
        diff = coords_a1 - coords_a2
        rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
        
        return rmsd
        
    except Exception as e:
        print(f"Error processing {pdb1_path} vs {pdb2_path}: {str(e)}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Perform all-by-all PDB alignment based on chain B, compute RMSD for chain A'
    )
    parser.add_argument('pdb_directory', type=str, 
                        help='Directory containing PDB files')
    parser.add_argument('-o', '--output', type=str, default='alignment_results.csv',
                        help='Output CSV file (default: alignment_results.csv)')
    
    args = parser.parse_args()
    
    # Get all PDB files
    pdb_dir = Path(args.pdb_directory)
    pdb_files = sorted(list(pdb_dir.glob('*.pdb')))
    
    if len(pdb_files) == 0:
        print(f"No PDB files found in {pdb_dir}")
        return
    
    print(f"Found {len(pdb_files)} PDB files")
    print(f"Will perform {len(pdb_files) * (len(pdb_files) - 1) // 2} pairwise comparisons")
    
    # Prepare output
    results = []
    
    # All-by-all comparison
    total_comparisons = len(pdb_files) * (len(pdb_files) - 1) // 2
    current = 0
    
    for i, pdb1 in enumerate(pdb_files):
        for j, pdb2 in enumerate(pdb_files):
            if i < j:  # Only compute upper triangle (avoid duplicates)
                current += 1
                if current % 10 == 0:
                    print(f"Progress: {current}/{total_comparisons}")
                
                rmsd = align_and_compute_rmsd(pdb1, pdb2)
                
                if rmsd is not None:
                    results.append({
                        'pdb1': pdb1.name,
                        'pdb2': pdb2.name,
                        'rmsd': rmsd
                    })
    
    # Write results to CSV
    with open(args.output, 'w', newline='') as csvfile:
        fieldnames = ['pdb1', 'pdb2', 'rmsd']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    print(f"\nDone! Results written to {args.output}")
    print(f"Successfully computed {len(results)} alignments")


if __name__ == '__main__':
    main()
