#!/usr/bin/env python3
"""
Identify near-duplicate PDB structures and select ones to remove.

Usage:
    python remove_scafs.py results.csv -t your_threshold_number -o pdbs_to_remove.txt
    cat pdbs_to_remove.txt | xargs rm
"""

import pandas as pd
import argparse


def find_duplicates_to_remove(csv_file, rmsd_threshold=0.1):
    """
    Find PDB pairs with RMSD below threshold and identify which ones to remove.
    
    Returns:
        to_remove: set of PDB files to remove
        duplicate_pairs: DataFrame of all duplicate pairs
    """
    # Load results
    df = pd.read_csv(csv_file)
    
    # Filter for near-duplicates
    duplicates = df[df['rmsd'] < rmsd_threshold].copy()
    
    print(f"Found {len(duplicates)} pairs with RMSD < {rmsd_threshold}")
    
    if len(duplicates) == 0:
        return set(), duplicates
    
    # Build a graph of duplicates
    # We'll use a greedy approach: iteratively remove the PDB that appears most frequently
    to_remove = set()
    remaining_pairs = duplicates.copy()
    
    while len(remaining_pairs) > 0:
        # Count how many times each PDB appears in remaining pairs
        pdb1_counts = remaining_pairs['pdb1'].value_counts()
        pdb2_counts = remaining_pairs['pdb2'].value_counts()
        all_counts = pd.concat([pdb1_counts, pdb2_counts]).groupby(level=0).sum()
        
        # Remove the PDB that appears most frequently
        most_common = all_counts.idxmax()
        to_remove.add(most_common)
        
        # Remove all pairs involving this PDB
        remaining_pairs = remaining_pairs[
            (remaining_pairs['pdb1'] != most_common) & 
            (remaining_pairs['pdb2'] != most_common)
        ]
    
    return to_remove, duplicates


def main():
    parser = argparse.ArgumentParser(
        description='Identify near-duplicate PDBs and select ones to remove'
    )
    parser.add_argument('csv_file', type=str, 
                        help='CSV file from alignment script')
    parser.add_argument('-t', '--threshold', type=float, default=0.1,
                        help='RMSD threshold for duplicates (default: 0.1)')
    parser.add_argument('-o', '--output', type=str, default='pdbs_to_remove.txt',
                        help='Output file listing PDBs to remove')
    
    args = parser.parse_args()
    
    # Find duplicates
    to_remove, duplicate_pairs = find_duplicates_to_remove(args.csv_file, args.threshold)
    
    # Display results
    print(f"\n{'='*60}")
    print(f"Near-duplicate pairs (RMSD < {args.threshold}):")
    print(f"{'='*60}")
    if len(duplicate_pairs) > 0:
        print(duplicate_pairs.to_string(index=False))
    else:
        print("None found!")
    
    print(f"\n{'='*60}")
    print(f"PDBs to remove ({len(to_remove)} total):")
    print(f"{'='*60}")
    for pdb in sorted(to_remove):
        print(pdb)
    
    # Save to file
    with open(args.output, 'w') as f:
        for pdb in sorted(to_remove):
            f.write(f"{pdb}\n")
    
    print(f"\n✓ List saved to {args.output}")
    
    # Summary stats
    df = pd.read_csv(args.csv_file)
    total_pdbs = len(set(df['pdb1']) | set(df['pdb2']))
    print(f"\nSummary:")
    print(f"  Total unique PDBs: {total_pdbs}")
    print(f"  PDBs to remove: {len(to_remove)}")
    print(f"  PDBs to keep: {total_pdbs - len(to_remove)}")


if __name__ == '__main__':
    main()
