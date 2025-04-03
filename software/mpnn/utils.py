


import os
import numpy as np

from Bio.PDB import PDBParser

# the lines below hide neverending warnings from biopython
import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio import BiopythonDeprecationWarning
warnings.simplefilter('ignore', PDBConstructionWarning)
warnings.simplefilter('ignore', BiopythonDeprecationWarning)
warnings.simplefilter('ignore', RuntimeWarning)


def get_offset_between_proteinmpnn_residue_index_and_PDB_residue_index(pdbfile: str):

    parser = PDBParser()
    structure = parser.get_structure(os.path.basename(pdbfile).strip('.pdb'), pdbfile)

    chain_to_start_offset_dict = {}
    for chain_obj in structure.get_chains():
        chain = chain_obj.get_id()
        for residue_obj in chain_obj:
            offset = residue_obj.get_id()[1] - 1
            chain_to_start_offset_dict[chain] = - offset
            break
    
    chain_to_offset_dict = {}
    for chain_obj in structure.get_chains():
        chain = chain_obj.get_id()
        chain_to_offset_dict[chain] = {}
        num_nonempty_icodes_so_far = 0
        for residue_obj in chain_obj:
            resnum, icode = residue_obj.get_id()[1], residue_obj.get_id()[2]
            if icode != ' ':
                num_nonempty_icodes_so_far += 1
            else:
                chain_to_offset_dict[chain][resnum] = chain_to_start_offset_dict[chain] + num_nonempty_icodes_so_far
            
    return chain_to_offset_dict


if __name__ == '__main__':
    pdbfile = '/gscratch/spe/gvisan01/tcr_pmhc/pdbs/ATLAS/3HG1.pdb'
    chain_to_offset_dict = get_offset_between_proteinmpnn_residue_index_and_PDB_residue_index(pdbfile)
    for chain in chain_to_offset_dict:
        print(chain)
        print(chain_to_offset_dict[chain])
        print()



