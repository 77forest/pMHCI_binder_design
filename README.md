# pMHCI_binder_design
Code repository to accompany manuscript: Design of high specificity binders for peptide-MHC-I complexes

# Update 2026
We now recommend to use AF3 for folding pMHCs.

# General pipeline for pMHC binder design
1. pMHC preparation.
1a. Identify your target
1b. If your target is a structure on the pdb, fetch it and remove any unnecessary chains and residues that will not be involved in binding. This leaves us with usually the first ~180 residues of most MHCs. If your pMHC target structure is unknown, we prefer to use AF3 to predict the structure and use up to all 5 model outputs as input.
1c. Renumber the pMHC so the first residue of the pMHC is residue 1 and the numbering is continuous with the peptide residues. Also relabel the MHC and peptide chains to be chain B.

2. Backbone generation
2a. Use RFdiffusion or similar tools to generate completely new scaffolds for your pMHC target, choosing hotspot residues as peptide residues.
2b. Alternatively, you can use a set of scaffolds as we demonstrated.
2b-i. For this use the folder of scaffolds and the script `align_chainB.py` to make a set of pdb files with the binder scaffolds docked onto your target.
2b-ii. Then use partial diffusion to optimize these scaffolds for your new target.

3. Use your favorite sequence design tool, we prefer ProteinMPNN.
   
4. Filter sequence outputs via contact to the peptide residues.
   
5. Predict sequences with your favorite structure prediction tool, we currently prefer AF3 as mentioned previously. However experimenting with other lighter-compute models can be worth it.

6. Filter, and repeat or iterate if necessary.
