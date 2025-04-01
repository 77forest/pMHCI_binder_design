#README: this script is used to generate jobs for calculating contact patch scores between binders and each residue of the peptide on a MHC molecule
#Notice: this script uses Rosetta; academics can obtain a license for free. Once installed, change the path below to your local installation of "rosetta_scritps"
#Usage:
#mkdir contact; cd contact; mkdir pep; silentsplitshuf ../name_of_your_silent_file.silent 1000; cd ../..; python contact_residue_jobs.py

import os
import subprocess

# Define the directory containing the files
directory = './contact/'  # Change this to your specific directory
subdirectory = './contact/pep/'
above = '../../../'

# Iterate over the files in the directory
for filename in os.listdir(directory):
    if filename.startswith("x"):
        # Create a subdirectory for each file
        sub_dir = os.path.join(subdirectory, filename)
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)

        # Iterate over the range to create subdirectories and commands
        for i in range(181, 190): #change to range of pdb residues (+1) corresponding to the peptide when MHC + peptide are a single continuous chain in the pdb file
            # Create a subdirectory for each value in the range
            contact_dir = os.path.join(sub_dir, f'contact_{i}')
            if not os.path.exists(contact_dir):
                os.makedirs(contact_dir)

            # Define the base command with the current value
            #HERE: change path to local installation of rosetta_scripts
            base_command = f'../../../../../software/contact/rosetta_scripts -parser:protocol ../../../../../software/contact/just_contact_patch.xml -parser:script_vars patchdock_res={i} -beta_nov16 -overwrite -out:file:scorefile score.sc -out:file:silent_struct_type binary -out:file:silent /dev/null'

            # Define the full command for the current file and range value
            command = f"{base_command} -in:file:silent {os.path.join(above, filename)}"

            # Write the command to a script file in the contact subdirectory
            script_path = os.path.join(contact_dir, 'run_command.sh')
            with open(script_path, 'w') as f:
                f.write("#!/bin/bash\n")
                f.write(command + "\n")

            # Make the script executable
            os.chmod(script_path, 0o755)

            # Change to the contact subdirectory, submit the job, and then change back
            original_dir = os.getcwd()
            os.chdir(contact_dir)
            subprocess.run(['sbatch', 'run_command.sh'])
            os.chdir(original_dir)

