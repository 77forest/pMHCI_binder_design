"""A file to submit batch jobs to the cluster.


usage:

    python ./batch_pMHC_fold.py --prefix pMHC_fold --script ./pmhc_fold.py --alleles A*02:01 A*02:01 --peptides RMFPNAPYL KLTPLCVTL --minib_dir ./input_pdbs/ --gres gpu:a4000:1 --structs_per_job 50 --p gpu --t 04:00:00

"""

#!/usr/bin/env python

# This is a script that automates the creation and submission of a batch of jobs in a Brian Coventry-style file structure
# This is super similar to createarrayjobs but just does python jobs instead
# Written by Nate Bennett (nrbennet@uw.edu), adapted by Nathan Greenwood (forest77@uw.edu)

import sys, subprocess, os
from argparse import ArgumentParser
import distutils.spawn
import glob
import re

import pandas as pd
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1

def my_rstrip(string, strip):
    if (string.endswith(strip)):
        return string[:-len(strip)]
    return string

def cmd(command, wait=True):
    the_command = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    if (not wait):
        return
    the_stuff = the_command.communicate()
    return str( the_stuff[0]) + str(the_stuff[1] )

# This does ceiling integer division and is stolen from stackoverflow, thanks stackoverflow!!
def ceildiv(a, b):
    return -(-a // b)


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

# 
# Read in Arguments
# 

if (len(sys.argv) == 1):
    print("")
    print('This script prepares your jobs to run as array jobs. It will create a Brian-style file structure for commands and runs.')
    print("Arguments:")
    print(" --prefix            : The name that you want to prefix the _commands and _runs directories")
    print(" --script            : The python script that you wish to run")

    print(" --alleles           : The alleles that you wish to submit")
    print(" --peptides          : The peptides that you wish to submit")
    print(" --query             : Optional, The query that you wish to submit")
    print(" --minib_dir         : The directory that contains the minib files")

    print(" --args              : A string of addition arguments that you wish to be used in the python script command")
    print(" --apptainer         : A path to the apptainer you would like to run this script with (default: /software/containers/mlfold.sif)")
    print(" --structs_per_job   : The number of structures that you wish to package into one job")
    print(" --cpus              : Optional, the number of cpus per job that you wish to use (default, 2)")
    print(" --mem               : Optional, the amount of memory that you wish to use per job in GB (default, 32)")
    print(" --p                 : Optional, the SLURM queue that you wish to submit your job to (default, cpu-bf)")
    print(" --t                 : Optional, the SLURM timelimit to use for your job (default, 12:00:00)")
    print(" --gres              : Optional, arguments that you would like to submit to SLURM's --gres field (default, '')")
    print(" --jobs_per_group    : Optional, the number of jobs per submission to SLURM (default, 1)")
    print(" --scratch           : Optional, whether to write the output of the runs to a mirrored directory on scratch (default, False)")
    print(" --dscratch          : Optional, whether to write the output of the runs to a mirrored directory on dscratch (default, False)")
    print(" --pretrain          : Optional, whether to use the non-fine tuned version of AF (default, False)")
    sys.exit(1)

parser = ArgumentParser()
parser.add_argument("--prefix", type=str, default="")
parser.add_argument("--script", type=str, default="")

parser.add_argument("--alleles", nargs='+', default="")
parser.add_argument("--peptides", nargs='+', default="")
parser.add_argument("--query", type=str, default="")
parser.add_argument("--minib_dir", type=str, default="")

parser.add_argument("--args", type=str, default="")
parser.add_argument("--apptainer", type=str, default="/software/containers/mlfold.sif")
parser.add_argument("--structs_per_job", type=int, default="0")
parser.add_argument("--cpus", type=int, default=2)
parser.add_argument("--mem", type=int, default=32)
parser.add_argument("--p", type=str, default="cpu-bf")
parser.add_argument("--t", type=str, default="12:00:00")
parser.add_argument("--gres", type=str, default="")
parser.add_argument("--jobs_per_group", type=int, default=1)
parser.add_argument("--scratch", action='store_true')
parser.add_argument("--dscratch", action='store_true')
parser.add_argument("--pretrain", action='store_true')

args = parser.parse_args(sys.argv[1:])

prefix = args.prefix
script_filename = args.script

alleles = args.alleles
peptides = args.peptides
query = args.query
minib_dir = args.minib_dir

args_string = args.args
structs_per_job = args.structs_per_job
cpus = args.cpus
mem = args.mem
queue = args.p
timelimit = args.t
gres = args.gres
group_size = args.jobs_per_group
scratch = args.scratch
dscratch = args.dscratch

if ( script_filename == '' ):
    sys.exit("This script needs a python script to run")
script_filename = os.path.abspath( script_filename )



if (prefix == ''):
    sys.exit("This script needs a prefix name to the _commands and _runs directories")

if not queue in ['cpu', 'cpu-bf', 'gpu', 'gpu-bf']:
    sys.exit("-p (the SLURM queue) must be in {cpu, cpu-bf, gpu, gpu-bf}")

# Input Checking for Timelimit
valid_time_format = re.compile(r'^[0-9][0-9]:[0-9][0-9]:[0-9][0-9]$')
if not valid_time_format.fullmatch(timelimit):
    # Strict time limit format matching
    sys.exit(f"-t (the SLURM time limit) must be of the format HH:MM:SS. You used {timelimit}. Exiting.")

if ( not gres == '' ):
    gres = "--gres=%s "%gres

scratchit = False
if ( scratch ):
    if ( os.getcwd().startswith( '/net/scratch' ) or os.getcwd().startswith( '/mnt/net/scratch' )):
        print("\nThe -scratch flag was supplied but I am already in /net/scratch, ignoring the flag")
    else:
        scratchit = True
        scratchdir = "/net/scratch"

total_path = os.getcwd()
if total_path.startswith( '/mnt' ):
    total_path = total_path[4:]

if( scratchit ):
    trunc_path = ''
    if ( total_path.startswith( '/mnt/home' ) ):
        trunc_path = total_path[9:]
    elif ( total_path.startswith( '/home' ) ):
        trunc_path = total_path[5:]
    else:
        sys.exit( 'Something is seriously wrong with the current working directory: %s' %total_path )

    runs_path = scratchdir + os.path.join( trunc_path, "%s_runs"%prefix)
    commands_path = scratchdir + trunc_path + "/%s_commands"%prefix

    # Just for testing
    print( 'You have chosen to write your output to scratch' )
    print( 'I am planning on writing your jobs to %s'%runs_path )

    cmd( "mkdir -p %s" %runs_path )
    cmd( "ln -s %s" %runs_path )

    cmd( "mkdir -p %s" %commands_path )
    cmd( "ln -s %s" %commands_path )

else:
    cmd( "mkdir %s_runs" %prefix )
    cmd( "mkdir %s_commands" %prefix )

    runs_path = os.path.join(total_path, "%s_runs"%prefix)
    commands_path = os.path.join(total_path, "%s_commands"%prefix)

os.chdir( commands_path )
cmd( "mkdir splits" )
os.chdir( 'splits' )

alignment_df = pd.read_csv('/net/scratch/forest77/paper/code_upload/pMHC_fold/alignments_all_alleles_vs_pdb_June29_2024.csv')


if args.query:
    seqs_to_fold = pd.read_csv(args.query, delim_whitespace=True)
    seqs_to_fold.columns = ['binder', 'description']

else:
    import warnings
    warnings.filterwarnings("ignore", message="Ignoring unrecognized record")

    data = {'binder': [], 'description': []}
    local_minib_dir = '../../' + args.minib_dir
    for file in os.listdir(local_minib_dir):
        if file.endswith('.pdb'):
            data['binder'].append((extract_protein_sequences(os.path.join(local_minib_dir, file)))['A'])
            data['description'].append(file[:-4])
    seqs_to_fold = pd.DataFrame(data)

alleles = args.alleles
peptides = args.peptides
# print(alleles)
# print(peptides)

data = {'entry_index': [], 'peptide_seq': [], 'allele_sequence': [], 'target_chainseq': [], 'targetid': [], 'allele_msa_file_names': [], 'description': [], 'pdb_name': []}

for allele, peptide in zip(alleles, peptides):
    allele_seq = alignment_df.loc[alignment_df['allele_name_query'] == allele, 'mhc_seq_query'].iloc[0]
    for i, row in seqs_to_fold.iterrows():
        data['entry_index'].append(i)
        data['peptide_seq'].append(peptide)
        data['allele_sequence'].append(allele_seq)
        data['target_chainseq'].append(f"{allele_seq}/{row['binder']}/{peptide}")
        data['targetid'].append(f"{allele}_{peptide}")
        data['allele_msa_file_names'].append(allele)
        desc = row['description'] + f"_{allele}_{peptide}"
        desc = desc.replace("*", "")
        desc = desc.replace(":", "")
        data['description'].append(desc)
        data['pdb_name'].append(row['description'])

df_sample = pd.DataFrame(data)
df_sample.to_csv('sample_df.csv', index=False)

# Ok now we need to split the sample_df into chunks
# We will split the sample_df into chunks of size structs_per_job
chunks = [df_sample[i:i+structs_per_job] for i in range(0, len(df_sample), structs_per_job)]

# Write each chunk to a file
for i, chunk in enumerate(chunks):
    chunk.to_csv(f'sample_df_chunk_{i}.csv', index=False)



commands = []
for x in glob.glob("*chunk*.csv"):
    full_x = os.path.join(commands_path,'splits', x)
    dirname = my_rstrip( x, ".csv" )
    ### Apptainer
    if args.pretrain:
        commands.append( f"cd {runs_path}; mkdir {dirname}; cd {dirname}; {args.apptainer} {script_filename} --out_dir {runs_path + '/' + dirname + '/'} --alleles {args.alleles} --peptides {args.peptides} --minib_dir {args.minib_dir} --pretrain --model_name model_2_ptm --seq_mode single_seq --df {full_x} {args_string} > log.log 2>&1" )

    else:
        commands.append( f"cd {runs_path}; mkdir {dirname}; cd {dirname}; {args.apptainer} {script_filename} --out_dir {runs_path + '/' + dirname + '/'} --alleles {args.alleles} --peptides {args.peptides} --minib_dir {args.minib_dir} --model_weights_path /net/scratch/forest77/paper/code_upload/pMHC_fold/mixed_mhc_pae_run6_af_mhc_params_20640.pkl --model_name model_2_ptm --seq_mode single_seq --df {full_x} {args_string} > log.log 2>&1" )

os.chdir( commands_path )
with open("commands.list", "w") as f:
    f.write("\n".join(commands))
    f.write("\n")

num_commands_min_1 = ceildiv( len(commands), group_size ) - 1
test_command = commands[0][ : -len( '> log.log 2>&1' ) ]

os.chdir( total_path )

assert( os.path.exists("/net/scratch/forest77/paper/code_upload/pMHC_fold/split_commands.py") )

retval = cmd( "python2 ./net/scratch/forest77/paper/code_upload/pMHC_fold/split_commands.py %s_commands/commands.list %s_commands %i %i %i %s"%(prefix, prefix, group_size, mem, cpus, prefix) )


command_string = "sbatch -p %s -t %s %s-a 0-%i %s_commands/%s_array.submit"%(queue, timelimit, gres, num_commands_min_1, prefix, prefix)

with open("run_submit.sh", 'w') as f:
        f.write( '#!/bin/bash' )
        f.write( '\n')
        f.write( command_string )
        f.write( '\n')
cmd('chmod +x run_submit.sh')

with open("test_command.sh", 'w' ) as f:
        f.write( '#!/bin/bash' )
        f.write( '\n')
        f.write( test_command )
        f.write( '\n')
cmd('chmod +x test_command.sh')


print( "" )
print("/"*100)
print( "Please please please run this command to test that your jobs will run correctly:" )
print( "" )
print( "./test_command.sh" )
print( "" )
print("/"*100)
print( "" )
print( "When you are ready and sure that everything is working this is the command that will submit your jobs:")
print( "" )
print( "./run_submit.sh" )
print( "" )

