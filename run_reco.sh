#!/bin/bash
#SBATCH --time 2-24:05:00
#SBATCH --nodes 1
#SBATCH --partition maxcpu
#SBATCH --job-name reco
#SBATCH --constraint='EPYC&7402' 
#SBATCH --output=/data/dust/user/dayhallh/physical_recos/job_logs/rec_%j.out
#SBATCH --error=/data/dust/user/dayhallh/physical_recos/job_logs/rec_%j.err
#SBATCH --mail-type=FAIL
##SBATCH --nodelist="max-wn063"         # you can select specific nodes, if necessary


# Change these to your files/directories ~~~~~~~~~~~~~~~
# directory of the analysis_chain repository
repo_dir=/home/dayhallh/training/physical_recos
# Where to find input and output, will be mounted to /user_data inside the container
data_dir=/data/dust/user/dayhallh/physical_recos
# Need to pick an input file, this also determines the output file (sim->rec)
#input_file=sim_output/cc_output_files/cc_20250110_164526.slcio
#input_file=sim_output/ccO_output_files/ccO_20250111_064152.slcio
#input_file=sim_output/ccA4_output_files/ccA4_20250110_180746.slcio
#input_file=sim_output/cc4_output_files/ccA4_20250110_180746.slcio
#input_file=sim_output/g4_output_files/g4_20250112_200139.slcio
#input_file=sim_output/g4_output_files/g4_20250113_170817.slcio
#input_file=sim_output/ccA10_output_files/ccA10_20250114_210118.slcio
#input_file=sim_output/ccO10_output_files/ccO10_20250121_152040.slcio
#input_file=sim_output/cciiO10_output_files/cciiO10_20250122_120206.slcio
input_file=sim_output/cciiA10_output_files/cciiA10_20250122_120328.slcio

# if desired, pick a detector model, default is ILD_l5_o1_v02
# detector_model=ILD_l5_o1_v02
# Your singularity, created with create_singularity.sh if needed
singularity_path=/data/dust/user/dayhallh/physical_recos/singularity/ubuntu22_latest.sif
export SINGULARITY_TMPDIR=/data/dust/user/dayhallh/physical_recos/singularity/tmp/
export SINGULARITY_CACHEDIR=/data/dust/user/dayhallh/physical_recos/singularity/cache/
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# store the current directory for later
starting_dir=$(pwd)


singularity instance start -H $PWD --bind /cvmfs:/cvmfs,$repo_dir:/user_repo,$data_dir:/user_data $singularity_path instance1


echo "-- Running Marlin ..."

singularity run instance://instance1 /user_repo/Di_Photon_rec_commands.sh $input_file $detector_model

echo "-- Done."
# stop the instance
singularity instance stop instance1
# return to the starting directory
cd $starting_dir
