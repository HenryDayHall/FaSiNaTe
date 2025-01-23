#!/bin/bash
#SBATCH --time 0-05:00:00
#SBATCH --nodes 1
#SBATCH --partition maxcpu
#SBATCH --job-name analysis
#SBATCH --output=/data/dust/user/dayhallh/physical_recos/job_logs/ana_%j.out
#SBATCH --error=/data/dust/user/dayhallh/physical_recos/job_logs/ana_%j.err
#SBATCH --mail-type=FAIL
##SBATCH --constraint='EPYC&7402' 
##SBATCH --nodelist="max-wn063"         # you can select specific nodes, if necessary
##SBATCH --nodelist="max-wn065"


# Change these to your files/directories ~~~~~~~~~~~~~~~
# Folder with ddml_diphoton_analysis repository in it
repo_dir=/home/dayhallh/training/physical_recos
# Where to find input and output, will be mounted to /user_data inside the container
data_dir=/data/dust/user/dayhallh/physical_recos
# Need to pick an input file, this also determines the output file (rec->ana)
#input_file=rec_output/cc_output_files/cc_20250110_164526_ILD_l5_o1_v02_REC.slcio
#input_file=rec_output/ccO_output_files/ccO_20250111_064152_ILD_l5_o1_v02_REC.slcio
#input_file=rec_output/ccA2_output_files/ccA2_20250111_050025_ILD_l5_o1_v02_REC.slcio
#input_file=rec_output/ccA4_output_files/ccA4_20250110_180746_ILD_l5_o1_v02_REC.slcio
#input_file=rec_output/ccA10_output_files/ccA10_20250114_210118_ILD_l5_o1_v02_REC.slcio
#input_file=rec_output/ccO10_output_files/ccO10_20250121_152040_ILD_l5_o1_v02_REC.slcio
#input_file=rec_output/cciiO10_output_files/cciiO10_20250122_120206_ILD_l5_o1_v02_REC.slcio
input_file=rec_output/cciiA2_output_files/cciiA2_20250122_120327_ILD_l5_o1_v02_REC.slcio
#input_file=rec_output/cciiA10_output_files/cciiA10_20250122_120328_ILD_l5_o1_v02_REC.slcio
#input_file=rec_output/g4_output_files/g4_20250112_200139_ILD_l5_o1_v02_REC.slcio
#input_file=rec_output/g4_output_files/g4_20250113_170817_ILD_l5_o1_v02_REC.slcio
# Need to say how many events to process
n_events=10000
# Your singularity, created with create_singularity.sh if needed
singularity_path=./singularity_dir/ubuntu22_latest.sif
export SINGULARITY_TMPDIR=./singularity_dir/tmp/
export SINGULARITY_CACHEDIR=./singularity_dir/cache/
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

singularity_path=$(realpath $singularity_path)
starting_dir=$PWD

cd /home/dayhallh/training/physical_recos/ddml_diphoton_analysis

singularity instance start -H $PWD --bind /cvmfs:/cvmfs,$repo_dir:/user_repo,$data_dir:/user_data $singularity_path instance1

echo "-- Running Marlin ..."

#singularity run instance://instance1 /user_repo/ddml_diphoton_analysis/DDML_DiPhoton_Analysis_commands.sh $input_file -n $n_events -e /user_repo/setup_env_stable_analysis.sh
singularity run instance://instance1 /user_repo/ddml_diphoton_analysis/DDML_DiPhoton_Analysis_commands.sh $input_file -n $n_events -e /user_repo/setup_env_stable.sh

echo "-- Done."
# stop the instance
singularity instance stop instance1
# return to the starting directory
cd $starting_dir
