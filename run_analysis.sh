#!/bin/bash
#SBATCH --time 0-05:00:00
#SBATCH --nodes 1
#SBATCH --partition maxcpu
#SBATCH --job-name analysis
#SBATCH --output=/gpfs/dust/maxwell/user/dayhallh/physical_recos/job_logs/ana_%j.out
#SBATCH --error=/gpfs/dust/maxwell/user/dayhallh/physical_recos/job_logs/ana_%j.err
#SBATCH --mail-type=FAIL
##SBATCH --constraint='EPYC&7402' 
##SBATCH --nodelist="max-wn063"         # you can select specific nodes, if necessary
##SBATCH --nodelist="max-wn065"


# Change these to your files/directories ~~~~~~~~~~~~~~~
# Folder with ddml_diphoton_analysis repository in it
steer_dir=/home/dayhallh/training/physical_recos
# Where to find input and output, will be mounted to /user_data inside the container
data_dir=/gpfs/dust/maxwell/user/dayhallh/physical_recos
# Need to pick an input file, this also determines the output file (rec->ana)
input_file=rec_output/di_photons_g4_output_files/ild-rec-e2020pdg22maxsep30centralposz50-p1-seed1_ILD_l5_o1_v02_REC.slcio
# Need to say how many events to process
n_events=100
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

export SINGULARITY_TMPDIR=/beegfs/desy/user/akorol/container/tmp/
export SINGULARITY_CACHEDIR=/beegfs/desy/user/akorol/container/cache/

$starting_dir=$PWD

cd /home/dayhallh/training/physical_recos/ddml_diphoton_analysis

singularity instance start -H $PWD --bind /cvmfs:/cvmfs,$steer_dir:/user_steer,$data_dir:/user_data /beegfs/desy/user/akorol/singularity_images/ubuntu22_wmesa_latest.sif instance1

echo "-- Running Marlin ..."

singularity run instance://instance1 /user_steer/ddml_diphoton_analysis/DDML_DiPhoton_Analysis_commands.sh $input_file -n $n_events

echo "-- Done."
# stop the instance
singularity instance stop instance1
# return to the starting directory
cd $starting_dir
