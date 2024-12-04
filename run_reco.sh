#!/bin/bash
#SBATCH --time 2-24:05:00
#SBATCH --nodes 1
#SBATCH --partition maxcpu
#SBATCH --job-name reco
#SBATCH --constraint='EPYC&7402' 
#SBATCH --output=/gpfs/dust/maxwell/user/dayhallh/physical_recos/job_logs/rec_%j.out
#SBATCH --error=/gpfs/dust/maxwell/user/dayhallh/physical_recos/job_logs/rec_%j.err
#SBATCH --mail-type=FAIL
##SBATCH --nodelist="max-wn063"         # you can select specific nodes, if necessary


# Change these to your files/directories ~~~~~~~~~~~~~~~
# With a Di_Photon_rec_commands.sh and a ddfastshowerml repository and setup_stable_env.sh in the same directory
repo_dir=/home/dayhallh/training/physical_recos
# Where to find input and output, will be mounted to /user_data inside the container
data_dir=/data/dust/user/dayhallh/physical_recos
# Need to pick an input file, this also determines the output file (sim->rec)
input_file=sim_output/di_photons_g4_output_files/ild-sim-e2020pdg22maxsep30centralposz50-p1-seed1.slcio
# if desired, pick a detector model, default is ILD_l5_o1_v02
# detector_model=ILD_l5_o1_v02
# Your singularity, created with create_singularity.sh if needed
singularity_path=./singularity_dir/ubuntu22_latest.sif
export SINGULARITY_TMPDIR=./singularity_dir/tmp/
export SINGULARITY_CACHEDIR=./singularity_dir/cache/
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
