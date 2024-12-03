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
# With a Di_Photon_rec_commands.sh
steer_dir=/home/dayhallh/training/physical_recos
# Where to find input and output, will be mounted to /user_data inside the container
data_dir=/gpfs/dust/maxwell/user/dayhallh/physical_recos
# Need to pick an input file, this also determines the output file (sim->rec)
input_file=sim_output/di_photons_g4_output_files/ild-sim-e2020pdg22maxsep30centralposz50-p1-seed1.slcio
# if desired, pick a detector model, default is ILD_l5_o1_v02
# detector_model=ILD_l5_o1_v02
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

export SINGULARITY_TMPDIR=/beegfs/desy/user/akorol/container/tmp/
export SINGULARITY_CACHEDIR=/beegfs/desy/user/akorol/container/cache/
# store the current directory for later
starting_dir=$(pwd)

cd /beegfs/desy/user/akorol/projects/PC_version/ddfastshowerml

singularity instance start -H $PWD --bind /cvmfs:/cvmfs,/beegfs/desy/user/akorol/projects/2_sim_calo/Regular_detector/ILDConfig/StandardConfig/production/lcgeo/v00-16-06:/RegularLcgio/,$steer_dir:/user_steer,$data_dir:/user_data /beegfs/desy/user/akorol/singularity_images/ubuntu22_wmesa_latest.sif instance1


echo "-- Running Marlin ..."

singularity run instance://instance1 /user_steer/Di_Photon_rec_commands.sh $input_file $detector_model

echo "-- Done."
# stop the instance
singularity instance stop instance1
# return to the starting directory
cd $starting_dir
