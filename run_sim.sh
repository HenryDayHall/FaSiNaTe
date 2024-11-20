#!/bin/bash
#SBATCH --time 2-24:05:00
#SBATCH --nodes 1
#SBATCH --partition maxcpu
#SBATCH --job-name sim
#SBATCH --constraint='EPYC&7402' 
##SBATCH --nodelist="max-wn063"         # you can select specific nodes, if necessary
##SBATCH --nodelist="max-wn065"

# Change these to your files/directories ~~~~~~~~~~~~~~~
# With a Di_Photon_sim_commands.sh and your steering file;
steer_dir=/home/dayhallh/training/physical_recos
# name of your model, must be a valid model in the steering file
model_name=cc
# Where you want the output to go
output_dir=/gpfs/dust/maxwell/user/dayhallh/physical_recos/sim_output
# Where your models are
model_dir=/beegfs/desy/user/akorol/projects/PC_version/ddfastshowerml/models
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

export SINGULARITY_TMPDIR=/beegfs/desy/user/akorol/container/tmp/
export SINGULARITY_CACHEDIR=/beegfs/desy/user/akorol/container/cache/
# store the current directory for later
starting_dir=$(pwd)

cd /beegfs/desy/user/akorol/projects/PC_version/ddfastshowerml
singularity instance start -H $PWD --bind /cvmfs:/cvmfs,/beegfs/desy/user/akorol/projects/2_sim_calo/Regular_detector/ILDConfig/StandardConfig/production/lcgeo/v00-16-06:/RegularLcgio/,$steer_dir:/user_steer,$output_dir:/user_output /beegfs/desy/user/akorol/singularity_images/ubuntu22_wmesa_latest.sif instance1


echo "-- Running DDSim ..."
echo "Steering directory: $steer_dir"

# singularity run instance://instance1 /user_steer/CaloClouds_Di_Photon_sim_commands.sh
singularity exec instance://instance1 /user_steer/Di_Photon_sim_commands.sh $model_dir $model_name

echo "-- Done."
# stop the instance
singularity instance stop instance1
# return to the starting directory
cd $starting_dir
