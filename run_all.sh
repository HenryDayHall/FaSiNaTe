#!/bin/bash
#SBATCH --time 2-24:05:00
#SBATCH --nodes 1
#SBATCH --partition maxcpu
#SBATCH --job-name sim
#SBATCH --constraint='EPYC&7402' 
#SBATCH --output=/gpfs/dust/maxwell/user/dayhallh/physical_recos/job_logs/sim_%j.out
#SBATCH --error=/gpfs/dust/maxwell/user/dayhallh/physical_recos/job_logs/sim_%j.err
#SBATCH --mail-type=FAIL
##SBATCH --nodelist="max-wn063"         # you can select specific nodes, if necessary

# Change these to your files/directories ~~~~~~~~~~~~~~~
# directory of the analysis_chain repository
repo_dir=/home/dayhallh/training/physical_recos
# name of your model, must be a valid model in the steering file
model_name=cc  # g4 cc bibae
# Where your models are, some will have been dowloaded when you compiled ddfastshowerml
# or maybe you have your own you want to test.
model_dir=/home/dayhallh/training/physical_recos/ddfastshowerml/models
# Where the input is found, and where the output will be left
data_dir=/data/dust/user/dayhallh/physical_recos/
# Where to find the input slcio with incident particles in the data_dir
input_file=inputs/ILD-gen-E2020pdg22maxSep30centralPosZ50-p1.slcio
# if desired, pick a detector model, default is ILD_l5_o1_v02
# detector_model=ILD_l5_o1_v02
# Your singularity, created with create_singularity.sh if needed
singularity_path=/data/dust/user/dayhallh/physical_recos/singularity/ubuntu22_latest.sif
export SINGULARITY_TMPDIR=/data/dust/user/dayhallh/physical_recos/singularity/tmp/
export SINGULARITY_CACHEDIR=/data/dust/user/dayhallh/physical_recos/singularity/cache/
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# work out the name of the file produced by sim
reco_input_basename=$(basename $reco_input)
reco_input_basename=${reco_input_basename//gen/sim}
reco_input=sim_output/${model_name}_output_files/${reco_input_basename}

# store the current directory for later
starting_dir=$(pwd)

#cd /beegfs/desy/user/akorol/projects/PC_version/ddfastshowerml

cd ddfastshowerml

#singularity instance start -H $PWD --bind /cvmfs:/cvmfs,$lcgeo:/RegularLcgio/,$repo_dir:/user_repo,$output_dir:/user_output $singularity_path instance1
singularity instance start -H $PWD --bind /cvmfs:/cvmfs,$repo_dir:/user_repo,$data_dir:/user_data,$model_dir:/model_dir $singularity_path instance1


echo "-- Running DDSim ..."

singularity exec instance://instance1 /user_repo/Di_Photon_sim_commands.sh $model_name $input_file

echo "-- Done with simulation."
echo "-- Running Marlin ..."

singularity run instance://instance1 /user_repo/Di_Photon_rec_commands.sh $reco_input $detector_model

echo "-- Done with reconstruction."

echo "TODO - plotting"
# stop the instance
singularity instance stop instance1
# return to the starting directory
cd $starting_dir
