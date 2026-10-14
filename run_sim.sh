#!/bin/bash
#SBATCH --time 2-00:00:00
#SBATCH --nodes 1
#SBATCH --partition maxcpu
#SBATCH --job-name sim
#SBATCH --constraint='EPYC&7402' 
#SBATCH --output=/data/dust/user/dayhallh/physical_recos/job_logs/sim_%j.out
#SBATCH --error=/data/dust/user/dayhallh/physical_recos/job_logs/sim_%j.err
#SBATCH --mail-type=FAIL
##SBATCH --nodelist="max-wn063"         # you can select specific nodes, if necessary

# Change these to your files/directories ~~~~~~~~~~~~~~~
# name of your model, must be a valid model in the steering file
model_name=cciiO10  # g4 cc bibae
# Where to find the input slcio with incident particles in the data_dir
input_file=inputs/ILD-gen-E2020pdg22maxSep30centralPosZ50-p1.slcio
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Paths come from the configuration (configs/default.yaml, overridden by
# configs/local.yaml; see the README). Run this from the repository root;
# sbatch starts jobs in the directory you submit from.
source "${FASINATE_REPO_DIR:-$PWD}/scripts/load_config.sh" || { return 1 2>/dev/null || exit 1; }
repo_dir=$FASINATE_REPO_DIR
data_dir=$FASINATE_PATHS_DATA_DIR
model_dir=$FASINATE_PATHS_MODEL_DIR
singularity_path=$FASINATE_CONTAINER_IMAGE
export SINGULARITY_TMPDIR=$FASINATE_CONTAINER_TMP_DIR
export SINGULARITY_CACHEDIR=$FASINATE_CONTAINER_CACHE_DIR

# store the current directory for later
starting_dir=$(pwd)

# The DDML build from scripts/build_ddml.sh is mounted at the same path as outside;
# Di_Photon_sim_commands.sh sources its environment file, $FASINATE_DDML_ENV_FILE.
#singularity instance start -H $PWD --bind /cvmfs:/cvmfs,$lcgeo:/RegularLcgio/,$repo_dir:/user_repo,$output_dir:/user_output $singularity_path instance1
singularity instance start -H $PWD --bind /cvmfs:/cvmfs,$FASINATE_DDML_BIND_PATHS,$repo_dir:/user_repo,$data_dir:/user_data,$model_dir:/model_dir $singularity_path instance1


echo "-- Running DDSim ..."

singularity exec instance://instance1 /user_repo/Di_Photon_sim_commands.sh $model_name $input_file

echo "-- Done."
# stop the instance
singularity instance stop instance1
# return to the starting directory
cd $starting_dir
