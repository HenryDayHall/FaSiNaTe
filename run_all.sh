#!/bin/bash
#SBATCH --time 0-24:05:00
#SBATCH --nodes 1
#SBATCH --partition maxcpu
#SBATCH --job-name sim
#SBATCH --constraint='EPYC&7402' 
##SBATCH --output=sim_%j.out
##SBATCH --error=sim_%j.err
##SBATCH --mail-type=FAIL
##SBATCH --nodelist="max-wn063"         # you can select specific nodes, if necessary

current_datetime=$(date +"%Y%m%d_%H%M%S")

# Change these to your files/directories ~~~~~~~~~~~~~~~
# name of your model, must be a valid model in the steering file
model_name=cciiA2  # g4 cc bibae
# Where to find the input slcio with incident particles in the data_dir
# input_file=ILD-endcap-gen-E2020pdg22maxSep90centralXYZ-0-740-2411.7_p1.slcio
# input_file=gen_tau_pi0_twophotons_trigger_250-SetA_2f_leptonic_eL_pR_file_61.slcio
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
# setup.sh in the container needs DDML built for this configuration
"$repo_dir/scripts/build_ddml.sh" --if-needed || { return 1 2>/dev/null || exit 1; }
"$repo_dir/scripts/fetch_detector_configs.sh" --if-needed || { return 1 2>/dev/null || exit 1; }
"$repo_dir/scripts/build_patches.sh" --if-needed || { return 1 2>/dev/null || exit 1; }

# store the current directory for later
starting_dir=$(pwd)

# The DDML build is mounted at the same path as outside, and setup.sh in the
# containers uses the configuration loaded here, which passes through to it.
#singularity instance start -H $PWD --bind /cvmfs:/cvmfs,$lcgeo:/RegularLcgio/,$repo_dir:/user_repo,$output_dir:/user_output $singularity_path instance1
singularity instance start -H $PWD --bind /cvmfs:/cvmfs,$FASINATE_DDML_BIND_PATHS,$FASINATE_DETECTOR_CONFIGS_BIND_PATHS${FASINATE_PATCHES_BIND_PATHS:+,$FASINATE_PATCHES_BIND_PATHS},$repo_dir:/user_repo,$data_dir:/user_data,$model_dir:/model_dir $singularity_path instance2


singularity exec instance://instance2 /user_repo/Di_Photon_full_chain_commands.sh $model_name $input_file $current_datetime
# stop the instance
singularity instance stop instance2


cd $starting_dir
read input_file analysis_file < tmp_ddml_diphoton_params_${current_datetime}
rm tmp_ddml_diphoton_params_${current_datetime}

cd ddml_diphoton_analysis
singularity instance start -H $PWD --bind /cvmfs:/cvmfs,$FASINATE_DDML_BIND_PATHS,$FASINATE_DETECTOR_CONFIGS_BIND_PATHS${FASINATE_PATCHES_BIND_PATHS:+,$FASINATE_PATCHES_BIND_PATHS},$repo_dir:/user_repo,$data_dir:/user_data $singularity_path instance3

singularity run instance://instance3 /user_repo/ddml_diphoton_analysis/DDML_DiPhoton_Analysis_commands.sh $input_file -n 15000 -o $analysis_file -x /user_repo/ddml_diphoton_analysis/xml/steer.xml -e /user_repo/setup.sh
# stop the instance
singularity instance stop instance3


echo "-- Done."
# return to the starting directory
cd $starting_dir
