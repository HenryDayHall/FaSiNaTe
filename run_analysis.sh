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
# Need to pick an input file in the data_dir, this also determines the output file (rec->ana)
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
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Paths come from the configuration (configs/default.yaml, overridden by
# configs/local.yaml; see the README). Run this from the repository root;
# sbatch starts jobs in the directory you submit from.
source "${FASINATE_REPO_DIR:-$PWD}/scripts/load_config.sh" || { return 1 2>/dev/null || exit 1; }
# Folder with the ddml_diphoton_analysis repository in it
repo_dir=$FASINATE_REPO_DIR
# mounted to /user_data inside the container
data_dir=$FASINATE_PATHS_DATA_DIR
singularity_path=$FASINATE_CONTAINER_IMAGE
export SINGULARITY_TMPDIR=$FASINATE_CONTAINER_TMP_DIR
export SINGULARITY_CACHEDIR=$FASINATE_CONTAINER_CACHE_DIR
# setup.sh in the container needs DDML built for this configuration
"$repo_dir/scripts/build_ddml.sh" --if-needed || { return 1 2>/dev/null || exit 1; }
"$repo_dir/scripts/fetch_detector_configs.sh" --if-needed || { return 1 2>/dev/null || exit 1; }
"$repo_dir/scripts/build_patches.sh" --if-needed || { return 1 2>/dev/null || exit 1; }

singularity_path=$(realpath $singularity_path)
starting_dir=$PWD

cd $repo_dir/ddml_diphoton_analysis

# The DDML build is mounted at the same path as outside, and setup.sh in the
# container uses the configuration loaded here, which passes through to it.
singularity instance start -H $PWD --bind /cvmfs:/cvmfs,$FASINATE_DDML_BIND_PATHS,$FASINATE_DETECTOR_CONFIGS_BIND_PATHS${FASINATE_PATCHES_BIND_PATHS:+,$FASINATE_PATCHES_BIND_PATHS},$repo_dir:/user_repo,$data_dir:/user_data $singularity_path instance1

echo "-- Running Marlin ..."

singularity run instance://instance1 /user_repo/ddml_diphoton_analysis/DDML_DiPhoton_Analysis_commands.sh $input_file -n $n_events -e /user_repo/setup.sh

echo "-- Done."
# stop the instance
singularity instance stop instance1
# return to the starting directory
cd $starting_dir
