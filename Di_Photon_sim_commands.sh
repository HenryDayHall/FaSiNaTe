#!/bin/bash

## commands to be run in singularity container
source /user_repo/setup_env_stable.sh;

source install/bin/thisDDML.sh;
cd ./scripts/;

export OMP_NUM_THREADS=1;

# check we have enough arguments
if [ "$#" -ne 2 ]; then
    echo "Illegal number of parameters";
    echo "Usage: ./Di_Photon_sim_commands.sh <model name> <input file>";
    exit 1;
fi
export DDSIM_MODEL_FOLDER=/model_dir
echo "Running the simulation with model: $1";
export DDSIM_MODEL=$1

input_file=$2
if [ ! -f $input_file ]; then
    inital_input_file=$input_file
    # it may be in the /user_data folder
    input_file=/user_data/$2
    if [ ! -f $input_file ]; then
        echo "Input file not found!";
        echo "Tried to look for the input file in the following locations:";
        echo "1. $inital_input_file";
        echo "2. $input_file";
        exit 1;
    fi
fi
echo "Starting the simulation from the input file: $input_file";

output_file_base=$(basename $input_file);
output_file_dir=/user_data/sim_output/$2_output_files/;
mkdir -p $output_file_dir;

current_datetime=$(date +"%Y%m%d_%H%M%S")
output_file=${output_file_dir}${DDSIM_MODEL}_${current_datetime}.slcio;

# run the simulation
ddsim --steeringFile /user_repo/ddsim_steer_di_photon.py --compactFile $lcgeo_DIR/ILD/compact/ILD_l5_o1_v02/ILD_l5_o1_v02.xml --inputFiles $input_file --outputFile $output_file;
