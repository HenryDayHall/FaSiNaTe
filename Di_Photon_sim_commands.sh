#!/bin/bash

## commands to be run in singularity container
source /user_repo/setup_env_stable.sh;

source install/bin/thisDDML.sh;
cd ./scripts/;

export OMP_NUM_THREADS=1;

# check we have enough arguments
if [ "$#" -ne 3 ]; then
    echo "Illegal number of parameters";
    echo "Usage: ./Di_Photon_sim_commands.sh <model folder> <model name> <input file>";
    exit 1;
fi
echo "Looking for model in folder: $1";
export DDSIM_MODEL_FOLDER=$1
echo "Running the simulation with model: $2";
export DDSIM_MODEL=$2

input_file=$3
if [ ! -f $input_file ]; then
    inital_input_file=$input_file
    # it may be in the /user_data folder
    input_file=/user_data/$3
    if [ ! -f $input_file ]; then
        echo "Input file not found!";
        echo "Tried to look for the input file in the following locations:";
        echo "1. $inital_input_file";
        echo "2. $input_file";
        exit 1;
    fi
fi
echo "Starting the simulation from the input file: $input_file";
# run the simulation
ddsim --steeringFile /user_repo/ddsim_steer_di_photon.py --compactFile $lcgeo_DIR/ILD/compact/ILD_l5_o1_v02/ILD_l5_o1_v02.xml --inputFiles $input_file;
