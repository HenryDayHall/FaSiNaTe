#!/bin/bash

## commands to be run in singularity container

source /user_steer/setup_env_stable2.sh;

### check build et al. are up to date and additional source
#cd build;
#cmake ..;
#make -j$(nproc) install;
source install/bin/thisDDML.sh;
cd ./scripts/;

export OMP_NUM_THREADS=1;

# check we have enough arguments
if [ "$#" -ne 2 ]; then
    echo "Illegal number of parameters";
    echo "Usage: ./Di_Photon_sim_commands.sh <model folder> <model name>";
    exit 1;
fi
echo "Looking for model in folder: $1";
export DDSIM_MODEL_FOLDER=$1
echo "Running the simulation with model: $2";
export DDSIM_MODEL=$2
# run the simulation
ddsim --steeringFile /user_steer/ddsim_steer_di_photon.py --compactFile $lcgeo_DIR/ILD/compact/ILD_l5_o1_v02/ILD_l5_o1_v02.xml;
