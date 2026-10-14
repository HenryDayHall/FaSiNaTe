#!/bin/bash

## commands to be run in singularity container
source /user_repo/setup.sh || exit 1;
cd "$DDML_SOURCE_DIR/scripts/" || exit 1;

export OMP_NUM_THREADS=1;

# check we have enough arguments
if [ "$#" -ne 3 ]; then
    echo "Illegal number of parameters";
    echo "Usage: ./Di_Photon_sim_commands.sh <model name> <input file> <current_datetime>";
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
# make it abs path
input_file=$(realpath $input_file);
echo "Starting the simulation from the input file: $input_file";

output_file_base=$(basename $input_file);
output_file_dir=/user_data/output/${output_file_base%.*}/;
mkdir -p $output_file_dir;

current_datetime=$3
output_file=${output_file_dir}${DDSIM_MODEL}_${current_datetime}_SIM.slcio;

# run the simulation
ddsim --steeringFile /user_repo/ddsim_steer_di_photon.py --compactFile $lcgeo_DIR/ILD/compact/ILD_l5_o1_v02/ILD_l5_o1_v02.xml --inputFiles $input_file --outputFile $output_file;

prod_dir=/user_repo/ILDConfig/StandardConfig/production/
with_gaps=/user_repo/MarlinStdReco_WITH_GAPS.xml

cp $with_gaps $prod_dir
cd $prod_dir

output_base_name=${output_file_dir}${DDSIM_MODEL}_${current_datetime};
echo Marlin $prod_dir/MarlinStdReco_WITH_GAPS.xml --constant.DetectorModel=ILD_l5_o1_v02 --constant.OutputBaseName=$output_base_name --global.LCIOInputFiles=$output_file --constant.RunBeamCalReco=false;
Marlin $prod_dir/MarlinStdReco_WITH_GAPS.xml --constant.DetectorModel=ILD_l5_o1_v02 --constant.OutputBaseName=$output_base_name --global.LCIOInputFiles=$output_file --constant.RunBeamCalReco=false;

cd /user_repo/ddml_diphoton_analysis

# write parameters for the analysis
output_file=/user_repo/tmp_ddml_diphoton_params_${current_datetime}
echo "${output_base_name}_REC.slcio ${output_base_name}_Analysis.csv" > $output_file
