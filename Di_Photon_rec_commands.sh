#!/bin/bash

## commands to be run in singularity container

source /user_repo/setup_env_stable.sh;

### check build et al. are up to date and additional source
#cd build;
#cmake ..;
#make -j$(nproc) install;
#source install/bin/thisDDML.sh;
#cd ./scripts/;

# first argument is the input file
if [ -n "$1" ]; then
    input_file=$1
else
    echo "No input file specified."
    echo "Usage: $0 <input_file> [detector_model]"
    exit 1
fi
if [ ! -f $input_file ]; then
    # maybe it's in the mounted directory
    input_file=/user_data/$input_file
    if [ ! -f $input_file ]; then
        echo "Input file not found at $1 or $input_file"
        exit 1
    else
        echo "Found input file at $input_file"
        # get the abs path
        input_file=$(readlink -f $input_file)
    fi
fi
echo "Input file: $input_file"

# if there is a second argument, it's the detector model
if [ -n "$2" ]; then
    detector_model=$2
else
    detector_model=ILD_l5_o1_v02
fi
echo "Detector model: $detector_model"

# the output file is the input file with "sim" replaced by "rec" and the detector model added
output_base=${input_file//sim/rec}
output_base=${output_base%.slcio}_${detector_model}
echo "Output base: $output_base"

output_dir=$(dirname $output_base)
if [ ! -d $output_dir ]; then
    mkdir -p $output_dir
fi
#### Reconstruction for Di-Photon samples E2020

prod_dir=/user_repo/ILDConfig/StandardConfig/production/
with_gaps=./MarlinStdReco_WITH_GAPS.xml

cp $with_gaps $prod_dir
cd $prod_dir

echo Marlin $with_gaps --constant.DetectorModel=$detector_model --constant.OutputBaseName=$output_base --global.LCIOInputFiles=$input_file --constant.RunBeamCalReco=false;
Marlin $with_gaps --constant.DetectorModel=$detector_model --constant.OutputBaseName=$output_base --global.LCIOInputFiles=$input_file --constant.RunBeamCalReco=false;
