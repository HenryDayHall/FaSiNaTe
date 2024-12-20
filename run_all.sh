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

# Change these to your files/directories ~~~~~~~~~~~~~~~
# directory of the analysis_chain repository
repo_dir=/data/dust/user/akorol/projects/analysis_chain
# name of your model, must be a valid model in the steering file
model_name=cc  # g4 cc bibae
# Where your models are, some will have been dowloaded when you compiled ddfastshowerml
# or maybe you have your own you want to test.
model_dir=/data/dust/user/akorol/projects/analysis_chain/ddfastshowerml/models
# Where the input is found, and where the output will be left
data_dir=/data/dust/user/akorol/projects/analysis_chain
# Where to find the input slcio with incident particles in the data_dir
# input_file=ILD-endcap-gen-E2020pdg22maxSep90centralXYZ-0-740-2411.7_p1.slcio
# input_file=gen_tau_pi0_twophotons_trigger_250-SetA_2f_leptonic_eL_pR_file_61.slcio
input_file=ILD-gen-E2020pdg22maxSep90centralPosZ70-p1.slcio
# Your singularity, created with create_singularity.sh if needed
singularity_path=/data/dust/user/akorol/singularity/ubuntu22_latest.sif
export SINGULARITY_TMPDIR=/data/dust/user/akorol/singularity/tmp/
export SINGULARITY_CACHEDIR=/data/dust/user/akorol/singularity/cache/
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# store the current directory for later
starting_dir=$(pwd)

#cd /beegfs/desy/user/akorol/projects/PC_version/ddfastshowerml

cd ddfastshowerml

#singularity instance start -H $PWD --bind /cvmfs:/cvmfs,$lcgeo:/RegularLcgio/,$repo_dir:/user_repo,$output_dir:/user_output $singularity_path instance1
singularity instance start -H $PWD --bind /cvmfs:/cvmfs,$repo_dir:/user_repo,$data_dir:/user_data,$model_dir:/model_dir $singularity_path instance1


singularity exec instance://instance1 /user_repo/Di_Photon_full_chain_commands.sh $model_name $input_file
# stop the instance
singularity instance stop instance1


cd $starting_dir
read input_file analysis_file < tmp_ddml_diphoton_params
rm tmp_ddml_diphoton_params

cd ddml_diphoton_analysis
singularity instance start -H $PWD --bind /cvmfs:/cvmfs,$repo_dir:/user_repo,$data_dir:/user_data $singularity_path instance2

singularity run instance://instance2 /user_repo/ddml_diphoton_analysis/DDML_DiPhoton_Analysis_commands.sh $input_file -n 15000 -o $analysis_file -x /user_repo/ddml_diphoton_analysis/xml/steer.xml
# stop the instance
singularity instance stop instance2


echo "-- Done."
# return to the starting directory
cd $starting_dir
