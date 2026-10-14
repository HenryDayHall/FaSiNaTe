# Reconstruction of CaloClouds and friends

This repository includes tools and guidance to run a full simulation and reconstruction with a model
that has been created in a similar manner to the [CaloClouds](https://gitlab.desy.de/ftx-sft/generative/point-cloud-diffusion).
It's an agglomeration of work done by Peter McKeown and Anatolii Korol.
There are roughly 5 steps;

1. Compile to model into c++, see `inference.py` (but you will need to implement changes).
2. Create a sample of inputs to start events from, see `Di_Photon_Input_slcio.py`.
3. Simulate events, using the model you compiled, see `run_sim.sh`.
4. Reconstruct the simulated events, see `run_reco.sh`.
5. Perform analysis on the events. Only a photon separation analysis is currently described, see `run_analysis.sh`.
6. Plots from multiple analysis can be compared, an example is given in `Di_Photon_Analysis_Plots.py`.

Steps 3, 4 and 5 can be run in sequence using `run_all.sh`, which chains the commands from `run_sim.sh`, `run_reco.sh` and `run_analysis.sh`. It's normally best to check each stage separately in your environment before running all of them together. It requires the same sort of modifications as steps 3, 4 and 5.

To use the repository, you must clone recursively, because you need the submodules.

```bash
git clone --recurse-submodules git@gitlab.desy.de:ftx-sft/generative/analysis_chain.git
```

Or if you have already cloned the repository, from inside the repository do

```bash
git submodule update --init --recursive
```

Almost all of this process in built and run inside a singularity container.
This isn't strictly necessary - at least on DESY's maxwell cluster, everything can be done directly in the shell. 
However, since the `key4hep` setup in `setup_env_stable.sh` can only be run once, and can create unpredictable issues if run multiple times, using the singularity ensures a clean environment no matter how you run the scripts.
Generally the `run_`...`.sh` scripts are designed to be submitted as slurm jobs, in which case you would have a clean environment anyway, but they can also be directly sourced.

Below, you can read a bit more about how to run each of these steps and what they are expected to produce.
The folder this repository has been cloned into is referred to as `repo_dir`.

## Compiling a model

The concept is to make a trace of the model using `poutine.trace`, which can then be run in the C++ simulation.
Tracing a model has a few restrictions;

- All the code for the model must be isolated in a single class.
- The model must use only native python and torch functions.

You can see an example of this in `inference.py`, which also produces a number of diagnostic plots.
It takes no input arguments, all updates are made by modifying the scrip itself.
If your model was produced from the same classes as the CaloClouds-2/3 models, you should be able to use this with minimal changes, otherwise, you will need to write your own inference class.


## Making the input sample

Here I'll describe one specific example that Peter wrote for the di-photon separation analysis that
he and Anatolii were doing.
You will likely want to change this step, to fit your analysis needs.

A complete example of this can be found in `Di_Photon_Input_slcio.py`.
Start by making a director to store the input file, then call like;

```bash
source setup_env_stable.sh
mkdir /data/dust/user/dayhallh/physical_recos/inputs
python3 Di_Photon_Input_slcio.py --pdg 22 --maxSep 30 --centralPosZ 50 --Emax 20 --Emin 20 --outdir /data/dust/user/dayhallh/physical_recos/inputs
```

This should yield a `gen-E2020pdg22maxSep30centralPosZ50-p1.csv` (the log) and, more interestingly, a `ILD-gen-E2020pdg22maxSep30centralPosZ50-p1.slcio`.
That is the file that is used to specify the position and direction of the incident particles in the simulation.


## Simulating events

For compatibility reasons, we need an ubuntu 22 image.
Let's label the folder for the image as `singularity_dir`.
A known working image can be created with;

```bash
# starting in a file system with plenty of space
mkdir singularity_dir
cd singularity_dir
singularity pull docker://ghcr.io/key4hep/key4hep-images/ubuntu22:latest
mkdir cache
mkdir tmp
```

You should now have a file called `ubuntu22_latest.sif`.

In addition you need to compile `ddfastshowerml`.
This is a submodule of this repository, so you should already have a copy.
There are instructions for compiling in the `ddfastshowerml/README.md`,
but it's important that the version of key4hep you use to build with is the same one you run with.
Also, since we are running in a container, you also need to build in a container
(it sometimes works without, but it's not very reliable).

In short;
ensure you start with a clean shell, in which no python environment has ever been activated.
It's a bit inclined to get confused.
Then the commands are;

```bash
singularity instance start -H $PWD --bind /cvmfs:/cvmfs,$repo_dir:/user_repo $singularity_path instance1
singularity instance shell instance://instance1
```

This will change your prompt to `Singularity>`, or something like that.

```bash
source setup_env_stable.sh
cd ddfastshowerml
mkdir build
cd build
cmake ..
make -j4 install
exit
```

Then stop the singularity instance with 

```bash
singularity instance stop instance1
```

You should only need to do this build once.

Assuming this build went smoothly, the last stage of setting up is to modify the paths in `run_sim.sh`.
Looking at the top of the file, you can see some example instructions for SLURM jobs, which you may want to modify,
then further down you must modify the paths/variables in this block;

```bash
# Change these to your files/directories ~~~~~~~~~~~~~~~
< Stuff to modify >
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
```

Now you can source this file to run the simulation. Or submit it to slurm with `sbatch run_sim.sh`.
It will produce output under `sim_output` in the data directory you specified in `run_sim.sh`.


## Reconstructing events

The simulated events need to be parsed by standard reconstructions algorithms.
This can be controlled from the `run_rec.sh` script, and has fewer dependances than simulation.

As with `run_sim.sh`, open `run_reco.sh` and fix the variables in the upper block to match your
environment.
Source this file to run the reconstruction, or submit it to slurm with `sbatch run_rec.sh`.

It will produce output under `rec_output` in the data directory you specified in `run_rec.sh`.

## Analysing the events

The output of the reconstruction is a `.csv` file, and should be fairly intuitive.
For the default photon separation reconstruction presented here, a plotting script is included.
The script is called `Di_Photon_Analysis_Plots.py`.

It's use is fairly rudimentary, and you will have to edit the paths used to find the `.csv` files produced by the output to match your environment.
This plotting script is not flexible and is indented more as inspiration, than for actual use.
