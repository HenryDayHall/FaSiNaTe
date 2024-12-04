# Reconstruction of CaloClouds and friends

This repository includes tools and guidance to run a full simulation and reconstruction with a model
that has been created in a similar manner to the [CaloClouds](https://gitlab.desy.de/ftx-sft/generative/point-cloud-diffusion).
It's an agglomeration of work done by Peter McKeown and Anatolii Korol.
There are roughly 4 steps;

1. Compile to model into c++ *TODO*
2. Simulate events, using the model you compiled, see `run_sim.sh`.
3. Reconstruct the simulated events, see `run_rec.sh`.
4. Perform analysis on the events. Only a photon separation analysis is currently described. *TODO*
To use the repository, you must clone recursively, because you need the submodules.

```bash
git clone --recurse-submodules TODO
```

Or if you have already cloned the repository, from inside the repository do

```bash
git submodule update --init --recursive
```

Below, you can read a bit more about how to run each of these steps and what they are expected to produce.
The folder this repository has been cloned into is referred to as `steer_dir`.

## Compiling a model

*TODO*


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

In short;
ensure you start with a clean shell, in which no python environment has ever been activated.
It's a bit inclined to get confused.
Then the commands are;

```bash
source setup_env_stable.sh
cd ddfastshowerml
mkdir build
cd build
cmake ..
make
```

Assuming this was build correctly the last stage of setting up is to modify the paths in `run_sim.sh`.
Looking at the top of the file, you can see some example instructions for SLURM jobs, which you may want to modify,
then further down you must modify the paths/variables in this block;

```bash
# Change these to your files/directories ~~~~~~~~~~~~~~~
# With a Di_Photon_sim_commands.sh and your steering file;
steer_dir=/home/dayhallh/training/physical_recos
# name of your model, must be a valid model in the steering file
model_name=cc  # g4 cc bibae
# Where you want the output to go
output_dir=/data/dust/user/dayhallh/physical_recos/sim_output
# Where your models are
model_dir=/beegfs/desy/user/akorol/projects/PC_version/ddfastshowerml/models
# Your singularity, created with create_singularity.sh if needed
singularity_path=/data/dust/user/dayhallh/physical_recos/singularity/ubuntu22_latest.sif
export SINGULARITY_TMPDIR=/data/dust/user/dayhallh/physical_recos/singularity/tmp/
export SINGULARITY_CACHEDIR=/data/dust/user/dayhallh/physical_recos/singularity/cache/
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
```

Now you can source this file to run the simulation. Or submit it to slurm with `sbatch run_sim.sh`.

