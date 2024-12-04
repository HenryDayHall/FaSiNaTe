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


