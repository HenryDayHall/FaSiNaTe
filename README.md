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

[DDML](https://github.com/key4hep/DDML), which runs the model inside the simulation, is not a submodule; `scripts/build_ddml.sh` builds it (see [Simulating events](#simulating-events)).

Almost all of this process in built and run inside a singularity container.
This isn't strictly necessary - at least on DESY's maxwell cluster, everything can be done directly in the shell. 
However, since the `key4hep` setup in `setup_env_stable.sh` can only be run once, and can create unpredictable issues if run multiple times, using the singularity ensures a clean environment no matter how you run the scripts.
Generally the `run_`...`.sh` scripts are designed to be submitted as slurm jobs, in which case you would have a clean environment anyway, but they can also be directly sourced.

Below, you can read a bit more about how to run each of these steps and what they are expected to produce.
The folder this repository has been cloned into is referred to as `repo_dir`.

## Configuration

Settings shared by the scripts are in `configs/default.yaml`:
where external software is cloned and built (`paths.software_dir`),
where inputs and outputs go (`paths.data_dir`),
where the steering file finds models (`paths.model_dir`),
which DDML and Key4hep release to build, and which container image to use.

Don't edit `default.yaml` for your own setup.
Put the settings you change in `configs/local.yaml`, which git ignores and the scripts read automatically, e.g.

```yaml
paths:
  software_dir: /data/dust/user/$USER/fasinate/external
  data_dir: /data/dust/user/$USER/fasinate/data
```

Settings can refer to each other (`${paths.software_dir}`) and to environment variables (`$USER`), and relative paths are relative to the repository.
To layer another file on top, e.g. for a site or a CI job, list it in `FASINATE_CONFIG`; `scripts/build_ddml.sh` also takes `--config FILE` and `--set section.key=value`.
Check the result with

```bash
python3 scripts/fasinate_config.py                      # all settings
python3 scripts/fasinate_config.py --get paths.data_dir  # one setting
```

This needs Python 3 with PyYAML (`python3 -m pip install --user pyyaml` if it's missing).
The `run_*.sh` scripts load the configuration through `scripts/load_config.sh`, so run them from the repository root (or set `FASINATE_REPO_DIR`).

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

The simulation runs the model inside `ddsim` through [DDML](https://github.com/key4hep/DDML).
Build DDML's `python-inference` branch once with

```bash
scripts/build_ddml.sh    # add --dry-run to see the settings and steps first
```

Following the [configuration](#configuration), this

1. clones DDML into `ddml.source_dir`, or fast-forwards an existing clone (or checks out `ddml.commit`, if set);
2. pulls the container image to `container.image` if it isn't there yet;
3. builds and installs DDML inside that container, in a clean shell, against the Key4hep release `key4hep.release`;
4. writes `ddml.env_file`, which sets up that same Key4hep release and then DDML, and checks that it works.

The Key4hep release you build with must be the one you run with.
The environment file records the release DDML was built against (also when `key4hep.release` is `latest`), and the simulation scripts source it inside the container.
At the end, the script prints the DDML commit and Key4hep release it used, ready to paste into `configs/local.yaml` to pin them.
Re-run it after changing either; `--clean` starts from an empty build directory.

Recent Key4hep releases are built for AlmaLinux 9 and Ubuntu 24.04 rather than Ubuntu 22.04, so the default image is `ghcr.io/key4hep/key4hep-images/alma9`.

The build downloads DDML's reference models into `models/` in the DDML clone, which is the default `paths.model_dir`, mounted as `/model_dir` for the steering file.
Put your own traced models there too, or point `paths.model_dir` at the folder that has them.

If your clone still has the old `ddfastshowerml` submodule, build DDML first, then keep the models you added to it (they aren't tracked anywhere) before removing it:

```bash
cp -n ddfastshowerml/models/*.pt "$(python3 scripts/fasinate_config.py --get paths.model_dir)"/
rm -rf ddfastshowerml .git/modules/ddfastshowerml
git config --remove-section submodule.ddfastshowerml
```

The last stage of setting up is to choose what to simulate in `run_sim.sh`.
Looking at the top of the file, you can see some example instructions for SLURM jobs, which you may want to modify,
then further down you set the model and input file in this block;

```bash
# Change these to your files/directories ~~~~~~~~~~~~~~~
< Stuff to modify >
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
```

Now you can source this file, from the repository root, to run the simulation. Or submit it to slurm with `sbatch run_sim.sh`.
It will produce output under `sim_output` in `paths.data_dir`.


## Reconstructing events

The simulated events need to be parsed by standard reconstructions algorithms.
This can be controlled from the `run_reco.sh` script, and has fewer dependances than simulation.

As with `run_sim.sh`, open `run_reco.sh` and choose the input file in the upper block;
the paths come from the configuration.
Source this file to run the reconstruction, or submit it to slurm with `sbatch run_reco.sh`.

It will produce output under `rec_output` in `paths.data_dir`.

## Analysing the events

The output of the reconstruction is a `.csv` file, and should be fairly intuitive.
For the default photon separation reconstruction presented here, a plotting script is included.
The script is called `Di_Photon_Analysis_Plots.py`.

It's use is fairly rudimentary, and you will have to edit the paths used to find the `.csv` files produced by the output to match your environment.
This plotting script is not flexible and is indented more as inspiration, than for actual use.
