# Reconstruction of CaloClouds and friends

This repository includes tools and guidance to run a full simulation and reconstruction with a model
that has been created in a similar manner to the [CaloClouds](https://gitlab.desy.de/ftx-sft/generative/point-cloud-diffusion).
It's an agglomeration of work done by Peter McKeown and Anatolii Korol.
There are roughly 5 steps;

1. Compile to model into c++, see `inference.py` (but you will need to implement changes).
2. Create a sample of inputs to start events from, see `Di_Photon_Input_slcio.py`.
3. Simulate events with each model, see `simulate_models.sh`, and with Geant4 as the reference, see `simulate_geant4.sh` and `slurm_simulate_geant4.sh`.
4. Reconstruct the simulated events, see `reconstruct_simulation.sh`.
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

[DDML](https://github.com/key4hep/DDML), which runs the model inside the simulation, is not a submodule; `setup.sh` builds it the first time you need it (see [Setting up the environment](#setting-up-the-environment)).

Almost all of this process in built and run inside a singularity container.
This isn't strictly necessary - at least on DESY's maxwell cluster, everything can be done directly in the shell. 
However, since the `key4hep` setup in `setup.sh` can only be run once, and can create unpredictable issues if run multiple times, using the singularity ensures a clean environment no matter how you run the scripts.
Generally the `run_`...`.sh` scripts are designed to be submitted as slurm jobs, in which case you would have a clean environment anyway, but they can also be directly sourced.

Below, you can read a bit more about how to run each of these steps and what they are expected to produce.
The folder this repository has been cloned into is referred to as `repo_dir`.

## Configuration

Settings shared by the scripts are in `configs/default.yaml`:
where external software is cloned and built (`paths.software_dir`),
where inputs and outputs go (`paths.data_dir`),
where the steering file finds models (`paths.model_dir`),
which DDML and Key4hep release to build, which container image to use,
and the benchmark: the `scenarios` to simulate and the `observables` to measure on them.

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

## Setting up the environment

Every step runs in the same environment: a Key4hep release with its ML libraries, plus a build of DDML's `python-inference` branch, which runs the model inside `ddsim`.
Get it in a clean bash shell with

```bash
source setup.sh
```

This reads the configuration, builds DDML if there's no build for this configuration yet, clones the detector configurations that reconstruction needs (ILDConfig; see [Reconstructing events](#reconstructing-events)) if they aren't there yet, builds the [patches to the Key4hep release](#fixes-the-key4hep-release-doesnt-have-yet) if they aren't built yet, then sets up the Key4hep release DDML was built against, DDML itself, and the patches.
Source it only once per shell, because the Key4hep setup can only run once.

The build is done by `scripts/build_ddml.sh`, which

1. clones DDML into `ddml.source_dir`, or fast-forwards an existing clone (or checks out `ddml.commit`, if set);
2. pulls the container image to `container.image` if it isn't there yet;
3. builds and installs DDML inside that container, in a clean shell, against the Key4hep release `key4hep.release`;
4. writes `ddml.env_file`, which `setup.sh` sources, and checks that it works.

`setup.sh` and the `run_*.sh` scripts run it with `--if-needed`, so it only builds when there's no build matching the configuration:
changing the DDML commit or branch, the build options, the Key4hep release or the container image means a rebuild.
Run `scripts/build_ddml.sh` yourself to update DDML to the head of its branch; `--dry-run` shows what it would do, and `--clean` starts from an empty build directory.
At the end, it prints the DDML commit and Key4hep release it used, ready to paste into `configs/local.yaml` to pin them.

DDML is built inside the container, so source `setup.sh` in that container, or on a host with the same OS (AlmaLinux, Rocky or RHEL 9 for the default image); elsewhere it stops and says which platform DDML was built for.
Recent Key4hep releases are built for AlmaLinux 9 and Ubuntu 24.04 rather than Ubuntu 22.04, so the default image is `ghcr.io/key4hep/key4hep-images/alma9`.
In the containers the `run_*.sh` scripts start, the commands source `/user_repo/setup.sh`, which uses the configuration passed in from outside.

### Fixes the Key4hep release doesn't have yet

Sometimes a bug that breaks the chain is already fixed upstream, but in no Key4hep release yet.
The `patches` section of the configuration lists such fixes: each is a package of the release, rebuilt from a newer version against the Key4hep release DDML is built against, which `setup.sh` uses in place of the release's own.
There is one so far:

- `marlintrkprocessors`: in MarlinTrkProcessors 2.12.8 and earlier, the TPC digitisation of ILD's reconstruction (`DDTPCDigiProcessor`) stops the whole job with `padTheta = nan` when it can't work out a hit's angle from three TPC hits, which backscatter from Geant4 showers can leave.
  It is fixed in [iLCSoft/MarlinTrkProcessors#79](https://github.com/iLCSoft/MarlinTrkProcessors/commit/f2da912fa583eb6da3e9aac4c8ba43fcde64efab), first released in `v02-12-09`, which is 2.12.8 with only that fix; so this builds `v02-12-09`.

`scripts/build_patches.sh` builds them, and `setup.sh` and the scripts that start a container run it with `--if-needed`, as they do `scripts/build_ddml.sh`.
For each patch, it finds the release's own build of the package (its library in `MARLIN_DLL`) and its version; if that is older than the patch's `fixed_in`, it clones the patch's `repository` at its `ref` into `patches.base_dir`, then builds and installs it in the same clean shell and container as DDML, with ROOT's C++ standard, as Key4hep builds its packages.
It then writes `patches.env_file`, which `setup.sh` sources after setting up Key4hep, and which puts each patch's library in `MARLIN_DLL` in place of the release's.
It rebuilds when the patches' settings or the Key4hep release change, and `--dry-run`, `--clean`, `--config` and `--set` work as for `scripts/build_ddml.sh`.
`setup.sh` says which patches it uses, e.g.

```
FaSiNaTe: patch: marlintrkprocessors v02-12-09 (3ba32e2ae54b) in place of the release's 2.12.8, until Key4hep has 2.12.9
```

and the reconstruction logs record them, next to the Key4hep release (`# patches`); the reconstruction's own log lists the library it loaded (`Loading library ...`).

**Removing a patch.**
A patch is only used while the release's own version is older than its `fixed_in`.
Once DDML is built against a release that has the fix, `setup.sh` stops using the patch by itself, without building it, and says that it can go:

```
FaSiNaTe: patch marlintrkprocessors isn't needed any more: Key4hep 2026-11-01 has marlintrkprocessors 2.12.9 (fixed_in 2.12.9); delete patches.marlintrkprocessors from configs/default.yaml
```

Then delete that entry from `configs/default.yaml`; nothing else needs to change, and its build in `patches.base_dir` can be deleted too.
To go on without a patch before that, e.g. if it doesn't build, set it off in `configs/local.yaml`:

```yaml
patches:
  marlintrkprocessors:
    enabled: false
```

A new patch needs an entry with the same keys, named after the package's Spack package, for a library of Marlin processors.

## Compiling a model

The concept is to make a trace of the model using `poutine.trace`, which can then be run in the C++ simulation.
Tracing a model has a few restrictions;

- All the code for the model must be isolated in a single class.
- The model must use only native python and torch functions.

You can see an example of this in `inference.py`, which also produces a number of diagnostic plots.
It takes no input arguments, all updates are made by modifying the scrip itself.
If your model was produced from the same classes as the CaloClouds-2/3 models, you should be able to use this with minimal changes, otherwise, you will need to write your own inference class.


## Making the input sample

Each benchmark scenario in the configuration says how its input events are made.
Make the inputs of all of them, or of the ones you name, with

```bash
./generate_input.sh                # every scenario
./generate_input.sh single_photon  # just this one
```

It runs in the same environment as the simulation, building DDML first if needed.
The single photon scenario gives one `.slcio` file per energy, each with an `.h5` file of the true energies, momenta and gun positions, in `input_single_particle_gun` under `paths.data_dir`.
To try a scenario with fewer events, override it in `configs/local.yaml`, e.g. `scenarios: {single_photon: {input: {n_events: 10}}}`; the file names don't include the number of events, so this replaces the full samples.

For other inputs, here I'll describe one specific example that Peter wrote for the di-photon separation analysis that
he and Anatolii were doing.
You will likely want to change this step, to fit your analysis needs.

A complete example of this can be found in `Di_Photon_Input_slcio.py`.
Start by making a director to store the input file, then call like;

```bash
source setup.sh
mkdir /data/dust/user/dayhallh/physical_recos/inputs
python3 Di_Photon_Input_slcio.py --pdg 22 --maxSep 30 --centralPosZ 50 --Emax 20 --Emin 20 --outdir /data/dust/user/dayhallh/physical_recos/inputs
```

This should yield a `gen-E2020pdg22maxSep30centralPosZ50-p1.csv` (the log) and, more interestingly, a `ILD-gen-E2020pdg22maxSep30centralPosZ50-p1.slcio`.
That is the file that is used to specify the position and direction of the incident particles in the simulation.


## Simulating events

The simulation runs the model inside `ddsim` through [DDML](https://github.com/key4hep/DDML), in the environment from [`setup.sh`](#setting-up-the-environment).

### Simulating the benchmark with each model

Each model is described by a file in `models/`, e.g. `models/caloclouds3.yaml`, which records:

- `representation`: how the model represents a shower, e.g. `point_cloud`;
- `fixed_at_training`: the factors fixed when it was trained, which it has no input for, e.g. the detector (and the compact file to simulate it in) and the particle;
- `context`: the factors it is given for each shower, e.g. the energy and direction, with their units and training ranges;
- `ddml`: how to run it, as DDML presets (from `python/ddml/configs.py` in DDML) passed to `ddsim --ml-model`, and the weights it loads, with their md5 sum.

`simulate_models.sh` simulates the input of every scenario with every model, following DDML's README:

```bash
./simulate_models.sh --dry-run                       # what would be simulated, and what is done already
./simulate_models.sh                                 # every model, every scenario
./simulate_models.sh -s single_photon caloclouds3    # one scenario, one model
./simulate_models.sh -n 10 caloclouds3               # a quick test: the first 10 events of each input
sbatch simulate_models.sh                            # as a slurm job, from the repository root
```

It always uses the DDML build from `scripts/build_ddml.sh` (building it first if needed) and gets its environment by sourcing `setup.sh` in a clean shell, inside the container unless `container.enabled` is false.
Before simulating, it checks that the DDML libraries, the `ddml` Python package and each preset's plugin come from that build, and that each model's weights match their md5 sum.

Each simulation writes `sim_output/<scenario>/<model>/<input name>.slcio` under `paths.data_dir`, with the `ddsim` log next to it.
Simulations whose output exists are skipped, so after a timeout or a failure, run it again to do the rest; `--force` simulates them again.
`ddsim` writes to a hidden file that is only renamed once it succeeds, so an output is always complete.
With `-n N`, the outputs are named `<input name>_n<N>.slcio`, so a test never counts as a full simulation.
Other input files can be simulated with `-i FILE`, which writes to `sim_output/custom/<model>/`.

All models use DDML's `scripts/ddsim_steer.py`. DDML's README names `ddsim_steer_cc3.py` for CaloClouds3, but that file only exists on DDML's `henry-all-showers` branch, and imports a function DDML has since replaced; apart from the particle gun settings, which an input file overrides, it is the same as `ddsim_steer.py`.

### Simulating the benchmark with Geant4

Full Geant4 simulation of the same inputs is the reference that the models are compared to.
`simulate_geant4.sh` works like `simulate_models.sh`, with the same options, and writes `sim_output/<scenario>/geant4/<input name>.edm4hep.root` under `paths.data_dir` (EDM4hep ROOT files, for the Gaudi-based analysis).
It runs `ddsim` with DDML's steering file but without DDML's fast simulation (`scripts/ddsim_steer_geant4.py`), so every other setting (physics list, crossing angle, filters, ...) is the one the models are simulated with, in the detector set by `geant4.compact` in the configuration.

Geant4 takes much longer than the models, so `simulate_geant4.sh` can simulate one input at a time, and `slurm_simulate_geant4.sh` submits a slurm job array with one input per job:

```bash
./simulate_geant4.sh --dry-run                                       # what would be simulated, and what is done already
./simulate_geant4.sh -s single_photon --file <input>.slcio           # just this input of the scenario
./slurm_simulate_geant4.sh                                           # a job array: one job per input still to simulate
./slurm_simulate_geant4.sh -m 10 -- --time 3-00:00:00                # at most 10 jobs at once, with a longer time limit
```

`slurm_simulate_geant4.sh` lists the inputs that still need simulating, building DDML first if needed, then submits the array.
Each job sources `setup.sh`, inside the container unless `container.enabled` is false (`scripts/run_in_environment.sh`), then runs `simulate_geant4.sh` on its input.
Each submission's list of inputs, job script and slurm logs go to `slurm/simulate_geant4_<time>/` under `paths.data_dir`; run it again to resubmit whatever failed or ran out of time.

In a shell that has sourced `setup.sh`, `simulate_geant4.sh` simulates right there; anywhere else it sets up the environment itself, as `simulate_models.sh` does.

### Simulating with your own steering file

Building DDML downloads its reference models into `models/` in the DDML clone, which is the default `paths.model_dir`, mounted as `/model_dir` for the steering file.
Put your own traced models there too, or point `paths.model_dir` at the folder that has them.

If your clone still has the old `ddfastshowerml` submodule, build DDML first (`scripts/build_ddml.sh`), then keep the models you added to it (they aren't tracked anywhere) before removing it:

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
It builds DDML first if needed, and will produce output under `sim_output` in `paths.data_dir`.


## Reconstructing events

The simulated events are reconstructed with ILD's standard reconstruction, run with Gaudi rather than Marlin:
ILDConfig's `ILDReconstruction.py` under `k4run`, as in the Key4hep tutorial [ILD Simulation and Reconstruction with Key4hep](https://key4hep.github.io/key4hep-doc/main/tutorials/key4hep-tutorials/gaudi_ild_reco/README.html).
`reconstruct_simulation.sh` reconstructs every finished simulation it finds of the scenarios in the configuration, by every model in `models/` and by Geant4:

```bash
./reconstruct_simulation.sh --dry-run                     # what would be reconstructed, and what is done already
./reconstruct_simulation.sh                               # every simulator, every scenario
./reconstruct_simulation.sh -s single_photon geant4       # one scenario, one simulator (a model name, or geant4)
./reconstruct_simulation.sh <data_dir>/sim_output/single_photon/caloclouds3/<name>.root   # one simulation
./reconstruct_simulation.sh -n 10 caloclouds3             # a quick test: the first 10 events of each
sbatch reconstruct_simulation.sh                          # as a slurm job, from the repository root
```

Each simulation `sim_output/<scenario>/<simulator>/<name>.root` (`.edm4hep.root` for Geant4) under `paths.data_dir` is reconstructed to `rec_output/<scenario>/<simulator>/<name>_REC.edm4hep.root`, with `<name>_AIDA.root`, `<name>_PfoAnalysis.root` and the log, `<name>.log`, next to it.
As with the simulation, reconstructions whose output exists are skipped (`--force` redoes them), `k4run` writes to hidden files that are only renamed once it succeeds, and with `-n N` the outputs are named `<name>_n<N>...`.
A simulation file outside `sim_output` is reconstructed to `rec_output/custom/files/`.

Each simulation is reconstructed in the detector it was simulated in: the model's `fixed_at_training.detector.compact_file`, or `geant4.compact`; `--compact FILE` overrides this.
`--noBeamCalReco` is always passed, as in the earlier Marlin reconstruction, and `--usingParticleGun` for `particle_gun` scenarios; add other arguments for `ILDReconstruction.py` to `reconstruction.extra_args` in the configuration, e.g. `[--lcioOutput, "on"]` to also write LCIO.

The TPC digitisation of the Key4hep releases so far (MarlinTrkProcessors 2.12.8) can stop the reconstruction of Geant4 events with `padTheta = nan`; the `marlintrkprocessors` patch fixes that (see [Fixes the Key4hep release doesn't have yet](#fixes-the-key4hep-release-doesnt-have-yet)).
Key4hep releases don't include ILDConfig, and the `ILDConfig` submodule is too old for recent releases, so ILDConfig is cloned to `detector_configs.ildconfig.source_dir` by `scripts/fetch_detector_configs.sh`.
Like `scripts/build_ddml.sh`, `setup.sh` runs it with `--if-needed` whenever it is sourced, which clones ILDConfig the first time and otherwise leaves the clone as it is; the scripts that start a container run it beforehand and mount the clone, since the container may have no network.
Run it yourself to update the clone to the head of its branch (`--dry-run` shows what it would do), and pin `detector_configs.ildconfig.commit` for benchmark runs; it prints the commit to paste into `configs/local.yaml`.
Like `simulate_geant4.sh`, it reconstructs right there in a shell that has sourced `setup.sh`, and otherwise sets up the environment itself, inside the container unless `container.enabled` is false.

`run_reco.sh` and `Di_Photon_rec_commands.sh` are the earlier Marlin reconstruction, with `MarlinStdReco_WITH_GAPS.xml`; apart from a hard-coded k4geo path, that steering file is the standard one, whose gap hits the Gaudi reconstruction uses too.

## Analysing the events

The output of the reconstruction is a `.csv` file, and should be fairly intuitive.
For the default photon separation reconstruction presented here, a plotting script is included.
The script is called `Di_Photon_Analysis_Plots.py`.

It's use is fairly rudimentary, and you will have to edit the paths used to find the `.csv` files produced by the output to match your environment.
This plotting script is not flexible and is indented more as inspiration, than for actual use.
