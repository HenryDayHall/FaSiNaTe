# Developing FaSiNaTe

Extending the benchmark, the tests, and where everything is. The harness every way of running the pipeline goes through is described in [harness.md](harness.md).

## Extending the benchmark

A model needs no code: see [Models](models.md).

### Adding a scenario

Add an entry under `scenarios` with an `input` section, in `configs/default.yaml` (override files can change a scenario, and disable it with `enabled: false`, but not add one). Input types are defined in `INPUT_TYPES` in [`ddml_bench/scenarios.py`](../ddml_bench/scenarios.py) (currently `particle_gun`, `di_photon_gun` and `external`); a new type of inputs made here needs a function that returns the commands making the inputs, one that names the files they produce without running them, the option that makes a command write into a given folder, the Python modules its commands need, whether its events are a particle gun's (for `--usingParticleGun`), and the folder its inputs go in by default. Inputs made elsewhere need no new type: list the files under an `external` input. To simulate each input several times with different `ddsim` seeds, list them in the scenario's `simulation_seeds` (with `one_seed_per_input: true`, each input once with its own). For shorter jobs, put each input's events in several files with `input.segments` (a new gun needs `--segment` and `--segments` options for it, as the two guns have). List the scenario under the `scenarios` of each observable that should be measured on it.

### Adding an observable

Observables are implemented in [`ddml_bench/measure.py`](../ddml_bench/measure.py), keyed by the name used under `observables` in the configuration: binned profiles of the hits in `KINDS`, and quantities kept per event in `PER_EVENT` (each with a function that checks its settings, one that names what it stores of each event, and one that measures those on a chunk of events; those that keep some items per object of an event, as `pi0` does per candidate, say which in `PER_CANDIDATE`), whose figures are in `EVENT_FIGURES` in [`ddml_bench/plot.py`](../ddml_bench/plot.py), as are the comparison metrics. Settings that only change the figures go in `DRAWING`, so that changing them doesn't measure again.

### Adding a stage

Each stage is a class in [`ddml_bench/stages/`](../ddml_bench/stages/) that adds its tasks to the graph and says when one is done, what it needs from the environment and how to run one; [`docs/harness.md`](harness.md) describes them.

## Tests

```bash
python3 -m pip install pyyaml numpy awkward uproot matplotlib h5py particle pytest
python3 -m pytest tests
```

The tests run the scripts end to end, `pipeline.sh` and `slurm_pipeline.sh` included, through the same launcher and `setup.sh` as on the cluster, with only the programs at the ends faked ([`tests/fakes/`](../tests/fakes/)): `ddsim` (which runs the steering files as `ddsim` does, with stand-ins for DD4hep's `DDSim` and `g4units`), `k4run` (which writes real EDM4hep-style ROOT files, so the observables and plots run for real), pyLCIO, DDML's Python package (with DDML's presets), and slurm's `sbatch` and `squeue`. They need no Key4hep, CVMFS or slurm, and run in CI on every push ([`.github/workflows/ci.yml`](../.github/workflows/ci.yml)), with shellcheck on the scripts. The jobs of the GitHub workflow run there too, each in a copy of the repository of its own, as on runners of their own (`tests/test_github.py`, with a stand-in for GitHub's `gh`).

## Repository layout

```
├── generate_input.sh                   # stage 1
├── simulate_geant4.sh                  # stage 2
├── simulate_models.sh                  # stage 3
├── reconstruct_simulation.sh           # stage 4
├── observables_from_reconstruction.sh  # stage 5
├── plot_observables.sh                 # stage 6
├── pipeline.sh                         # all six stages, here, one task after another
├── slurm_pipeline.sh                   # all six stages, as slurm jobs of one file each
├── low_cost.sh                         # a model on the first events of each input, against the published full samples
├── slurm_simulate_geant4.sh            #   ... just the Geant4 stage
├── slurm_timing.sh                     # the timing benchmark, as slurm jobs on CPU and GPU nodes
├── timing.sh                           #   ... here, one task after another
├── timing_summary.sh                   #   ... and its results in one table
├── setup.sh                            # interactive environment
├── ILD_Particle_Gun.py                 # particle-gun input generator
├── ILD_Di_Photon_Gun.py                # di-photon input generator
├── configs/default.yaml                # shared configuration (override in configs/local.yaml)
├── configs/pipeline_test.yaml          # a very light test run of the whole pipeline
├── configs/github.yaml                 # the low-cost mode on GitHub's runners (.github/workflows/low-cost.yml)
├── models/                             # one YAML file per model (and a model's own Python code and weights)
├── unused_models/                      # model descriptions kept out of the runs
├── ddml_bench/                         # the harness: layout, stages, task graph, executors
├── ddml_timing/                        # the timing benchmark (docs/timing.md)
├── scripts/                            # building, the environment launcher, the steering files, ddml-bench, ddml-timing
├── tests/                              # the tests, with the fakes they run the pipeline on
└── docs/                               # the documentation beyond the README, and its figures
```

Some files predate the current pipeline and are kept for reference only: `create_singularity.sh` (an old Ubuntu 22.04 container setup; the image is now pulled automatically) and `MODELS.md` (names of older CaloClouds variants).
