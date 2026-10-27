# The harness

`ddml_bench/` is the harness of the benchmark plan: the code every way of
running the pipeline goes through, on a laptop, on slurm, and later on GitHub
Actions runners. It has one module that names every file, a registry of the
stages that says what each stage's tasks are, one task graph, and executors
that run the graph's tasks.

```
ddml_bench/
├── config.py           the configuration (configs/*.yaml), as before
├── layout.py           every path: the one module that names files
├── scenarios.py        the scenarios' inputs: their commands and file names
├── models.py           the simulators: Geant4 and models/*.yaml
├── ddml_configs.py     DDML model configurations written out in models/*.yaml
├── stages/             the stage registry, in pipeline order
│   ├── base.py         what a stage declares
│   ├── generate.py     stage 1
│   ├── simulate.py     stages 2 (geant4) and 3 (models)
│   ├── reconstruct.py  stage 4
│   ├── observables.py  stage 5 (the measuring itself is in measure.py)
│   └── plot.py         stage 6 (the figures and metrics are in plot.py)
├── graph.py            the plan, the task graph and each task's status
├── executors/          local (one task after another) and slurm (a job each)
├── environment.py      getting into the environment from setup.sh
├── manifest.py         the manifests of a data folder's records, for publishing it
├── fetch.py            fetching published files instead of making them
├── process.py          running ddsim, k4run and the gun, with logs and clean-up
└── cli.py              scripts/ddml-bench, which the shell scripts run
```

## How a run goes

```
pipeline.sh, stage scripts                  slurm_pipeline.sh
  ddml-bench run --executor local             ddml-bench run --executor slurm
  ddml-bench stage NAME
          │                                            │
          ├─ make_plan(): read the configuration, the models and the options,
          │  check them, name every input (no environment needed)
          │                                            │
  run_in_environment.sh                       run_in_environment.sh
  ddml-bench local PLAN                       ddml-bench tasks PLAN TASKS
    Graph(plan).find_status()                   Graph(plan).find_status()
    each stage's check_environment()            each stage's check_environment()
    run_task() for each task to do, in order          │
                                              submit: sbatch job.sh KEY, a job per task,
                                              --dependency=afterok on the tasks it needs
                                                       │ (in each job)
                                              run_in_environment.sh
                                              ddml-bench run-task DIR KEY → run_task()
```

`make_plan()` runs where the configuration files are, before anything is
built, so mistakes in the configuration or the options show up at once. It
also names every input: the gun's module imports what it needs to generate
events only when it does, so the names need nothing beyond the standard
library. Working out what is done needs the environment (the observables'
check reads `.npz` files with numpy), as do the stages' environment checks,
so that happens in the environment the tasks run in, before anything runs or
is submitted.

## The layout

`layout.py` names every file under `paths.data_dir`: inputs, simulations,
reconstructions, measurements, plots and logs. A `Layout` is one sample: the
full inputs, or the first N events of each (`-n N`). The sample is chosen
once, when an input is simulated, and its name carries on: the simulation of
the first N events of `<input>` is `<input>_n<N>`, its reconstruction
`<input>_n<N>_REC.edm4hep.root`, the measurements `<simulator>_n<N>.npz`, and
the metrics `metrics_n<N>.csv`. A scenario with `simulation_seeds` has each
input simulated once with each `ddsim` seed, and the seed goes into the names
the same way: `<input>_simseed<S>`, or `<input>_n<N>_simseed<S>`. Every task
records its seed (`null` for those without one of their own), and the
observables record each reconstructed file's seed in the `.npz`. Records of
files, such as the reconstructed files in a `.npz`, are relative to the data
folder (`Layout.relative`), so they hold when the data is copied elsewhere.

## The stage registry

A stage is a class in `stages/` (see `stages/base.py`), and `stages.STAGES`
lists them in order. A stage:

- adds its tasks to the graph (`add_tasks`), one per file it makes, using the
  tasks of the stages before it (`graph.inputs`, `graph.simulations`,
  `graph.reconstructions`, `graph.measuring`);
- says when a task is done (`current`): by default when its output exists; the
  observables when their `.npz` is up to date by what it records (the
  settings and the reconstructed files, which needn't be there any more), the
  plots when `metrics.csv` is newer than every `.npz`;
- says what in the environment would stop its tasks (`check_environment`):
  the DDML build, the plugins of the presets and model configurations, the
  weights' md5 sums (and that the plugins load those weights), the compact
  files, ILDConfig, the Python modules each needs;
- runs one task (`run`), raising `StageError` if it fails.

Five attributes shape how the graph treats a stage's tasks: `gathers` (a
task out of date while anything it gathers is still to be made: the
observables and the plots), `records_inputs` (a task that gathers, but knows
from its own record of what it gathered whether it is up to date, so what it
gathers that is to be made only because it isn't there doesn't count: the
observables), `takes_what_there_is` (the plots, of whatever is measured,
rather than waiting for everything), `intermediate` (made only for what is
made from it: the simulations and reconstructions) and `blocked_is_error` (an
input that isn't made, for the simulations, makes a local run fail).

A task is a plain dict, so it can be written to `tasks.json` and run from
there alone:

```json
{
  "key": "reconstruct:/data/rec_output/single_photon/geant4/p22_..._seed0_ip_surface_REC.edm4hep.root",
  "stage": "reconstruct",
  "scenario": "single_photon",
  "simulator": "geant4",
  "name": "p22_..._seed0_ip_surface",
  "seed": null,
  "inputs": ["/data/sim_output/single_photon/geant4/p22_..._seed0_ip_surface.edm4hep.root"],
  "input_file": "/data/input_single_particle_gun/p22_..._seed0_ip_surface.slcio",
  "output": "/data/rec_output/single_photon/geant4/p22_..._seed0_ip_surface_REC.edm4hep.root",
  "needs": ["geant4:/data/sim_output/single_photon/geant4/p22_..._seed0_ip_surface.edm4hep.root"],
  "params": {"base": "...", "compact": "ILD/compact/ILD_l5_o1_v02/ILD_l5_o1_v02.xml",
             "events": -1, "args": ["--noBeamCalReco", "--usingParticleGun"],
             "done_suffix": "_REC.edm4hep.root", "ildconfig_dir": "..."},
  "status": "todo", "why": "not made yet", "redo": false, "changes": false,
  "after": ["geant4:..."]
}
```

The key is `<stage>:<output>`, as only one task makes each file; slurm
submissions record it in `jobs.tsv`, so a later submission waits for a job
still queued for the same task rather than submitting it again.

## How a model runs

A model's description says how DDML runs it (`ddml_bench/models.py`): with
DDML's presets, passed to ddsim with `--ml-model` and DDML's steering file,
as DDML's README runs them; or with DDML model configurations written out
(`ddml.model_configs`), each a preset to start from and the fields of DDML's
`ModelConfig` to set, for a model DDML has no preset for. Those run with
`scripts/ddsim_steer_models.py`, each configuration an option
`--ml-config '<JSON>'` (`ddml_configs.option()`, so a simulation's log has
everything it ran with): the steering file runs DDML's steering file with
DDML's own Python package, swapping only DDML's `get_fastsim_configuration()`
for one that makes the configurations of those options with
`ddml_configs.build()`. The models stage's environment check makes them with
the same function, so what it checks is what ddsim runs. Geant4 runs
`scripts/ddsim_steer_geant4.py` in the same way, with no fast simulation, so
every simulator shares every other setting of DDML's steering file. The
manifest records the presets or configurations of each model's simulations,
from their logs, and fetching published measurements of a model needs them to
be the model's.

A model can be the closure test of another (`closure`): the same model
through another backend. The plots (`ddml_bench/plot.py`, `closures()`) then
draw it against that one alone, as the reference, and say for each scenario
whether everything the two measured is the same, number for number, as it
should be where both simulated the same events with the same seeds and the
description says so.

## The task graph

`Graph(plan)` asks each stage for its tasks, in order, so each task comes
after the tasks it needs, and checks that no two tasks make the same file.
`find_status()` gives every task a status from what is on disk: `done`,
`todo`, `blocked` (it needs what isn't done and won't be made this time),
`unneeded` (an intermediate task, a simulation or reconstruction, that isn't
there, but that nothing still to be made needs) or `unselected` (not chosen
by `--stages`, the simulators, `--scenario` or the files given). A task whose
output exists stays done even if something it needs is still to be made,
unless that is made anew (`changes`: forced with `--force`, made from
something made anew, or there but out of date): then it is made again too
(`redo`).

Then, from the last task back, an intermediate task that is to be made only
because it isn't there is `unneeded` if every task that needs it, selected or
not, is done without it or isn't needed either. A task is done without what
it needs when its output is there (it stays done when that is made), or when
its stage `records_inputs`: the measurements, up to date by their `.npz`
file's record, need none of their reconstructions, so neither those nor their
simulations are made. The plots don't count, as they plot whatever is
measured, so measurements still to be made are always needed. Whatever stages
run, the simulations and reconstructions of a simulator whose `.npz` file is
up to date are then left out, whether they were deleted after measuring or
the `.npz` file was fetched with the inputs: adding a model to fetched
references runs only its own tasks and the plots. Forcing a stage makes its
tasks all the same, and so does giving a simulation file to reconstruct.

## Executors

- **local** (`executors/local.py`): runs the tasks to do one after another in
  this environment; a task whose needs failed is skipped, and the others go
  on. `pipeline.sh` and the stage scripts use it.
- **slurm** (`executors/slurm.py`): submits a job per task to do, each waiting
  for the jobs of the tasks it needs, and waits for jobs of earlier
  submissions still queued for the same tasks. `slurm_pipeline.sh` uses it.

Both run a task with `executors.run_task()`, which a slurm job reaches through
`ddml-bench run-task DIR KEY`. A GitHub Actions shard, one task per job of a
matrix, will run the same command.

## The environment

`scripts/run_in_environment.sh` is the one way into the environment the
tasks run in: it builds what `setup.sh` needs if it isn't there (DDML,
ILDConfig, the patches), then sources `setup.sh` in a clean shell, inside the
container unless `container.enabled` is false, with the repository, the data,
the software and the configuration files mounted at the same paths. The
clean shell and the container are set up by `scripts/environment.sh`, which
`build_ddml.sh` and `build_patches.sh` use too. In a shell that has sourced
`setup.sh`, the harness runs right there.

## Tests

`tests/` runs the scripts end to end, through the launcher and `setup.sh`,
with fakes for the programs at the ends (`tests/fakes/`): `ddsim`, `k4run`
(writing real EDM4hep-style ROOT files, so the observables and the plots run
for real), pyLCIO, DDML's Python package, and slurm's `sbatch` and `squeue`
(running each job as it is submitted, or queueing them). See the README's
Tests section; CI runs them on every push.

## Extending it

- **A stage**: a class in `stages/` with the methods above, added to
  `stages.STAGES` in its place in the order; its tasks name their files with
  the layout, and its runner writes to a hidden file that is renamed when
  complete, as the others do.
- **An executor**: anything that runs the graph's tasks to do (status `todo`)
  in an order where each comes after its `after`, by `run_task()` or
  `ddml-bench run-task`.
- **A published output**: `fetch.prepare()` fetches, before anything works out
  what is done, the files of the records in `published` that tasks of the run
  would make (the inputs, and the measurements by what their manifest says
  they hold), checked by sha256, so the tasks are then done; with `--dry-run`
  it puts them in the plan (`to_fetch`), and `find_status()` gives those tasks
  `fetch`: to do, needing nothing, and done without what they need when the
  graph leaves out what isn't needed. It runs where the plan is made, so on
  slurm the login node fetches and the jobs need no internet. Fetching another
  kind of output is a check in `fetch.choose()` of whether the published file
  is what the task would make.
