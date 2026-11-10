# The timing benchmark

The time per event of every simulator on the inputs of every scenario, on CPU
nodes and on GPU nodes. It runs in jobs of its own, apart from the pipeline's,
so that each kind of node can be asked for, and it simulates exactly as the
pipeline does: the same configuration, inputs, models and environment, and the
same `ddsim` command.

```bash
./slurm_timing.sh --dry-run     # what is timed, and the jobs it would submit
./slurm_timing.sh               # a job per input, simulator and kind of node
./timing_summary.sh             # the results, as a table: timing/summary.csv
```

The code is in [`ddml_timing/`](../ddml_timing/). It uses the pipeline's
harness, [`ddml_bench/`](../ddml_bench/), for the configuration, the scenarios,
the models, the environment and the `ddsim` commands, and changes none of it.

## What a task is

A task times one input of a scenario with one simulator on one kind of node.
The kinds of node are in the configuration's `timing.devices`
([`configs/default.yaml`](../configs/default.yaml)):

| Node | Inference | Simulators | Partition | Constraint |
|---|---|---|---|---|
| `cpu` | on the CPU | `all`: Geant4 and every model in `models/` | `maxcpu` | `EPYC&7402&512G` |
| `gpu` | on the GPU (`cuda`) | `models`: every model whose inference can run on the GPU | `maxgpu` | `A100-PCIE-80GB&GPUx1` |

Each enabled scenario's inputs are timed, each simulated with the first of the
scenario's `simulation_seeds`, if it has any. With the default configuration
that is 15 inputs (10 single-photon energies, 2 di-photon energies, 3 τ
files). On the `cpu` nodes all 5 simulators are timed, which is 75 tasks. On
the `gpu` nodes the 2 models that can use the GPU are timed, which is 30 tasks.

The task simulates the input's first `timing.warmup_events` + `timing.events`
events (10 + 200 by default), plus one more, whose start ends the last event
timed. The first `warmup_events` aren't counted: a model's first calls are
slower (TorchScript's profiling executor, and on the GPU the kernels' first
launch), as are the caches.

- **Single core.** `ddsim` runs on one CPU of the job's (`timing.pin_cpu`),
  as in the benchmark paper's timings, with one thread (`OMP_NUM_THREADS=1`,
  as the pipeline's simulations).
- **The node to itself.** Each job asks for its node to itself
  (`--exclusive`, in each kind of node's `sbatch_args`).
- **Output.** The simulation goes to the node's temporary folder and is deleted
  afterwards, so the network file system doesn't add to the times
  (`timing.keep_simulations: true` keeps it beside the results).

An input with fewer events than asked for is timed to its end, and its results
say so (`events.timed_fewer`).

## How each event is timed

`ddsim` prints a line as each event starts, before it reads the event from the
input:

```
GenerationInit   INFO  +++ Initializing event 12. Within run:0 event 12.
```

The runner ([`ddml_timing/runner.py`](../ddml_timing/runner.py)) reads
`ddsim`'s output through a pipe as it comes out. DD4hep flushes its output
before every line it prints, and `ddsim` runs under `stdbuf -oL` where it is
available. At each of these lines the runner notes the wall-clock time and
the CPU time `ddsim` has used so far, from `/proc/<pid>/stat`, to the clock
tick of 10 ms.

An event's time runs from its line to the next event's. It covers everything
the pipeline spends on that event: reading it, Geant4, the model's showers,
DDML placing their hits, and writing the event out. Everything before the
first event isn't in any event's time: Python, the geometry, the model's
loading, and Geant4's physics tables. Within each event the runner also
counts:

- `barrel_showers`: the showers a model placed in the ECAL barrel, from the
  line DDML prints for each (`PolyhedraBarrelGeometry::localToGlobal`).
- `model_calls` and `inference_s`: the calls of a model's embedded-Python
  inference, and how long they took (see below).

## The inference, and the GPU

A timing run uses its own steering file,
[`ddml_timing/ddsim_steer_timing.py`](../ddml_timing/ddsim_steer_timing.py). It
runs the simulator's own steering file, the one the pipeline uses: DDML's
`ddsim_steer.py`, `scripts/ddsim_steer_models.py` or
`scripts/ddsim_steer_geant4.py`. Before that, it sets up the timing of the
models' embedded-Python inference
([`ddml_timing/inference.py`](../ddml_timing/inference.py)).

DDML's embedded-Python plugins import a module and call its `run_inference()`
for every shower. Both modules the benchmark uses keep their network as a torch
module at the module's top level (`_model`):

- DDML's `python/examples/cc3_sf_2a_wrapper.py` (`caloclouds3`)
- `models/caloclouds3_rescaled.py`

The steering file imports the module first, as the plugin will, and the
plugin then gets the same module. Each of these torch modules is replaced
with a stand-in. The stand-in prints the time of each call
(`ddml-timing: inference <module>.<name> <seconds>`). On a GPU node it also
moves the network to the GPU, moves each call's tensors there, and brings the
result back to the CPU. Nothing else in the module changes, and a call on the
GPU is timed until its result is back.

Which models can use the GPU is found from DDML's own Python package, in the
environment, from each model's presets or configurations:

- **Embedded-Python plugins** (a `PythonModule`): `caloclouds3` and
  `caloclouds3_rescaled` run on the GPU.
- **LibTorch and ONNX Runtime plugins**: `caloclouds3_paper` and `l2lflows`
  run on the CPU only. DDML's `TorchInference` creates its tensors with
  `torch::kCPU`, and DDML has no embedded-Python plugin for L2LFlows. These
  are skipped on the GPU nodes, and the report says why. With
  `timing.devices.gpu.also_cpu_only_models: true` they are timed there too,
  on the node's CPU, and their results say `inference_on: cpu (LibTorch)`.

Before any GPU job is submitted, the PyTorch in the environment must have been
built with CUDA. If it wasn't, the GPU tasks are blocked, saying so, and the
CPU tasks are still submitted. In each GPU job, PyTorch must also find a GPU,
or `ddsim` stops at once, saying why. On a GPU node the job passes the GPU
into the container (`APPTAINER_NV=1`, i.e. `apptainer --nv`, or
`SINGULARITY_NV=1`) when `container.enabled` is true.

What isn't known until it runs on a GPU node: whether each TorchScript network
runs on the GPU as it is. A network that makes tensors on the CPU inside itself
fails on the GPU, and the task's log and `.failed.json` then show PyTorch's
error.

## Results

Under `<paths.data_dir>/timing/<node>/<scenario>/<simulator>/`, for each input,
named after its simulation in the pipeline with `_t<N>` for the N events timed:

| File | What it holds |
|---|---|
| `<name>_t<N>.json` | the task's settings (`identity`), the node (host, CPU model, GPUs, slurm job), the software, what the inference ran on, the `ddsim` command, and the statistics of the times |
| `<name>_t<N>.events.csv` | each event: `event`, `warmup`, `start_s`, `wall_s`, `cpu_s`, `model_calls`, `inference_s`, `barrel_showers` |
| `<name>_t<N>.log` | what ran, at the top, then `ddsim`'s output |
| `<name>_t<N>.failed.json` | why the last attempt failed, until one succeeds |

The statistics are kept for the wall-clock time, the CPU time, the model calls,
the inference time and the showers per event. Each has the mean, the standard
deviation, the standard error and its share of the mean (`rel_sem`), the
median, the minimum, the maximum and the total. It also has the means of the
first and second halves of the events timed, with their difference in units of
its uncertainty (`drift_z`), which shows whether the warm-up was long enough.
The warm-up's own times are kept as `warmup_wall`.

A task is done when its `.json` records the settings the run would time it
with: the events, the warm-up, where the inference runs, the CPU pinning, the
input and seed, and what the simulator is. For a model, that last one means
its presets or configurations, weights and the sha256 sums of its description
and own code, as the pipeline's manifests record them. Changing any of these
times it again.

`./timing_summary.sh` gathers the results into one table, printed and written
to `timing/summary.csv`. It has a row for each result:

- `mean_s`, `sd_s`, `sem_s`, `rel_sem_pct`: the mean wall-clock time per
  event, with its spread, its standard error, and that error as a percentage
  of the mean.
- `median_s` and `cpu_mean_s`: the median, and the mean CPU time per event.
- `showers_per_event`, `inference_per_event_s`, `model_calls_per_event`: the
  model's showers per event, and where they were timed, its inference calls
  and their time per event.
- `events_for_10pct`: how many events a 10% standard error needs, at the
  spread seen.
- `speedup` and `speedup_error`: Geant4's mean time per event of the same
  input over the model's, with its uncertainty from both standard errors.
  Geant4 is taken from the same kind of node, or else from the `cpu` nodes,
  as `speedup_reference` says.
- The node (`host`, `cpu_model`, `gpu`), `inference_on` and `drift_z`.

It also has a row for each kind of node, scenario and simulator over all its
inputs (`input` is `all`, or `all_t<N>` for each number of events timed where
there are results of several). That row is the mean of the inputs' means, each
input weighing the same: for the single photons that is every energy alike, as
the paper's average over 10–100 GeV. Its standard error comes from the inputs'
standard errors.

## Accuracy against time: the Pareto fronts

Whenever the plots are drawn (`plot_observables.sh`, or the pipeline's plot
stage) and `timing/summary.csv` is there, each scenario also gets Pareto
fronts for each kind of node in the table
([`ddml_bench/pareto.py`](../ddml_bench/pareto.py)):
`pareto_js_<node>__<scenario>.png` and `pareto_wasserstein_<node>__<scenario>.png`
next to the other figures, every metric of the scenario on one axis with a
marker each, and `pareto.csv` with every point. Each model's time is the row
of its scenario over all the inputs (`all`, or the `all_t<N>` of the most
events) on that kind of node, and Geant4's the same on that kind of node, or
else on the `cpu` nodes. Its accuracy in each metric is its Jensen–Shannon
distance, or its Wasserstein distance, from Geant4 (the χ² for the energy
figures and the π⁰ categories), divided by the mean of the models', so that
every metric is on the same scale.
The README's [Pareto fronts](../README.md#the-pareto-fronts) section has the
details. The pipeline draws the plots again when the table is newer than
`metrics.csv`.

## Options

`slurm_timing.sh`, which submits the jobs, and `timing.sh`, which times here,
one task after another, take the same options:

```bash
./slurm_timing.sh -s tau_pairs caloclouds3     # one scenario, one simulator
./slurm_timing.sh -d gpu                        # only the GPU nodes
./slurm_timing.sh -n 20 --warmup 2              # a quick try: 20 events each
./slurm_timing.sh --force                       # time again what is timed
./slurm_timing.sh -- --account=myproject        # more sbatch options
./timing.sh -d gpu -s single_photon caloclouds3 # on this (GPU) node
```

`timing.sh` times as the `cpu` nodes unless given `-d`. Its results go where
the jobs' do, and record the node they were timed on.

`slurm_timing.sh` runs on a login node, from the repository root. Before
submitting anything, it works out what is done in the environment the jobs
have, as `slurm_pipeline.sh` does, building DDML first if needed. It then runs
the pipeline's checks of the Geant4 and models stages: the DDML build, `ddsim`,
the compact files, and each model's plugins, weights and modules. So a missing
weight stops it there, rather than failing every job.

The inputs must be there already: `./generate_input.sh` makes the guns', and
the τ files are put in `input_physics_samples/` by hand. A task whose input
isn't there is blocked.

Each submission gets a folder, `timing/slurm/timing_<time>/`, with:

- the plan and the tasks with their statuses
- the job script
- the jobs submitted (`jobs.tsv`)
- the report
- each job's slurm log

A task that a job of an earlier submission is still queued or running for
isn't submitted again, so `./slurm_timing.sh` can be run again at any time to
resubmit what failed or timed out.

## Changing it

- **The constraints, partitions and time limits**: `timing.devices.<node>` in
  `configs/local.yaml`, e.g. `partition: allgpu`. Add `--gres=gpu:1` to its
  `sbatch_args` if the cluster wants the GPU asked for.
- **Another kind of node**: an entry under `timing.devices` in
  `configs/default.yaml`, with its `inference`, `simulators` and slurm
  settings.
- **More or fewer events**: `timing.events` and `timing.warmup_events`. The
  results of each number of events are files of their own (`_t<N>`).
