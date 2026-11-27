# Testing a model at low cost

`low_cost.sh` tests a model on the first events of each input, against the full samples of Geant4, its second sample and the published models. Only the model's sample is simulated, reconstructed and measured, at a part of the full benchmark's cost, and it is drawn against the others' full measurements, fetched as published ([Reusing published measurements](published.md#reusing-published-measurements)). It runs anywhere the pipeline does, and on GitHub's runners for a model pushed to a fork ([On GitHub's runners](#on-githubs-runners)):

```bash
# add your model to models/ (models/mymodel.yaml), then
./low_cost.sh --dry-run mymodel          # what it fetches, what it costs, and what it would run
./low_cost.sh mymodel                    # here, one task after another
./low_cost.sh --executor slurm mymodel   # as slurm jobs (options after -- go to sbatch)
./low_cost.sh -n 250 mymodel             # the first 250 events of every input
```

A run, in order:

- checks that the configuration pins the software the published measurements were made with: `key4hep.release`, `ddml.commit` and `detector_configs.ildconfig.commit`. A model simulated with other software isn't compared like with like, so a run stops, printing the lines to put in `configs/local.yaml` (the next run builds DDML again at that commit, if needed). `--any-stack` compares anyway, with a warning;
- fetches what the full benchmark would fetch, where it isn't there: the inputs, the full measurements of Geant4, its second sample and the published models, and the timing table. A data folder with the full benchmark's own measurements (on a cluster, say) fetches none of them, and `--no-fetch` compares with what is there;
- prints what the sample costs, scenario by scenario, estimated from the published time per event of a model like it on the CPU nodes (`low_cost.estimate`);
- simulates, reconstructs and measures the model's first N events of each input (of each segment, for inputs in segments), `observables/<model>_n<N>.npz`, then draws them against the full measurements in `observables/plots/low_cost_n<N>/`: every figure, `metrics.csv`, `fits.csv`, the Pareto fronts and `dashboard.html`. The full sample's plots aren't touched.

The number of events is `low_cost.events` in the configuration, 1000 of each input of each scenario by default, and `-n N` gives every scenario N. Cost and statistics trade differently in each scenario, so each can have a number of its own: the τ pairs cost the most per event, as Geant4 simulates most of their particles whatever the model (about two thirds of a CaloClouds3-like model's 20 core-hours at 1000 events each). The sample is then named after all of them, the most common number first and each scenario of another after it, e.g. `observables/<model>_n1000_tau_pairs500.npz` and `plots/low_cost_n1000_tau_pairs500/`, and each simulation after its own scenario's number (`<input>_n500_simseed42...`), so what a scenario has simulated already with that number isn't simulated again. To leave a scenario out, disable it (`scenarios.<scenario>.enabled: false`).

A sample of few events is noisier than a full one, which pushes its metrics up. So in its figures:

- its legend entry gives its events of each input, e.g. `MyModel (n=1000)`, the text in the corner its events (showers, π⁰s) in the figure, and each of its metrics has its **statistical floor**: what the metric would be if the model were Geant4 itself, with as many events of each input. A metric within its floor (and the floor's spread) is as close to Geant4 as that many events can tell. The model and Geant4 simulate the same events, so the floor of an observable stored per event (the energy, the angles, the separation, the π⁰s) is the mean over 100 draws of as many of Geant4's second sample's events of the same inputs, compared with Geant4 as the model is. A profile, stored as each input's sums, has each bin's sums drawn about their means with their variances, file by file ([`ddml_bench/floors.py`](../ddml_bench/floors.py)). The χ² has no floor, as its uncertainties already take both samples' sizes;
- `metrics.csv` gives each of its metrics its `events`, `floor` and `floor_uncertainty` (the spread of the draws, the metric's scatter from statistics alone), and `projected`, the metric taken to the full sample's size. That is, for a divergence (JS), its value less its floor plus the full sample's floor, what Geant4's second sample measures; for a distance (W1, L1), the same in quadrature; and for a χ², its excess over its degrees of freedom, scaled as its uncertainties shrink with more events, on top of the second sample's χ². These are first order: exact for the Jensen–Shannon divergence's expected value, approximate for the rest, and noisy where the metric is near its floor. For Geant4's second sample, `floor` gives the profiles' floors at the full size, drawn as a sample's are, beside what it measured: a check of how the profiles are drawn;
- the [Pareto fronts](stages.md#the-pareto-fronts) draw the sample at its projected metrics, with the floor's spread as a dashed error bar, where the model is timed (the [timing benchmark](timing.md), on the kind of node the published timing is of);
- the π⁰ categories are counted per 1000 events of each seed, so samples of other sizes compare;
- a closure test needs its twin's sample of the same size, and is otherwise skipped.

`plot_observables.sh --sample MODEL[:N]` draws the same figures again (without `:N`, of the sample of `low_cost.events`).

## On GitHub's runners

Fork FaSiNaTe, add a model to `models/` and push: the workflow [`.github/workflows/low-cost.yml`](../.github/workflows/low-cost.yml) tests every model in `models/` without published full measurements in the low-cost mode, on GitHub's free runners, and puts the plots on the run's page. [`docs/github.md`](github.md) says how, step by step.

- **The jobs.** A plan fetches what is published and splits each model's sample into chains: a simulation, its reconstruction, and the measurement of that one reconstruction. It packs the chains into shards, each a job of its own in Key4hep's AlmaLinux 9 container with CVMFS, as DDML's own CI runs. The software is built once, into a cache. A report merges the chains' measurements into the model's `.npz`, the same, array for array, as measuring them together, and draws it against the full samples.
- **The configuration** is [`configs/github.yaml`](../configs/github.yaml), with the software pinned to the published measurements', and the `github` section of `configs/default.yaml`.
- **The cost.** For a model about as fast as CaloClouds3 at 1000 events of each input: 46 chains in 5 shards of about 2.6 h, about 14 runner-hours. A sample is made once: a later run of the same model, software and settings takes it, for 90 days. A plan over `github.budget_hours` is refused.
- **The results.** The run's summary has each metric with its floor and projection, beside Geant4's second sample's and the best published model's. Also on the run's page: `dashboard.html`, unzipped, and `low-cost-results`, with the figures, CSVs and measurements. They can go on the fork's GitHub Pages too.
- **No Pareto fronts yet**, as the model isn't timed on the runners.

The jobs are `ddml-bench github plan|check|shard|report`, which run anywhere the pipeline does ([`ddml_bench/github.py`](../ddml_bench/github.py)).
