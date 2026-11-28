# Testing a model on GitHub's runners

The [low-cost mode](low_cost.md) tests a model on the first events of each input, against the published full samples of Geant4, its second sample and the published models. The workflow [`.github/workflows/low-cost.yml`](../.github/workflows/low-cost.yml) runs it on GitHub's free runners. Fork FaSiNaTe, add a model to `models/`, push, and the model is simulated, reconstructed, measured and drawn against the others, with the plots on the run's page.

Nothing is simulated at full statistics on GitHub. The full samples are made on a cluster and published on Zenodo, and every run compares with them.

## In a fork

1. **Fork** the repository and keep the fork public. Runners are free for public repositories; a private one has a few dollars of minutes a month, which a run would use up.
2. **Turn Actions on.** In a fork, workflows stay off until you open the *Actions* tab and enable them.
3. **Add your model** as `models/<name>.yaml` ([How to add a model](../README.md#how-to-add-a-model), [Models](models.md)), with its own Python module if it has one. Its weights go either in DDML's build, as DDML's presets' do, or in the repository (`ddml.weights_in: fasinate`), under GitHub's 100 MB per file.
4. **Push**, to any branch. A push that changes `models/`, `configs/github.yaml` or the workflow runs it. The run is under *Actions* → *low-cost benchmark*. The plan job's summary says what is tested and what it should cost. The report's summary has the metrics and links to the plots.

A push of the default branch can also put the plots on the fork's GitHub Pages:

- under *Settings* → *Pages*, set *Source* to *GitHub Actions*;
- under *Settings* → *Secrets and variables* → *Actions* → *Variables*, add `FASINATE_PAGES` with the value `true`.

The plots are then at `https://<owner>.github.io/<repository>/`. Each run replaces the last.

To run it by hand, use *Run workflow* on its page under *Actions*. That button appears once the workflow is on the default branch. Its options are:

- **models**: the models to test, separated by spaces. By default, every model in `models/` without published full measurements is tested. A published model can be named too, to compare its low-cost sample with its full one.
- **events**: the first N events of each input of every scenario. By default, `low_cost.events` in `configs/github.yaml` is used.
- **fresh**: make the samples again, even those an earlier run made.

## What runs

| Job | Where | What it does | Takes |
| --- | --- | --- | --- |
| plan | the runner | Fetches what is published (cached between runs), checks that the software is pinned to what the published measurements were made with, and chooses the models. Splits each model's sample into chains (a simulation, its reconstruction, and the measurement of that one reconstruction). Packs the chains into shards. | ~2 min |
| software | Key4hep's container | Builds DDML, ILDConfig and the patches against the pinned Key4hep release, once, into the cache, without their build folders. Then checks that what is cached can run every chain. | ~1 min when cached; the first build, longer |
| preflight | Key4hep's container | Checks each model with chains before any of it is simulated ([stages.md](stages.md#0-check-the-models)). Keeps what it found, the summaries, full logs and plots of the barrel shots, as the artifact `low-cost-preflight`, and puts the summaries in the job's summary. The shards start only if every model passes. | a few minutes |
| shard | Key4hep's container, a job each | Runs its chains, `github.parallel` at once, the longest first. Keeps each chain's measurement and logs, and deletes the simulation and reconstruction. | up to `github.shard_hours` |
| report | the runner | Merges each model's measurements into one `.npz` and draws it against the full samples. Writes the summary. | ~5 min |
| keep | the runner | Keeps each sample made, as an artifact, for 90 days. | seconds |
| pages | the runner | Publishes the plots, if `FASINATE_PAGES` is `true`, from the default branch. | ~1 min |

The jobs with the software run in `ghcr.io/key4hep/key4hep-images/alma9-cvmfs`, Key4hep's AlmaLinux 9 image with CVMFS, as [DDML's own CI](https://github.com/key4hep/DDML/tree/main/.github/workflows) does: `--privileged`, then `/mount.sh`. The pipeline then needs no container of its own (`container.enabled: false` in [`configs/github.yaml`](../configs/github.yaml)). The configuration also pins `key4hep.release`, `ddml.commit` and `detector_configs.ildconfig.commit` to the published record's. When `published.records` moves to a record made with other software, the plan stops and prints the new pins to put there.

[`ddml_bench/github.py`](../ddml_bench/github.py) has the details.

## What it costs

The plan estimates each chain as `./low_cost.sh` does (`low_cost.estimate`: a published model's time per event on the timed CPU nodes, the reconstruction's, and a start-up), divided by `github.speed`, the runner's speed against one of those cores (0.7, a guess). For a model about as fast as CaloClouds3 at 1000 events of each input:

| Scenario | Chains | Each |
| --- | ---: | ---: |
| single photon | 10 | 0.2 h |
| di-photon | 27 (9 segments × 3 seeds) | 0.2 h |
| τ pairs | 9 | 2.1 h |

That is 5 shards of about 2.6 h each, two chains at once on each, about 14 runner-hours. The first run of a fork also builds the software.

The report says what the chains took against their estimates, and so how fast the runners were. Set `github.speed` to that, and the next plans will be closer.

Runs are kept small:

- **Only what is new runs.** Only models without published full measurements are tested. A sample made before, of the same model, software and settings, is reused for 90 days instead of being made again. Its fingerprint covers the model's description (but not its label or plot style), its own Python code, the software, the events, the observables, and the inputs. So adding a second model later runs only that one. Changing a model's colour runs only the report.
- **Budget.** A plan over `github.budget_hours` of runner time (40) is refused, as is one with a shard over `github.deadline_hours`. Use fewer events of the costly scenarios, e.g. `low_cost.events.tau_pairs: 500` in `configs/github.yaml`, or fewer models at once.
- **One run per branch at a time.** A push while one runs waits for it, and then takes what it made.

## The results

- **The report's summary**, on the run's page, has a table for each model. Each row has the metric of a figure, its statistical floor at the sample's size, and its value projected to the full sample's size. Next to these are Geant4's second sample's value and the best published model's. A metric within its floor is as close to Geant4 as that many events can tell. ([Testing a model at low cost](low_cost.md) explains the floors and projections.)
- **`dashboard.html`**, uploaded unzipped, opens in the browser from the run's page. It has every figure, with a menu to show or hide each model.
- **`low-cost-results`** has the figures as PNG and PDF, plus `metrics.csv`, `fits.csv`, `run.json` (the plan, with what each chain took) and the models' merged measurements (`samples/<model>_n<N>.npz`). `plot_observables.sh --sample <model>:<N>` draws them again elsewhere, next to the published full samples.

There are no Pareto fronts yet: the model isn't timed on the runners.

## When something fails

- **A model that fails its preflight** fails the preflight job, so no shard runs. The job's summary says which step failed and why; the artifact `low-cost-preflight` has each model's `preflight.log` and the plots of its barrel shots. Run `./preflight_models.sh <model>` locally to try again faster.
- **A chain that fails** fails its shard, which still runs its other chains and uploads what it made. The end of the failed chain's log is printed in the shard's log. Its logs (`logs/<chain>.log`, `.simulation.log`, `.reconstruction.log`) are in the shard's `low-cost-parts-<n>` artifact.
- **The report** draws the models whose chains are all there. For the others, it lists the chains missing and fails.
- **Re-run failed jobs** re-runs the failed shards, then the report. A shard re-run takes the chains its earlier attempt made, and runs the rest.
- **A shard near GitHub's 6-hour limit** starts no chain after `github.deadline_hours` (4.5 h) from the start of its job. It stops any still running at `github.stop_hours` (5.5 h), so that what it made is uploaded.
- **Disk.** A runner starts another chain only with `github.min_free_gb` (6 GB) free, unless none is running. Each shard prints its CPUs, memory and disk first.
- **The software cache** is rebuilt when its key changes: the pins, the build scripts or the image. GitHub drops a cache that goes unused for 7 days.

## Running the jobs elsewhere

The jobs are `ddml-bench github` commands. They run wherever the pipeline does, which is how the tests run them (`tests/test_github.py`, on the fakes).

```bash
export FASINATE_CONFIG=configs/github.yaml   # or your own configuration
scripts/ddml-bench github plan --models mymodel -n 100   # writes low_cost_run/plan.json
scripts/ddml-bench github check              # builds what is missing, checks the environment
scripts/ddml-bench github shard 0            # each shard of the plan, anywhere with the same commit
scripts/ddml-bench github report             # low_cost_run/results/
```

## Settings

The `github` section of [`configs/default.yaml`](../configs/default.yaml); change them in `configs/github.yaml`.

| Setting | Default | What it is |
| --- | ---: | --- |
| `parallel` | 2 | chains at once on a runner (4 vCPUs, 16 GB) |
| `speed` | 0.7 | a runner's speed at a chain, against a core of the timed nodes |
| `shard_hours` | 4 | the longest a shard is planned to take; as few shards as keep under it |
| `max_shards` | 10 | at most this many shards (a free account runs 20 jobs at once) |
| `deadline_hours` | 4.5 | a shard starts no chain after this, from the start of its job |
| `stop_hours` | 5.5 | a shard stops its chains still running at this |
| `setup_hours` | 0.25 | each shard's start-up, for the budget |
| `budget_hours` | 40 | a plan of more runner-hours than this is refused |
| `min_free_gb` | 6 | the disk a runner keeps free before it starts another chain |
