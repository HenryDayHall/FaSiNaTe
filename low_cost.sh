#!/usr/bin/env bash
# Test models on the first events of each input, against the full samples of
# Geant4, its second sample and the other models, as published: only the
# models' tasks of that sample are run, here one after another (or as slurm
# jobs, with --executor slurm), then its plots against the full samples, at a
# part of the cost of the full benchmark. GitHub's runners run it for a model
# pushed to a fork, split into jobs (.github/workflows/low-cost.yml,
# docs/github.md).
#
# Usage: ./low_cost.sh [options] MODEL ... [-- SBATCH_OPTION ...]
#
#   MODEL                a model in models/ to test (one or more)
#   -n, --events N       the first N events of each input of every scenario
#                        (default: low_cost.events in the configuration, a
#                        number for each scenario)
#   -f, --force STAGE    make the outputs of STAGE again (repeatable, or a
#                        list separated by commas)
#       --executor E     local (the default), or slurm: a job per task, and
#                        options after -- go to every sbatch call
#       --any-stack      compare with the published measurements even where
#                        this configuration pins other software than they
#                        were made with
#   -c, --config FILE    merge FILE over the configuration (repeatable)
#       --dry-run        show what is fetched, what each stage would do and
#                        what the sample costs, then stop
#       --no-fetch       fetch nothing: compare with the full measurements
#                        here (e.g. the full benchmark's, on a cluster)
#   -h, --help           show this help
#
# First, the published records' software is checked against the pins of the
# configuration (key4hep.release, ddml.commit and
# detector_configs.ildconfig.commit): a run stops, saying what to pin in
# configs/local.yaml, rather than compare a model simulated with other
# software. Then everything the full benchmark would fetch is fetched, where it
# isn't here: the inputs, the full samples' measurements of Geant4, its second
# sample and the published models, and the timing table. Then, for each model,
# the first N events of each input of each scenario are simulated,
# reconstructed and measured (observables/<model>_n<N>.npz; a scenario with a
# number of its own adds _<scenario><M>), and drawn against the full samples
# in observables/plots/low_cost_n<N>/: every figure, metrics.csv with each
# metric's statistical floor at the model's sample size (what Geant4 itself
# would measure with as many events) and its value projected to the full
# sample's size, the Pareto fronts (if the model is timed), and dashboard.html.
# The full samples' own plots aren't touched. A run prints what the sample
# costs, estimated from the published timing of a model like it
# (low_cost.estimate); the tau pairs cost the most per event.
#
# In a shell that has sourced setup.sh, this runs right there; anywhere else in
# the environment from setup.sh (scripts/run_in_environment.sh), as
# pipeline.sh does.

if [[ ${BASH_SOURCE[0]} != "$0" ]]; then
    echo "low_cost.sh exits when something fails, so run it instead of sourcing it" >&2
    return 1
fi
set -euo pipefail

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "${BASH_SOURCE[0]}"; }
for arg in "$@"; do
    case $arg in -h | --help) usage; exit 0 ;; --) break ;; esac
done

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[[ -f $here/scripts/ddml-bench ]] || here=$PWD
exec "${FASINATE_PYTHON:-python3}" "${FASINATE_REPO_DIR:-$here}/scripts/ddml-bench" low-cost "$@"
