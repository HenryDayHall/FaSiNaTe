#!/usr/bin/env bash
# Run the whole benchmark pipeline on a slurm cluster: work out what each stage
# still has to do from what is already on disk, and submit it as slurm jobs of
# one file each, each job waiting for the jobs that make what it reads
# (sbatch --dependency=afterok). The jobs:
#
#   generate     one per input file still to make, with ILD_Particle_Gun.py
#   geant4       one per input still to simulate with Geant4, and one per
#                input for its second sample, geant4_2 (other seeds)
#   models       one per input and model still to simulate
#   reconstruct  one per simulation still to reconstruct
#   observables  one per simulator whose measurements aren't up to date, after
#                all its reconstructions
#   plot         one, after all the measurements, unless the plots are up to
#                date
#
# Usage: ./slurm_pipeline.sh [options] [SIMULATOR ...] [-- SBATCH_OPTION ...]
#
#   SIMULATOR            only these simulators: models in models/, geant4, or
#                        geant4_2, its second sample (default: all of them)
#   -s, --scenario NAME  only this scenario's inputs (repeatable; default:
#                        every scenario)
#       --stages LIST    only these stages, separated by commas (default: all)
#   -n, --events N       a test run on the first N events of each input, with
#                        outputs named _n<N>, as the stage scripts' -n N makes
#   -f, --force STAGE    make the outputs of STAGE, again where they exist and
#                        even where nothing needs them, and everything made
#                        from them (repeatable, or a list separated by commas)
#   -c, --config FILE    merge FILE over the configuration, for this and every
#                        job it submits (repeatable), e.g. --config
#                        configs/pipeline_test.yaml for a very light test run
#       --dry-run        show what each stage has done and what would be
#                        submitted, then stop
#       --no-fetch       make everything here, fetching nothing published
#   -h, --help           show this help
#
# Options after -- go to every sbatch call, e.g. -- --account=myproject; the
# rest come from the slurm section of the configuration (partition,
# constraint and time limit of each stage's jobs).
#
# First, the published inputs and measurements (published in the
# configuration: a Zenodo record) that this run would make the same way, and
# that aren't here yet, are fetched instead (--dry-run says which). Nothing
# here is replaced, and a run says why a published file isn't used.
#
# A task is done when its output exists; the measurements when their .npz file
# is up to date by what it records (the settings, and the reconstructions it
# was made from, which needn't be there any more), and the plots when they are
# up to date with the measurements. A simulation or reconstruction is made
# only if something still to be made needs it, so none is made for
# measurements that are up to date, e.g. fetched with the inputs: adding a
# model then submits only its jobs and the plots'. A task whose stage,
# simulator or scenario isn't selected isn't submitted, and nor is anything
# that needs it. Run this again at any time, e.g. to resubmit what
# failed or timed out: what is done is skipped, and a task that a job of an
# earlier submission is still queued or running for isn't submitted again, but
# waited for. Jobs that wait for a job that fails are cancelled.
#
# Each submission gets a folder, <slurm.log_dir>/pipeline_<time> (in
# paths.data_dir by default), with the plan (plan.json), every task and its
# status (tasks.json), the job script (job.sh), the jobs submitted (jobs.tsv),
# what this printed (report.txt) and each job's slurm log (logs/). Each job
# runs one task, `ddml-bench run-task`, in the environment from setup.sh; the
# tasks write their own logs next to their outputs, as the stage scripts do.
#
# Run this on a login node, from the repository root: it works out what is
# done and checks the environment the jobs will have in that environment
# (inside the container unless container.enabled is false), building DDML
# first if needed, so that the jobs don't have to; --dry-run does too.

if [[ ${BASH_SOURCE[0]} != "$0" ]]; then
    echo "slurm_pipeline.sh exits when something fails, so run it instead of sourcing it" >&2
    return 1
fi
set -euo pipefail

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "${BASH_SOURCE[0]}"; }
for arg in "$@"; do
    case $arg in
        --) break ;;
        -h | --help) usage; exit 0 ;;
    esac
done

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec "${FASINATE_PYTHON:-python3}" "${FASINATE_REPO_DIR:-$here}/scripts/ddml-bench" run --executor slurm "$@"
