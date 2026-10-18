#!/usr/bin/env bash
# Run the whole benchmark pipeline on a slurm cluster: work out what each stage
# still has to do from what is already on disk, and submit it as slurm jobs of
# one file each, each job waiting for the jobs that make what it reads
# (sbatch --dependency=afterok). The jobs:
#
#   generate     one per input file still to make, with ILD_Particle_Gun.py
#   geant4       one per input still to simulate with Geant4 (simulate_geant4.sh)
#   models       one per input and model still to simulate (simulate_models.sh)
#   reconstruct  one per simulation still to reconstruct (reconstruct_simulation.sh)
#   observables  one per simulator whose measurements aren't up to date, after
#                all its reconstructions (observables_from_reconstruction.sh)
#   plot         one, after all the measurements, unless the plots are up to
#                date (plot_observables.sh)
#
# Usage: ./slurm_pipeline.sh [options] [SIMULATOR ...] [-- SBATCH_OPTION ...]
#
#   SIMULATOR            only these simulators: models in models/, or geant4
#                        (default: all of them)
#   -s, --scenario NAME  only this scenario's inputs (repeatable; default:
#                        every scenario)
#       --stages LIST    only these stages, separated by commas (default: all)
#   -n, --events N       a test run on the first N events of each input, with
#                        outputs named _n<N>, as the stage scripts' -n N makes
#   -f, --force STAGE    make the outputs of STAGE again where they exist, and
#                        everything made from them (repeatable, or a list
#                        separated by commas)
#   -c, --config FILE    merge FILE over the configuration, for this and every
#                        job it submits (repeatable), e.g. --config
#                        configs/pipeline_test.yaml for a very light test run
#       --dry-run        show what each stage has done and what would be
#                        submitted, then stop
#   -h, --help           show this help
#
# Options after -- go to every sbatch call, e.g. -- --account=myproject; the
# rest come from the slurm section of the configuration (partition,
# constraint and time limit of each stage's jobs).
#
# A task is done when its output exists; the measurements when they are up to
# date with the reconstructions, and the plots with the measurements. A task
# whose stage, simulator or scenario isn't selected isn't submitted, and nor is
# anything that needs it. Run this again at any time, e.g. to resubmit what
# failed or timed out: what is done is skipped, and a task that a job of an
# earlier submission is still queued or running for isn't submitted again, but
# waited for. Jobs that wait for a job that fails are cancelled.
#
# Each submission gets a folder, <slurm.log_dir>/pipeline_<time> (in
# paths.data_dir by default), with the plan (plan.json), every task and its
# status (tasks.json), the job script (job.sh), the jobs submitted (jobs.tsv),
# what this printed (report.txt) and each job's slurm log (logs/). The stage
# scripts write their own logs next to their outputs, as always.
#
# Run this on a login node, from the repository root: it names the inputs and
# checks the environment the jobs will have in that environment (inside the
# container unless container.enabled is false), building DDML first if needed,
# so that the jobs don't have to; --dry-run does too.

if [[ ${BASH_SOURCE[0]} != "$0" ]]; then
    echo "slurm_pipeline.sh exits when something fails, so run it instead of sourcing it" >&2
    return 1
fi
set -euo pipefail

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "${BASH_SOURCE[0]}"; }
die() { echo "slurm_pipeline: $*" >&2; exit 1; }

plan_args=()
sbatch_args=()
configs=()
dry_run=false
while (($#)); do
    case $1 in
        -s | --scenario | --stages | -f | --force)
            (($# >= 2)) || die "$1 needs a value (see --help)"
            [[ $2 != -* ]] || die "$1 needs a value, not the option '$2' (see --help)"
            case $1 in
                -s | --scenario) plan_args+=(--scenario "$2") ;;
                -f | --force) plan_args+=(--force "$2") ;;
                *) plan_args+=(--stages "$2") ;;
            esac
            shift 2
            ;;
        -n | --events)
            (($# >= 2)) || die "$1 needs a number of events"
            [[ $2 =~ ^[1-9][0-9]*$ ]] || die "$1 needs a whole number of events, not '$2'"
            plan_args+=(--events "$2")
            shift 2
            ;;
        -c | --config)
            (($# >= 2)) || die "$1 needs a configuration file"
            [[ $2 != -* ]] || die "$1 needs a configuration file, not the option '$2'"
            [[ -f $2 ]] || die "configuration file not found: $2"
            configs+=("$(realpath -- "$2")")
            shift 2
            ;;
        --dry-run) dry_run=true; shift ;;
        -h | --help) usage; exit 0 ;;
        --) shift; sbatch_args=("$@"); break ;;
        -*) die "unknown option '$1' (see --help)" ;;
        *) plan_args+=("$1"); shift ;;
    esac
done

# The configuration of this run and of every job: the files in
# FASINATE_CONFIG, then those given with --config, by their full paths, as the
# jobs run elsewhere
config_files=()
IFS=: read -r -a given <<<"${FASINATE_CONFIG:-}"
for file in "${given[@]}"; do
    if [[ -n $file ]]; then config_files+=("$(realpath -m -- "$file")"); fi
done
config_files+=("${configs[@]}")
FASINATE_CONFIG=$(IFS=:; echo "${config_files[*]}")
export FASINATE_CONFIG

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=scripts/load_config.sh
source "${FASINATE_REPO_DIR:-$here}/scripts/load_config.sh"
repo=$FASINATE_REPO_DIR
python=${FASINATE_PYTHON:-python3}
if ! $dry_run; then
    command -v sbatch >/dev/null ||
        die "sbatch isn't on PATH; run this on a slurm login node (--dry-run shows what it would do)"
fi
if ((${#config_files[@]})); then
    echo "slurm_pipeline: with the configuration in ${config_files[*]}"
fi

# The folder of this submission. Submissions take turns, so that two at once
# don't both submit the same jobs.
log_dir=$FASINATE_SLURM_LOG_DIR
mkdir -p -- "$log_dir"
if $dry_run; then
    dir=$(mktemp -d "$log_dir/.dry_run.XXXXXX")
    trap 'rm -rf -- "$dir"' EXIT
else
    if command -v flock >/dev/null; then
        exec 8>"$log_dir/.slurm_pipeline.lock"
        if ! flock -n 8; then
            echo "slurm_pipeline: waiting for another slurm_pipeline.sh to finish submitting"
            flock 8
        fi
    fi
    dir=$log_dir/pipeline_$(date +%Y%m%d_%H%M%S)_$$
    mkdir -p -- "$dir"
fi
mkdir -p -- "$dir/logs"

# 1. The plan, here, where the configuration files are, so that mistakes in
# them or in the options show up before anything is built
"$python" "$repo/scripts/pipeline_jobs.py" plan "${plan_args[@]}" >"$dir/plan.json" ||
    die "can't plan the jobs (see above)"

# 2. What each stage has done and has to do, worked out in the environment the
# jobs run in: it names the inputs, and checks that it can run them
echo "slurm_pipeline: finding what is done, in the environment the jobs have"
"$repo/scripts/run_in_environment.sh" --bind "$dir" -- \
    python3 "$repo/scripts/pipeline_jobs.py" tasks "$dir/plan.json" "$dir/tasks.json" ||
    die "working out the jobs failed (see above)"

# 3. The job script of every job: job.sh STAGE OUTPUT ARGUMENT ... runs the
# stage's script with the arguments, and fails unless OUTPUT is there after
{
    echo "#!/usr/bin/env bash"
    echo "# Written by slurm_pipeline.sh on $(date '+%F %T'). Every job of this submission"
    echo "# runs it, as job.sh STAGE OUTPUT ARGUMENT ...: it runs the script of STAGE"
    echo "# with the ARGUMENTs, on one file, and fails unless that makes OUTPUT. The jobs"
    echo "# are listed in jobs.tsv, and what they run in tasks.json."
    echo "set -euo pipefail"
    printf 'repo=%q\n' "$repo"
    printf 'export FASINATE_CONFIG=%q\n' "$FASINATE_CONFIG"
    if [[ -n ${FASINATE_PYTHON:-} ]]; then printf 'export FASINATE_PYTHON=%q\n' "$FASINATE_PYTHON"; fi
    cat <<'EOF'
# sbatch hands the job the environment it was submitted from; if that had
# sourced setup.sh, the stage scripts would run in it rather than set up their
# own, the one the jobs were checked in
unset DDML_SOURCE_DIR KEY4HEP_STACK
stage=$1 output=$2
shift 2
case $stage in
    generate)
        mkdir -p -- "$(dirname "$output")"
        command=("$repo/scripts/run_in_environment.sh" --bind "$(dirname "$output")" --
            python3 "$repo/scripts/pipeline_jobs.py" generate "$@")
        ;;
    geant4) command=("$repo/simulate_geant4.sh" "$@") ;;
    models) command=("$repo/simulate_models.sh" "$@") ;;
    reconstruct) command=("$repo/reconstruct_simulation.sh" "$@") ;;
    observables) command=("$repo/observables_from_reconstruction.sh" "$@") ;;
    plot) command=("$repo/plot_observables.sh" "$@") ;;
    *) echo "slurm_pipeline: unknown stage '$stage'" >&2; exit 1 ;;
esac
echo "slurm_pipeline: job ${SLURM_JOB_ID:-} on $(hostname), $(date '+%F %T'): $stage, making $output"
printf 'slurm_pipeline: running'; printf ' %q' "${command[@]}"; echo
cd "$repo"
status=0
"${command[@]}" || status=$?
if ((status == 0)) && [[ ! -e $output ]]; then
    echo "slurm_pipeline: $stage finished without making $output" >&2
    status=1
fi
echo "slurm_pipeline: job ${SLURM_JOB_ID:-} finished $(date '+%F %T'), with status $status"
exit "$status"
EOF
} >"$dir/job.sh"
chmod +x "$dir/job.sh"

# 4. Submit, here, where sbatch is
mode=()
if $dry_run; then mode+=(--dry-run); fi
status=0
"$python" "$repo/scripts/pipeline_jobs.py" submit "${mode[@]}" "$dir" -- "${sbatch_args[@]}" ||
    status=$?
# Keep the folder only if it has jobs
if ! $dry_run && [[ ! -e $dir/jobs.tsv ]]; then
    rm -rf -- "$dir"
fi
exit "$status"
