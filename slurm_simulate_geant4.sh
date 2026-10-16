#!/usr/bin/env bash
# Submit a slurm job array that simulates the benchmark scenarios with full
# Geant4, one input file per job. Each job sources setup.sh (inside the
# container unless container.enabled is false), then runs simulate_geant4.sh
# on its file. Inputs whose output already exists are left out, so run this
# again to resubmit what failed or timed out.
#
# Usage: ./slurm_simulate_geant4.sh [options] [-- SBATCH_OPTION ...]
#
#   -s, --scenario NAME   simulate this scenario's inputs (repeatable; default:
#                         every scenario, unless --input is given)
#   -i, --input FILE      simulate FILE, as the scenario "custom" (repeatable)
#   -n, --events N        simulate only the first N events of each input
#   -f, --force           simulate again even where the output exists
#   -m, --max-running N   run at most N of the jobs at once
#       --dry-run         write the job script and the list of inputs, but
#                         don't submit them
#   -h, --help            show this help
#
# Options after -- go to sbatch, e.g. -- --time 3-00:00:00 --partition allcpu;
# they override the job script's defaults (2 days on maxcpu, as in the other
# scripts).
#
# Each submission gets a folder, <paths.data_dir>/slurm/simulate_geant4_<time>,
# with the inputs (inputs.tsv: job N simulates line N), the job script (job.sh)
# and each job's slurm log (job_<N>.out). ddsim's log goes next to each output,
# as with simulate_geant4.sh. Run this on a login node, from the repository
# root: it lists the inputs in the simulation environment, building DDML first
# if needed, so that the jobs don't have to.

if [[ ${BASH_SOURCE[0]} != "$0" ]]; then
    echo "slurm_simulate_geant4.sh exits when something fails, so run it instead of sourcing it" >&2
    return 1
fi
set -euo pipefail

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "${BASH_SOURCE[0]}"; }
die() { echo "slurm_simulate_geant4: $*" >&2; exit 1; }

list_args=()     # for listing the inputs
job_args=()      # for simulate_geant4.sh in each job
sbatch_args=()
max_running=
dry_run=false
while (($#)); do
    case $1 in
        -s | --scenario)
            (($# >= 2)) || die "$1 needs a scenario name"
            list_args+=(--scenario "$2")
            shift 2
            ;;
        -i | --input)
            (($# >= 2)) || die "$1 needs a file"
            [[ -f $2 ]] || die "input file not found: $2"
            list_args+=(--input "$(realpath -- "$2")")
            shift 2
            ;;
        -n | --events)
            (($# >= 2)) || die "$1 needs a number of events"
            [[ $2 =~ ^[1-9][0-9]*$ ]] || die "$1 needs a whole number of events, not '$2'"
            list_args+=(--events "$2")
            job_args+=(--events "$2")
            shift 2
            ;;
        -f | --force)
            list_args+=(--force)
            job_args+=(--force)
            shift
            ;;
        -m | --max-running)
            (($# >= 2)) || die "$1 needs a number of jobs"
            [[ $2 =~ ^[1-9][0-9]*$ ]] || die "$1 needs a whole number of jobs, not '$2'"
            max_running=$2
            shift 2
            ;;
        --dry-run) dry_run=true; shift ;;
        -h | --help) usage; exit 0 ;;
        --) shift; sbatch_args=("$@"); break ;;
        *) die "unknown option or argument '$1' (see --help)" ;;
    esac
done

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=scripts/load_config.sh
source "${FASINATE_REPO_DIR:-$here}/scripts/load_config.sh"
repo=$FASINATE_REPO_DIR
if ! $dry_run; then
    command -v sbatch >/dev/null || die "sbatch isn't on PATH; run this on a slurm login node"
fi

dir=$FASINATE_PATHS_DATA_DIR/slurm/simulate_geant4_$(date +%Y%m%d_%H%M%S)_$$
mkdir -p -- "$dir"
list=$dir/inputs.tsv

# The inputs still to simulate, worked out in the simulation environment
"$repo/simulate_geant4.sh" "${list_args[@]}" --list "$list" ||
    die "listing the inputs failed (see above)"
count=$(wc -l <"$list")
if ((count == 0)); then
    rm -f -- "$list"
    rmdir -- "$dir"
    echo "slurm_simulate_geant4: nothing to simulate, so nothing was submitted"
    exit 0
fi

# Each job: setup.sh, then simulate_geant4.sh on its line of the list
{
    cat <<'EOF'
#!/usr/bin/env bash
#SBATCH --job-name sim_geant4
#SBATCH --nodes 1
#SBATCH --partition maxcpu
#SBATCH --time 2-00:00:00
#SBATCH --constraint='EPYC&7402'
#SBATCH --mail-type=FAIL
EOF
    echo "#"
    echo "# Written by slurm_simulate_geant4.sh on $(date '+%F %T'). Job N of the array"
    echo "# simulates line N of inputs.tsv with Geant4: it sources setup.sh, inside the"
    echo "# container unless container.enabled is false, then runs simulate_geant4.sh"
    echo "# on that one file."
    echo "set -euo pipefail"
    printf 'repo=%q\n' "$repo"
    printf 'list=%q\n' "$list"
    printf 'job_args=(%s)\n' "$(if ((${#job_args[@]})); then printf '%q ' "${job_args[@]}"; fi)"
    cat <<'EOF'
line=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$list")
IFS=$'\t' read -r scenario input <<<"$line"
if [[ -z $input ]]; then
    echo "no input on line $SLURM_ARRAY_TASK_ID of $list" >&2
    exit 1
fi
echo "job ${SLURM_ARRAY_JOB_ID}_$SLURM_ARRAY_TASK_ID on $(hostname): Geant4 on $scenario, $input"
exec "$repo/scripts/run_in_environment.sh" --bind "$(dirname "$input")" -- \
    "$repo/simulate_geant4.sh" --scenario "$scenario" --file "$input" "${job_args[@]}"
EOF
} >"$dir/job.sh"
chmod +x "$dir/job.sh"

array=1-$count${max_running:+%$max_running}
submit=(sbatch --parsable --array="$array" --output="$dir/job_%a.out" "${sbatch_args[@]}" "$dir/job.sh")
if $dry_run; then
    echo "slurm_simulate_geant4: dry run; this would submit $count job(s), one per line of $list:"
    printf ' %q' "${submit[@]}"
    echo
    exit 0
fi
job_id=$("${submit[@]}") || die "sbatch failed (see above); the job script is $dir/job.sh"
job_id=${job_id%%;*}
echo "slurm_simulate_geant4: submitted job array $job_id, $count job(s), one per line of $list"
echo "slurm_simulate_geant4: slurm logs go to $dir/job_<N>.out; follow them with squeue -j $job_id"
