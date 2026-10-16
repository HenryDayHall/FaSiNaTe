#!/usr/bin/env bash
# Simulate the benchmark scenarios with full Geant4, the reference that the
# models are compared to, as simulate_models.sh does for the models: ddsim with
# DDML's steering file from the build by scripts/build_ddml.sh, but without
# DDML's fast simulation (scripts/ddsim_steer_geant4.py), in the environment
# that setup.sh gives.
#
# Usage: ./simulate_geant4.sh [options]
#
#   -s, --scenario NAME  simulate this scenario's inputs (repeatable; default:
#                        every scenario, unless --input is given)
#   -i, --input FILE     simulate FILE, as the scenario "custom" (repeatable)
#       --file FILE      simulate only FILE, an input of the one --scenario
#                        given (or of "custom"): one file at a time, as the
#                        jobs of slurm_simulate_geant4.sh do
#   -n, --events N       simulate only the first N events of each input
#   -f, --force          simulate again even where the output exists
#       --dry-run        show what would be simulated and skipped, then stop
#       --list FILE      write the inputs still to simulate to FILE, one per
#                        line (scenario, a tab, input), then stop
#   -h, --help           show this help
#
# Each simulation writes
#   <paths.data_dir>/sim_output/<scenario>/geant4/<input name>.edm4hep.root
# (<input name>_n<N>.edm4hep.root with --events), with its log next to it, and
# is skipped if that output exists. ddsim writes to a hidden file that is only
# renamed once ddsim succeeds, so an output is always complete. Without
# --file, the inputs are simulated one after another; for many, or long,
# simulations, use slurm_simulate_geant4.sh, which runs one input per job.
#
# In a shell that has sourced setup.sh, this simulates right there. Anywhere
# else it first builds DDML if needed, then runs itself in a clean shell that
# sources setup.sh, inside the container unless container.enabled is false
# (scripts/run_in_environment.sh). Run it from the repository root.

#SBATCH --time 2-00:00:00
#SBATCH --nodes 1
#SBATCH --partition maxcpu
#SBATCH --job-name sim_geant4
#SBATCH --constraint='EPYC&7402'
#SBATCH --mail-type=FAIL

if [[ ${BASH_SOURCE[0]} != "$0" ]]; then
    echo "simulate_geant4.sh exits when something fails, so run it instead of sourcing it" >&2
    return 1
fi
set -euo pipefail

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "${BASH_SOURCE[0]}"; }
die() { echo "simulate_geant4: $*" >&2; exit 1; }

# simulate_one SCENARIO INPUT OUTPUT LOG COMPACT EVENTS: run ddsim on INPUT,
# writing to a hidden file that becomes OUTPUT once ddsim succeeds.
simulate_one() {
    local scenario=$1 input=$2 output=$3 log=$4 compact=$5 events=$6
    local dir=${output%/*} name=${output##*/}
    local partial=$dir/.${HOSTNAME%%.*}.$$.$name
    local status=0 start why
    # The command simulate_models.sh runs, with the Geant4 steering file in
    # place of DDML's (it runs DDML's, leaving out the fast simulation), and from
    # the same folder.
    local cmd=(ddsim --steeringFile "$FASINATE_REPO_DIR/scripts/ddsim_steer_geant4.py"
        --compactFile "$compact"
        --inputFiles "$input" --outputFile "$partial" --numberOfEvents="$events")

    mkdir -p -- "$dir" || return 1
    {
        echo "# simulate_geant4.sh: Geant4 on $scenario"
        echo "# input     $input"
        echo "# output    $output, once ddsim succeeds"
        echo "# started   $(date '+%F %T') on $HOSTNAME"
        echo "# Key4hep   ${DDML_KEY4HEP_RELEASE:-unknown}"
        echo "# DDML      ${DDML_COMMIT:-unknown}, steering from $DDML_SOURCE_DIR/scripts/ddsim_steer.py"
        printf '# command   (in %s)' "$DDML_SOURCE_DIR/scripts"
        printf ' %q' "${cmd[@]}"
        echo
    } >"$log" || return 1
    echo "simulate_geant4:   log: $log"

    _sim_partial=$partial _sim_log=$log
    start=$SECONDS
    (cd "$DDML_SOURCE_DIR/scripts" && exec "${cmd[@]}") </dev/null >>"$log" 2>&1 || status=$?
    _sim_partial='' _sim_log=''

    if ((status != 0)); then
        why="ddsim exited with status $status"
    elif [[ ! -s $partial ]]; then
        why="ddsim wrote no output"
    elif ! mv -f -- "$partial" "$output"; then
        why="could not rename $partial to $output"
    else
        echo "# finished  $(date '+%F %T'), after $((SECONDS - start)) s" >>"$log"
        echo "simulate_geant4:   done in $((SECONDS - start)) s"
        return 0
    fi
    rm -f -- "$partial"
    echo "# FAILED    $(date '+%F %T'): $why" >>"$log"
    echo "simulate_geant4:   FAILED: $why; the end of the log:" >&2
    tail -n 20 -- "$log" | sed 's/^/    /' >&2
    return 1
}

# simulation_interrupted STATUS: the trap for INT and TERM (e.g. from scancel),
# which removes the half-written output.
simulation_interrupted() {
    if [[ -n ${_sim_partial:-} ]]; then
        rm -f -- "$_sim_partial"
        echo "# INTERRUPTED $(date '+%F %T')" >>"$_sim_log"
    fi
    echo "simulate_geant4: interrupted" >&2
    exit "$1"
}

# ---------------------------------------------------------------------------

plan_args=()
bind_args=()
file=
list=
plan=
dry_run=false
while (($#)); do
    case $1 in
        -s | --scenario)
            (($# >= 2)) || die "$1 needs a scenario name"
            plan_args+=(--scenario "$2")
            shift 2
            ;;
        -i | --input | --file)
            (($# >= 2)) || die "$1 needs a file"
            [[ -f $2 ]] || die "input file not found: $2"
            if [[ $1 == --file ]]; then
                [[ -z $file ]] || die "--file can only be given once"
                file=$2
                plan_args+=(--file "$2")
            else
                plan_args+=(--input "$2")
            fi
            bind_args+=(--bind "$(dirname "$(realpath -- "$2")")")
            shift 2
            ;;
        -n | --events)
            (($# >= 2)) || die "$1 needs a number of events"
            [[ $2 =~ ^[1-9][0-9]*$ ]] || die "$1 needs a whole number of events, not '$2'"
            plan_args+=(--events "$2")
            shift 2
            ;;
        -f | --force) plan_args+=(--force); shift ;;
        --dry-run) dry_run=true; shift ;;
        --list)
            (($# >= 2)) || die "--list needs a file to write"
            list=$(realpath -m -- "$2")
            mkdir -p -- "$(dirname "$list")"
            bind_args+=(--bind "$(dirname "$list")")
            shift 2
            ;;
        # The plan that this script works out before it runs itself in the
        # environment from setup.sh
        --plan)
            (($# >= 2)) || die "--plan needs the plan"
            plan=$2
            shift 2
            ;;
        -h | --help) usage; exit 0 ;;
        *) die "unknown option or argument '$1' (see --help)" ;;
    esac
done
mode_args=()
if $dry_run; then mode_args+=(--dry-run); fi
if [[ -n $list ]]; then mode_args+=(--list "$list"); fi

if [[ -z ${DDML_SOURCE_DIR:-} || -z ${KEY4HEP_STACK:-} ]]; then
    # Not in the environment from setup.sh yet. Work out the plan here, where
    # the configuration files are, so that mistakes show up early, then run
    # this script again in that environment.
    # Under slurm this runs from a copy of the script, so then use the current directory.
    here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
    [[ -f $here/scripts/load_config.sh ]] || here=$PWD
    # shellcheck source=scripts/load_config.sh
    source "${FASINATE_REPO_DIR:-$here}/scripts/load_config.sh"
    if [[ -z $plan ]]; then
        plan=$("${FASINATE_PYTHON:-python3}" "$FASINATE_REPO_DIR/scripts/geant4_jobs.py" plan \
            "${plan_args[@]}") || exit 1
    fi
    exec "$FASINATE_REPO_DIR/scripts/run_in_environment.sh" "${bind_args[@]}" -- \
        "$FASINATE_REPO_DIR/simulate_geant4.sh" --plan "$plan" "${mode_args[@]}"
fi

# In the environment from setup.sh, which loaded the configuration too
[[ -n ${FASINATE_REPO_DIR:-} ]] ||
    die "this shell has DDML set up, but not the FaSiNaTe configuration; source setup.sh"
if [[ -z $plan ]]; then
    plan=$(python3 "$FASINATE_REPO_DIR/scripts/geant4_jobs.py" plan "${plan_args[@]}") || exit 1
fi

# Check the environment, and skip what is done already. A list for slurm jobs
# must not include missing inputs, so only a dry run lets those through.
step=(jobs)
if $dry_run; then step+=(--dry-run); fi
jobs=$(python3 "$FASINATE_REPO_DIR/scripts/geant4_jobs.py" "${step[@]}" <<<"$plan") || exit 1
lines=()
if [[ -n $jobs ]]; then mapfile -t lines <<<"$jobs"; fi

if [[ -n $list ]]; then
    tmp=$(mktemp "$list.XXXXXX")
    for line in "${lines[@]}"; do
        IFS=$'\t' read -r scenario input _ <<<"$line"
        printf '%s\t%s\n' "$scenario" "$input"
    done >"$tmp"
    mv -f -- "$tmp" "$list"
    echo "simulate_geant4: listed ${#lines[@]} input(s) to simulate in $list"
    exit 0
fi
if $dry_run; then
    echo "simulate_geant4: dry run, so nothing was simulated"
    exit 0
fi
if ((${#lines[@]} == 0)); then
    echo "simulate_geant4: nothing to simulate"
    exit 0
fi

# One thread, as in the earlier simulation scripts and the paper's timings
export OMP_NUM_THREADS=1
# Keep the Python output in step with the rest of the log
export PYTHONUNBUFFERED=1
trap 'simulation_interrupted 130' INT
trap 'simulation_interrupted 143' TERM

simulated=()
failed=()
i=0
for line in "${lines[@]}"; do
    i=$((i + 1))
    IFS=$'\t' read -r scenario input output log compact events <<<"$line"
    [[ -n $events ]] || die "can't read the job '$line'"
    echo "simulate_geant4: [$i/${#lines[@]}] Geant4 on $scenario: $input"
    if simulate_one "$scenario" "$input" "$output" "$log" "$compact" "$events"; then
        simulated+=("$output")
    else
        failed+=("$output")
    fi
done

echo "simulate_geant4: simulated ${#simulated[@]} of ${#lines[@]}"
if ((${#failed[@]})); then
    echo "simulate_geant4: these failed (each has its log next to it):" >&2
    printf '  %s\n' "${failed[@]}" >&2
    exit 1
fi
