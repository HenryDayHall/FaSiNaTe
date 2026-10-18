#!/usr/bin/env bash
# Simulate the benchmark scenarios with every model described in models/, or
# the ones named, running ddsim with DDML as in DDML's README: with the DDML
# build from scripts/build_ddml.sh, in the environment that setup.sh gives,
# inside the container unless container.enabled is false. This supersedes
# run_sim.sh and Di_Photon_sim_commands.sh for these models.
#
# Usage: ./simulate_models.sh [options] [MODEL ...]
#
#   -s, --scenario NAME  simulate this scenario's inputs (repeatable; default:
#                        every scenario, unless --input is given)
#   -i, --input FILE     simulate FILE, as the scenario "custom" (repeatable)
#   -n, --events N       simulate only the first N events of each input
#   -f, --force          simulate again even where the output exists
#       --dry-run        show what would be simulated and skipped, then stop
#   -h, --help           show this help
#
# Each simulation writes
#   <paths.data_dir>/sim_output/<scenario>/<model>/<input name>.root
# (<input name>_n<N>.root with --events), with its log next to it, and is
# skipped if that output exists. ddsim writes to a hidden file that is only
# renamed once ddsim succeeds, so an output is always complete. Make the
# scenarios' inputs first with ./generate_input.sh.
#
# Run it from the repository root, or submit it from there with sbatch. The
# input names are worked out in the simulation environment, so --dry-run also
# builds DDML first if needed.

#SBATCH --time 2-00:00:00
#SBATCH --nodes 1
#SBATCH --partition maxcpu
#SBATCH --job-name sim_models
#SBATCH --constraint='EPYC&7402'
#SBATCH --mail-type=FAIL

if [[ ${BASH_SOURCE[0]} != "$0" ]]; then
    echo "simulate_models.sh exits when something fails, so run it instead of sourcing it" >&2
    return 1
fi
set -euo pipefail

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "${BASH_SOURCE[0]}"; }
die() { echo "simulate_models: $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# These functions run in a clean shell (bash --noprofile --norc with a minimal
# environment; see the end of the script), inside the container unless
# container.enabled is false, so they only rely on their arguments and the
# FASINATE_* variables.

# simulate_all PLAN DRY_RUN: in the environment from setup.sh, work out which
# simulations are still to do, and run them.
simulate_all() {
    local plan=$1 dry_run=$2 jobs line i=0
    local model scenario input output compact presets events
    local step=(jobs) lines=() simulated=() failed=()

    # Check the environment, and skip what is done already
    if [[ $dry_run == true ]]; then step+=(--dry-run); fi
    jobs=$(python3 "$FASINATE_REPO_DIR/scripts/simulation_jobs.py" "${step[@]}" <<<"$plan") ||
        return 1
    if [[ $dry_run == true ]]; then
        echo "simulate_models: dry run, so nothing was simulated"
        return 0
    fi
    if [[ -z $jobs ]]; then
        echo "simulate_models: nothing to simulate"
        return 0
    fi
    mapfile -t lines <<<"$jobs"

    # One thread, as in the earlier simulation scripts and the paper's timings
    export OMP_NUM_THREADS=1
    # Keep the Python output in step with the rest of the log
    export PYTHONUNBUFFERED=1
    trap 'simulation_interrupted 130' INT
    trap 'simulation_interrupted 143' TERM

    for line in "${lines[@]}"; do
        i=$((i + 1))
        IFS=$'\t' read -r model scenario input output compact presets events <<<"$line"
        if [[ -z $events ]]; then
            echo "simulate_models: can't read the job '$line'" >&2
            return 1
        fi
        echo "simulate_models: [$i/${#lines[@]}] $model on $scenario: $input"
        if simulate_one "$model" "$scenario" "$input" "$output" "$compact" "$presets" "$events"; then
            simulated+=("$output")
        else
            failed+=("$output")
        fi
    done

    echo "simulate_models: simulated ${#simulated[@]} of ${#lines[@]}"
    if ((${#failed[@]})); then
        echo "simulate_models: these failed (each has its log next to it):" >&2
        printf '  %s\n' "${failed[@]}" >&2
        return 1
    fi
}

# simulate_one MODEL SCENARIO INPUT OUTPUT COMPACT PRESETS EVENTS: run ddsim on
# INPUT as the DDML README does, writing to a hidden file that becomes OUTPUT
# once ddsim succeeds, with the log next to OUTPUT.
simulate_one() {
    local model=$1 scenario=$2 input=$3 output=$4 compact=$5 presets=$6 events=$7
    local dir=${output%/*} name=${output##*/}
    local log=${output%.root}.log
    local partial=$dir/.${name%.root}.${HOSTNAME%%.*}.$$.root
    local preset status start why
    local preset_list=() ml_models=()

    IFS=, read -r -a preset_list <<<"$presets"
    for preset in "${preset_list[@]}"; do ml_models+=(--ml-model "$preset"); done
    # The README's ddsim command, with DDML's steering file. The README names
    # ddsim_steer_cc3.py, but that only exists on DDML's henry-all-showers
    # branch, and imports get_presets_from_args, which DDML has since replaced;
    # apart from the particle gun, which an input file overrides, it matches
    # ddsim_steer.py. ddsim runs from DDML's scripts/ folder, as in the README,
    # since presets can give model paths relative to it.
    local cmd=(ddsim --steeringFile "$DDML_SOURCE_DIR/scripts/ddsim_steer.py"
        --compactFile "$compact" "${ml_models[@]}"
        --inputFiles "$input" --outputFile "$partial" --numberOfEvents="$events")

    mkdir -p -- "$dir" || return 1
    {
        echo "# simulate_models.sh: $model on $scenario"
        echo "# input     $input"
        echo "# output    $output, once ddsim succeeds"
        echo "# started   $(date '+%F %T') on $HOSTNAME"
        echo "# Key4hep   $DDML_KEY4HEP_RELEASE"
        echo "# DDML      $DDML_COMMIT, installed in $DDML_INSTALL_DIR"
        printf '# command   (in %s)' "$DDML_SOURCE_DIR/scripts"
        printf ' %q' "${cmd[@]}"
        echo
    } >"$log" || return 1
    echo "simulate_models:   log: $log"

    _sim_partial=$partial _sim_log=$log
    start=$SECONDS
    (cd "$DDML_SOURCE_DIR/scripts" && exec "${cmd[@]}") </dev/null >>"$log" 2>&1
    status=$?
    _sim_partial='' _sim_log=''

    if ((status != 0)); then
        why="ddsim exited with status $status"
    elif [[ ! -s $partial ]]; then
        why="ddsim wrote no output"
    elif ! mv -f -- "$partial" "$output"; then
        why="could not rename $partial to $output"
    else
        echo "# finished  $(date '+%F %T'), after $((SECONDS - start)) s" >>"$log"
        echo "simulate_models:   done in $((SECONDS - start)) s"
        return 0
    fi
    rm -f -- "$partial"
    echo "# FAILED    $(date '+%F %T'): $why" >>"$log"
    echo "simulate_models:   FAILED: $why; the end of the log:" >&2
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
    echo "simulate_models: interrupted" >&2
    exit "$1"
}

# ---------------------------------------------------------------------------

plan_args=()
models=()
input_dirs=()
dry_run=false
while (($#)); do
    case $1 in
        -s | --scenario)
            (($# >= 2)) || die "$1 needs a scenario name"
            plan_args+=(--scenario "$2")
            shift 2
            ;;
        -i | --input)
            (($# >= 2)) || die "$1 needs a file"
            [[ -f $2 ]] || die "input file not found: $2"
            plan_args+=(--input "$2")
            input_dirs+=("$(dirname "$(realpath -- "$2")")")
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
        -h | --help) usage; exit 0 ;;
        -*) die "unknown option '$1' (see --help)" ;;
        *) models+=("$1"); shift ;;
    esac
done

# Under slurm this runs from a copy of the script, so then use the current directory.
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[[ -f $here/scripts/load_config.sh ]] || here=$PWD
# shellcheck source=scripts/load_config.sh
source "${FASINATE_REPO_DIR:-$here}/scripts/load_config.sh"

# Work out the plan before building anything, so mistakes show up early.
plan=$("${FASINATE_PYTHON:-python3}" "$FASINATE_REPO_DIR/scripts/simulation_jobs.py" plan \
    "${plan_args[@]}" -- "${models[@]}") || exit 1

# The DDML build that setup.sh sets up, built for this configuration if needed
"$FASINATE_REPO_DIR/scripts/build_ddml.sh" --if-needed
# and the detector configurations, which setup.sh needs too (cloned if needed)
"$FASINATE_REPO_DIR/scripts/fetch_detector_configs.sh" --if-needed
# and the patches to the release, which setup.sh applies (built if needed)
"$FASINATE_REPO_DIR/scripts/build_patches.sh" --if-needed
mkdir -p "$FASINATE_PATHS_DATA_DIR"

# Run in a clean shell, as build_ddml.sh does: the Key4hep setup can only run
# once per shell, and leftovers from this one (Python environments especially)
# confuse it. The configuration goes along, except bash arrays.
env_vars=(HOME="$HOME" USER="${USER:-$(id -un)}"
    PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
for var in LOGNAME TERM; do
    if [[ -n ${!var:-} ]]; then env_vars+=("$var=${!var}"); fi
done
for var in "${!FASINATE_@}"; do
    if [[ $(declare -p "$var") != "declare -a"* ]]; then env_vars+=("$var=${!var}"); fi
done

container=()
where="this shell"
if [[ $FASINATE_CONTAINER_ENABLED == true && -z ${APPTAINER_CONTAINER:-}${SINGULARITY_CONTAINER:-} && ! -d /.singularity.d ]]; then
    runtime=$(command -v apptainer || command -v singularity) ||
        die "container.enabled is true, but neither apptainer nor singularity is on PATH"
    if [[ $(basename "$runtime") == apptainer ]]; then prefix=APPTAINER; else prefix=SINGULARITY; fi
    export "${prefix}_TMPDIR=$FASINATE_CONTAINER_TMP_DIR" "${prefix}_CACHEDIR=$FASINATE_CONTAINER_CACHE_DIR"
    mkdir -p "$FASINATE_CONTAINER_TMP_DIR" "$FASINATE_CONTAINER_CACHE_DIR"
    # Mount the folders the simulation uses at the same paths as outside, so
    # the paths in the plan hold in there too; skip folders inside others.
    IFS=, read -r -a dirs <<<"/cvmfs,$FASINATE_REPO_DIR,$FASINATE_PATHS_DATA_DIR,$FASINATE_DDML_BIND_PATHS,$FASINATE_DETECTOR_CONFIGS_BIND_PATHS,${FASINATE_PATCHES_BIND_PATHS:-}"
    dirs+=("${input_dirs[@]}")
    binds=
    for dir in "${dirs[@]}"; do
        [[ -n $dir ]] || continue
        for other in "${dirs[@]}"; do
            if [[ -n $other && $dir == "$other"/* ]]; then continue 2; fi
        done
        if [[ ,$binds, != *",$dir,"* ]]; then binds+=${binds:+,}$dir; fi
    done
    container=("$runtime" exec --bind "$binds" "$FASINATE_CONTAINER_IMAGE")
    where="$FASINATE_CONTAINER_IMAGE, mounting $binds"
fi

# In there, get the README's environment from setup.sh: Key4hep with its ML
# libraries, then the DDML build, with DDML's python/ and python/examples on
# PYTHONPATH. It is sourced at the top level and without arguments, as the
# Key4hep setup expects, before anything here sets variables it could change.
# shellcheck disable=SC2016 # expanded in the clean shell
run='sim_plan=$1 sim_dry_run=$2
set --
source "$FASINATE_REPO_DIR/setup.sh" || exit 1
simulate_all "$sim_plan" "$sim_dry_run"'
echo "simulate_models: running in $where"
"${container[@]}" /usr/bin/env -i "${env_vars[@]}" bash --noprofile --norc -c \
    "$(declare -f simulate_all simulate_one simulation_interrupted)
$run" simulate_models "$plan" "$dry_run" ||
    die "simulating failed (see above)"
