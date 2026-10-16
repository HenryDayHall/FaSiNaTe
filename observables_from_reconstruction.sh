#!/usr/bin/env bash
# Measure the benchmark observables (`observables` in the configuration) on the
# reconstructed events of each simulator whose reconstruction is finished, and
# gather the numbers every plot of them needs into one .npz file per simulator,
# for plot_observables.sh.
#
# Usage: ./observables_from_reconstruction.sh [options] [SIMULATOR ...]
#
#   SIMULATOR        measure only these: models in models/, or geant4
#                    (default: all of them)
#   -n, --events N   use the reconstructions of the first N events of each
#                    input (<input>_n<N>_REC.edm4hep.root, as written by
#                    reconstruct_simulation.sh -n N), writing <simulator>_n<N>.npz
#   -f, --force      measure again even where the .npz file is up to date
#       --dry-run    show what would be measured and skipped, then stop
#   -h, --help       show this help
#
# A simulator's reconstruction is finished when every input of every scenario
# the observables are measured on has its reconstructed file,
#   <paths.data_dir>/rec_output/<scenario>/<simulator>/<input>_REC.edm4hep.root
# Simulators that aren't finished are skipped, and said so. Each one measured
# writes
#   <paths.data_dir>/observables/<simulator>.npz
# with each profile per input file, as sums over events of each bin's energy
# and of its square (scripts/observables.py lists the keys). Before measuring a
# simulator, this checks whether its .npz file already has every key the
# configuration asks for, with the same settings and from the same
# reconstructed files, and skips it if so.
#
# The input file names, and reading the EDM4hep files (with uproot), need the
# environment from setup.sh: in a shell that has sourced it, this runs right
# there; anywhere else it runs itself in a clean shell that sources setup.sh,
# inside the container unless container.enabled is false. Run it from the
# repository root, or submit it from there with sbatch.

#SBATCH --time 1-00:00:00
#SBATCH --nodes 1
#SBATCH --partition maxcpu
#SBATCH --job-name observables
#SBATCH --mail-type=FAIL

if [[ ${BASH_SOURCE[0]} != "$0" ]]; then
    echo "observables_from_reconstruction.sh exits when something fails, so run it instead of sourcing it" >&2
    return 1
fi
set -euo pipefail

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "${BASH_SOURCE[0]}"; }
die() { echo "observables: $*" >&2; exit 1; }

plan_args=()
plan=
dry_run=false
while (($#)); do
    case $1 in
        -n | --events)
            (($# >= 2)) || die "$1 needs a number of events"
            [[ $2 =~ ^[1-9][0-9]*$ ]] || die "$1 needs a whole number of events, not '$2'"
            plan_args+=(--events "$2")
            shift 2
            ;;
        -f | --force) plan_args+=(--force); shift ;;
        --dry-run) dry_run=true; shift ;;
        # The plan that this script works out before it runs itself in the
        # environment from setup.sh
        --plan)
            (($# >= 2)) || die "--plan needs the plan"
            plan=$2
            shift 2
            ;;
        -h | --help) usage; exit 0 ;;
        -*) die "unknown option '$1' (see --help)" ;;
        *) plan_args+=("$1"); shift ;;
    esac
done
mode_args=()
if $dry_run; then mode_args+=(--dry-run); fi

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
        plan=$("${FASINATE_PYTHON:-python3}" "$FASINATE_REPO_DIR/scripts/observables.py" \
            plan "${plan_args[@]}") || exit 1
    fi
    exec "$FASINATE_REPO_DIR/scripts/run_in_environment.sh" -- \
        "$FASINATE_REPO_DIR/observables_from_reconstruction.sh" --plan "$plan" "${mode_args[@]}"
fi

# In the environment from setup.sh, which loaded the configuration too
[[ -n ${FASINATE_REPO_DIR:-} ]] ||
    die "this shell has DDML set up, but not the FaSiNaTe configuration; source setup.sh"
if [[ -z $plan ]]; then
    plan=$(python3 "$FASINATE_REPO_DIR/scripts/observables.py" plan "${plan_args[@]}") || exit 1
fi
export PYTHONUNBUFFERED=1
exec python3 "$FASINATE_REPO_DIR/scripts/observables.py" compute "${mode_args[@]}" <<<"$plan"
