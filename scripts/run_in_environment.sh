#!/usr/bin/env bash
# Run a command in the simulation environment: source setup.sh in a clean
# shell, inside the container unless container.enabled is false, then run the
# command there.
#
# Usage: scripts/run_in_environment.sh [--bind DIR]... [--] COMMAND [ARG ...]
#
#   --bind DIR  also mount DIR in the container, at the same path (repeatable)
#   -h, --help  show this help
#
# /cvmfs, the repository, paths.data_dir and the DDML build are mounted at the
# same paths as outside, so paths mean the same in there. DDML is built first
# if there is no build for this configuration (scripts/build_ddml.sh
# --if-needed). The configuration goes in as the FASINATE_* variables.

if [[ ${BASH_SOURCE[0]} != "$0" ]]; then
    echo "run_in_environment.sh starts a new shell, so run it instead of sourcing it" >&2
    return 1
fi
set -euo pipefail

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "${BASH_SOURCE[0]}"; }
die() { echo "run_in_environment: $*" >&2; exit 1; }

extra_dirs=()
while (($#)); do
    case $1 in
        --bind)
            (($# >= 2)) || die "--bind needs a folder"
            extra_dirs+=("$(realpath -m -- "$2")")
            shift 2
            ;;
        -h | --help) usage; exit 0 ;;
        --) shift; break ;;
        -*) die "unknown option '$1' (see --help)" ;;
        *) break ;;
    esac
done
(($#)) || die "give the command to run (see --help)"

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=scripts/load_config.sh
source "${FASINATE_REPO_DIR:-$(dirname "$here")}/scripts/load_config.sh"

# setup.sh in a container needs DDML built for this configuration already
"$FASINATE_REPO_DIR/scripts/build_ddml.sh" --if-needed
# and the detector configurations, which setup.sh needs too (cloned if needed)
"$FASINATE_REPO_DIR/scripts/fetch_detector_configs.sh" --if-needed
# and the patches to the release, which setup.sh applies (built if needed)
"$FASINATE_REPO_DIR/scripts/build_patches.sh" --if-needed
mkdir -p "$FASINATE_PATHS_DATA_DIR"

# A clean shell, as build_ddml.sh uses: the Key4hep setup can only run once per
# shell, and leftovers from this one (Python environments especially) confuse
# it. The configuration goes along, except bash arrays.
env_vars=(HOME="$HOME" USER="${USER:-$(id -un)}"
    PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
for var in LOGNAME TERM; do
    if [[ -n ${!var:-} ]]; then env_vars+=("$var=${!var}"); fi
done
for var in "${!FASINATE_@}"; do
    if [[ $(declare -p "$var") != "declare -a"* ]]; then env_vars+=("$var=${!var}"); fi
done

container=()
if [[ $FASINATE_CONTAINER_ENABLED == true && -z ${APPTAINER_CONTAINER:-}${SINGULARITY_CONTAINER:-} && ! -d /.singularity.d ]]; then
    runtime=$(command -v apptainer || command -v singularity) ||
        die "container.enabled is true, but neither apptainer nor singularity is on PATH"
    if [[ $(basename "$runtime") == apptainer ]]; then prefix=APPTAINER; else prefix=SINGULARITY; fi
    export "${prefix}_TMPDIR=$FASINATE_CONTAINER_TMP_DIR" "${prefix}_CACHEDIR=$FASINATE_CONTAINER_CACHE_DIR"
    mkdir -p "$FASINATE_CONTAINER_TMP_DIR" "$FASINATE_CONTAINER_CACHE_DIR"
    # Mount the folders at the same paths as outside; skip folders inside others.
    IFS=, read -r -a dirs <<<"/cvmfs,$FASINATE_REPO_DIR,$FASINATE_PATHS_DATA_DIR,$FASINATE_DDML_BIND_PATHS,$FASINATE_DETECTOR_CONFIGS_BIND_PATHS,${FASINATE_PATCHES_BIND_PATHS:-}"
    dirs+=("${extra_dirs[@]}")
    binds=
    for dir in "${dirs[@]}"; do
        [[ -n $dir ]] || continue
        for other in "${dirs[@]}"; do
            if [[ -n $other && $dir == "$other"/* ]]; then continue 2; fi
        done
        if [[ ,$binds, != *",$dir,"* ]]; then binds+=${binds:+,}$dir; fi
    done
    container=("$runtime" exec --bind "$binds" "$FASINATE_CONTAINER_IMAGE")
    echo "run_in_environment: in $FASINATE_CONTAINER_IMAGE, mounting $binds" >&2
fi

# setup.sh is sourced at the top level and without arguments, as the Key4hep
# setup expects, before anything here sets variables it could change.
# shellcheck disable=SC2016 # expanded in the clean shell
run='_run_in_environment_command=("$@")
set --
source "$FASINATE_REPO_DIR/setup.sh" || exit 1
exec "${_run_in_environment_command[@]}"'
exec "${container[@]}" /usr/bin/env -i "${env_vars[@]}" bash --noprofile --norc -c "$run" run_in_environment "$@"
