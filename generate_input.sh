#!/usr/bin/env bash
# Make the input events of the benchmark scenarios, the `scenarios` in the
# configuration (configs/default.yaml, then configs/local.yaml; see README.md),
# under paths.data_dir.
#
# Usage: ./generate_input.sh [SCENARIO ...]
#
# Without arguments it makes the input of every scenario. Each scenario's
# input.type says how; scripts/scenario_inputs.py turns that into commands,
# which run in the environment setup.sh gives (DDML is built first if needed),
# inside the container unless container.enabled is false.
#
# Run it from the repository root (or set FASINATE_REPO_DIR), not with source.

if [[ ${BASH_SOURCE[0]} != "$0" ]]; then
    echo "generate_input.sh exits when something fails, so run it instead of sourcing it" >&2
    return 1
fi
set -euo pipefail

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "${BASH_SOURCE[0]}"; }
die() { echo "generate_input: $*" >&2; exit 1; }

for arg in "$@"; do
    case $arg in
        -h | --help) usage; exit 0 ;;
        -*) die "unknown option '$arg' (see --help)" ;;
    esac
done

# Under slurm this runs from a copy of the script, so then use the current directory.
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[[ -f $here/scripts/load_config.sh ]] || here=$PWD
# shellcheck source=scripts/load_config.sh
source "${FASINATE_REPO_DIR:-$here}/scripts/load_config.sh"

# Work out the commands before building anything, so mistakes show up early.
commands=$("${FASINATE_PYTHON:-python3}" "$FASINATE_REPO_DIR/scripts/scenario_inputs.py" "$@")
[[ -n $commands ]] || die "the configuration has no scenarios"

"$FASINATE_REPO_DIR/scripts/build_ddml.sh" --if-needed
mkdir -p "$FASINATE_PATHS_DATA_DIR"

# Run them in a clean shell, as build_ddml.sh does: the Key4hep setup can only
# run once per shell, and leftovers from this one (Python environments
# especially) confuse it. The configuration goes along, except bash arrays.
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
    # Mount the folders the commands use at the same paths as outside, so the
    # paths in the configuration hold in there too; skip folders inside others.
    IFS=, read -r -a dirs <<<"/cvmfs,$FASINATE_REPO_DIR,$FASINATE_PATHS_DATA_DIR,$FASINATE_DDML_BIND_PATHS"
    binds=
    for dir in "${dirs[@]}"; do
        for other in "${dirs[@]}"; do
            if [[ $dir == "$other"/* ]]; then continue 2; fi
        done
        if [[ ,$binds, != *",$dir,"* ]]; then binds+=${binds:+,}$dir; fi
    done
    container=("$runtime" exec --bind "$binds" "$FASINATE_CONTAINER_IMAGE")
    where="$FASINATE_CONTAINER_IMAGE, mounting $binds"
fi

echo "generate_input: running in $where:"
printf '%s\n' "$commands"
# shellcheck disable=SC2016 # expanded in the clean shell
run='source "$FASINATE_DDML_ENV_FILE" || exit 1
set -e
eval "$1"'
"${container[@]}" /usr/bin/env -i "${env_vars[@]}" bash --noprofile --norc -c "$run" generate_input "$commands" ||
    die "making the inputs failed (see above)"
echo "generate_input: done; the inputs are under $FASINATE_PATHS_DATA_DIR"
