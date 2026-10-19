#!/usr/bin/env bash
# Run a command in the environment the pipeline runs in: source setup.sh in a
# clean shell, inside the container unless container.enabled is false, then
# run the command there.
#
# Usage: scripts/run_in_environment.sh [--bind DIR]... [--] COMMAND [ARG ...]
#
#   --bind DIR  also mount DIR in the container, at the same path (repeatable)
#   -h, --help  show this help
#
# /cvmfs, the repository, paths.data_dir, the DDML build, the detector
# configurations, the patches and the folders of the configuration files in
# FASINATE_CONFIG are mounted at the same paths as outside, so paths mean the
# same in there. What setup.sh needs is built first if it isn't there for this
# configuration (scripts/build_ddml.sh, fetch_detector_configs.sh and
# build_patches.sh, with --if-needed). The configuration goes in as the
# FASINATE_* variables. scripts/environment.sh has the details.

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
# shellcheck source=scripts/environment.sh
source "$FASINATE_REPO_DIR/scripts/environment.sh"

# What setup.sh needs, which it can't build in a container: DDML built for this
# configuration, the detector configurations and the patches to the release.
# None of them read the input, which goes to the command.
"$FASINATE_REPO_DIR/scripts/build_ddml.sh" --if-needed </dev/null
"$FASINATE_REPO_DIR/scripts/fetch_detector_configs.sh" --if-needed </dev/null
"$FASINATE_REPO_DIR/scripts/build_patches.sh" --if-needed </dev/null
mkdir -p "$FASINATE_PATHS_DATA_DIR"

# The folders of the configuration files, which the scripts in there read
config_dirs=()
IFS=: read -r -a config_files <<<"${FASINATE_CONFIG:-}"
for file in "${config_files[@]}"; do
    if [[ -n $file ]]; then config_dirs+=("$(dirname "$(realpath -m -- "$file")")"); fi
done

# shellcheck disable=SC2119 # no variables besides the usual ones
fasinate_clean_env
fasinate_container_command /cvmfs "$FASINATE_REPO_DIR" "$FASINATE_PATHS_DATA_DIR" \
    "$FASINATE_DDML_BIND_PATHS" "$FASINATE_DETECTOR_CONFIGS_BIND_PATHS" \
    "${FASINATE_PATCHES_BIND_PATHS:-}" "${config_dirs[@]}" "${extra_dirs[@]}" || exit 1
if ((${#fasinate_container[@]})); then
    echo "run_in_environment: in $FASINATE_CONTAINER_IMAGE, mounting ${fasinate_container[3]}" >&2
fi

# setup.sh is sourced at the top level and without arguments, as the Key4hep
# setup expects, before anything here sets variables it could change.
# shellcheck disable=SC2016 # expanded in the clean shell
run='_run_in_environment_command=("$@")
set --
source "$FASINATE_REPO_DIR/setup.sh" || exit 1
exec "${_run_in_environment_command[@]}"'
exec "${fasinate_container[@]}" /usr/bin/env -i "${fasinate_env[@]}" \
    bash --noprofile --norc -c "$run" run_in_environment "$@"
