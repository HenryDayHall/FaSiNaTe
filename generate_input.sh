#!/usr/bin/env bash
# Make the input events of the benchmark scenarios, the `scenarios` in the
# configuration (configs/default.yaml, then configs/local.yaml; see README.md),
# under paths.data_dir.
#
# Usage: ./generate_input.sh [options] [SCENARIO ...]
#
#   SCENARIO           make only these scenarios' inputs (default: every
#                      scenario)
#   -f, --force        make the inputs again even where they exist
#   -c, --config FILE  merge FILE over the configuration (repeatable)
#       --dry-run      show what would be made and what is there, then stop
#       --no-fetch     make everything here, fetching nothing published
#   -h, --help         show this help
#
# Each scenario's input.type says how its inputs are made (ddml_bench/scenarios.py);
# a particle_gun input runs ILD_Particle_Gun.py once per energy. Each input is
# made in a hidden folder next to it and moved into place when it is complete,
# and an input that exists is skipped. The commands run in the environment
# setup.sh gives (what it needs is built first if needed), inside the container
# unless container.enabled is false.
#
# Run it from the repository root (or set FASINATE_REPO_DIR), not with source.

if [[ ${BASH_SOURCE[0]} != "$0" ]]; then
    echo "generate_input.sh exits when something fails, so run it instead of sourcing it" >&2
    return 1
fi
set -euo pipefail

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "${BASH_SOURCE[0]}"; }
for arg in "$@"; do
    case $arg in -h | --help) usage; exit 0 ;; esac
done

# Under slurm this runs from a copy of the script, so then use the current directory.
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[[ -f $here/scripts/ddml-bench ]] || here=$PWD
exec "${FASINATE_PYTHON:-python3}" "${FASINATE_REPO_DIR:-$here}/scripts/ddml-bench" stage generate "$@"
