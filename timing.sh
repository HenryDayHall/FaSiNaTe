#!/usr/bin/env bash
# Time simulators on the scenarios' inputs here, one task after another: what
# slurm_timing.sh submits as jobs, run on this node instead, e.g. on an
# interactive GPU node with --device gpu, or to try the timing on a few events
# (docs/timing.md). The results go where slurm_timing.sh's go, and record the
# node they were timed on.
#
# Usage: ./timing.sh [options] [SIMULATOR ...]
#
#   SIMULATOR            only these simulators: models in models/, or geant4
#                        (default: the kind of node's own, timing.devices)
#   -s, --scenario NAME  only this scenario's inputs (repeatable)
#   -d, --device NODE    time as this kind of node of timing.devices, e.g. gpu
#                        for the models' inference on this node's GPU
#                        (repeatable; default: cpu)
#       --file FILE      only this input of the scenarios (repeatable)
#   -n, --events N       time N events of each input (default: timing.events)
#       --warmup N       after N events that aren't counted (default:
#                        timing.warmup_events)
#   -f, --force          time again what is timed already
#   -c, --config FILE    merge FILE over the configuration (repeatable)
#       --dry-run        show what is timed and what would be run, then stop
#   -h, --help           show this help
#
# e.g. ./timing.sh -n 20 --warmup 2 -s single_photon caloclouds3
#
# In a shell that has sourced setup.sh, this runs right there; anywhere else,
# in the environment from setup.sh (scripts/run_in_environment.sh), as the
# pipeline's scripts do.

if [[ ${BASH_SOURCE[0]} != "$0" ]]; then
    echo "timing.sh exits when something fails, so run it instead of sourcing it" >&2
    return 1
fi
set -euo pipefail

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "${BASH_SOURCE[0]}"; }
for arg in "$@"; do
    case $arg in -h | --help) usage; exit 0 ;; esac
done

# Under slurm this runs from a copy of the script, so then use the current directory.
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[[ -f $here/scripts/ddml-timing ]] || here=$PWD
exec "${FASINATE_PYTHON:-python3}" "${FASINATE_REPO_DIR:-$here}/scripts/ddml-timing" run --executor local "$@"
