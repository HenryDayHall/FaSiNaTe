#!/usr/bin/env bash
# Tabulate the timing results (slurm_timing.sh, timing.sh): for each kind of
# node, scenario, simulator and input, the mean wall-clock time per event over
# the events timed, with its standard error, the speed-up on Geant4, the mean
# CPU time, the models' inference per event, and how many events a 10%
# standard error needs; then each simulator over all of a scenario's inputs.
# It prints the table and writes it to <paths.data_dir>/timing/summary.csv
# (docs/timing.md lists the columns).
#
# Usage: ./timing_summary.sh [options]
#
#   -d, --dir DIR        the results' folder (default: paths.data_dir's
#                        timing/), e.g. copied from elsewhere
#   -o, --output FILE    write the table here instead
#   -n, --events N       only the results of N events timed
#   -c, --config FILE    merge FILE over the configuration (repeatable)
#   -h, --help           show this help
#
# It needs only Python 3 (and PyYAML, to read the configuration without --dir).

if [[ ${BASH_SOURCE[0]} != "$0" ]]; then
    echo "timing_summary.sh exits when something fails, so run it instead of sourcing it" >&2
    return 1
fi
set -euo pipefail

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "${BASH_SOURCE[0]}"; }
for arg in "$@"; do
    case $arg in -h | --help) usage; exit 0 ;; esac
done

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[[ -f $here/scripts/ddml-timing ]] || here=$PWD
exec "${FASINATE_PYTHON:-python3}" "${FASINATE_REPO_DIR:-$here}/scripts/ddml-timing" summary "$@"
