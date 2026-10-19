#!/usr/bin/env bash
# Plot every observable measured by observables_from_reconstruction.sh, with all
# the simulators that have it on the same axes, as in the shower-profile figures
# of the benchmark paper: Geant4 as a grey filled histogram, each model as a
# line with its statistical uncertainty, and each model's relative deviation
# from Geant4 below. The legend gives each model's distance from Geant4 (the
# observable's metric in the configuration).
#
# Usage: ./plot_observables.sh [options] [SIMULATOR ...]
#
#   SIMULATOR          plot only these simulators, and Geant4 (default: every
#                      measurement found)
#   -n, --events N     plot the measurements of the first N events
#                      (<simulator>_n<N>.npz)
#   -d, --dir DIR      the folder with the .npz files (default: observables in
#                      paths.data_dir)
#   -o, --out DIR      where the figures go (default: <dir>/plots)
#       --format LIST  figure formats, e.g. pdf,png (the default)
#   -h, --help         show this help
#
# Each figure is <observable>_<profile>__<scenario>.<format>, e.g.
# radial_profile_r100__single_photon.pdf, and the metrics go to metrics.csv
# next to them. It runs with this shell's python3 if that has numpy and
# matplotlib, and otherwise in the environment from setup.sh.

if [[ ${BASH_SOURCE[0]} != "$0" ]]; then
    echo "plot_observables.sh exits when something fails, so run it instead of sourcing it" >&2
    return 1
fi
set -euo pipefail

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "${BASH_SOURCE[0]}"; }
die() { echo "plot_observables: $*" >&2; exit 1; }

args=()
bind_args=()
while (($#)); do
    case $1 in
        -n | --events)
            (($# >= 2)) || die "$1 needs a number of events"
            [[ $2 =~ ^[1-9][0-9]*$ ]] || die "$1 needs a whole number of events, not '$2'"
            args+=(--events "$2")
            shift 2
            ;;
        -d | --dir | -o | --out)
            (($# >= 2)) || die "$1 needs a folder"
            case $1 in -d | --dir) option=--dir ;; *) option=--out ;; esac
            mkdir -p -- "$2"
            args+=("$option" "$(realpath -- "$2")")
            bind_args+=(--bind "$(realpath -- "$2")")
            shift 2
            ;;
        --format)
            (($# >= 2)) || die "--format needs a list of formats"
            args+=(--format "$2")
            shift 2
            ;;
        -h | --help) usage; exit 0 ;;
        -*) die "unknown option '$1' (see --help)" ;;
        *) args+=("$1"); shift ;;
    esac
done

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
entry=$here/scripts/ddml-bench
python=${FASINATE_PYTHON:-python3}
if "$python" -c 'import numpy, matplotlib' 2>/dev/null; then
    exec "$python" "$entry" plot "${args[@]}"
fi
echo "plot_observables: $python has no numpy or matplotlib, so plotting in the environment from setup.sh" >&2
exec "$here/scripts/run_in_environment.sh" "${bind_args[@]}" -- python3 "$entry" plot "${args[@]}"
