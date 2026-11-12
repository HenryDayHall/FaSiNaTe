#!/usr/bin/env bash
# Plot every observable measured by observables_from_reconstruction.sh, with all
# the simulators that have it on the same axes, as in the shower-profile figures
# of the benchmark paper: Geant4 as a grey filled histogram, each model as a
# line with its statistical uncertainty, and each model's relative deviation
# from Geant4 below. Geant4's second sample (geant4_2: Geant4 again, with other
# seeds) is drawn as a model is, a black dashed line, so its distance from
# Geant4 is Geant4's own. The legend gives each model's distance from Geant4
# (the observable's metric in the configuration). The reconstructed energy gives
# the energy linearity (mu90 against the true energy, fitted with a straight
# line) and resolution (sigma90/mu90, fitted with a + b/sqrt(E) + c/E), from
# the central 90% of the events at each energy. The intrinsic angles give the
# distributions of the residuals, reconstructed - true, of theta and phi of
# the showers' axes, from the PCA of all their hits in the bounding box and of
# the most energetic 4% of those, binned as observables.intrinsic_angle says
# (max_residual, n_bins).
#
# Usage: ./plot_observables.sh [options] [SIMULATOR ...]
#
#   SIMULATOR          plot only these simulators, and Geant4 with its second
#                      sample (default: every measurement found)
#   -n, --events N     plot the measurements of the first N events
#                      (<simulator>_n<N>.npz)
#   -d, --dir DIR      the folder with the .npz files (default: observables in
#                      paths.data_dir)
#   -o, --out DIR      where the figures go (default: <dir>/plots)
#       --format LIST  figure formats, e.g. pdf,png (the default)
#       --reference S  compare the others with the simulator S instead of
#                      Geant4
#       --no-closure   leave out the closure tests
#       --no-dashboard don't write dashboard.html
#       --timing FILE  the timing benchmark's table, for the Pareto fronts
#                      (default: timing/summary.csv in the data folder of the
#                      .npz files, if timing_summary.sh has made it)
#       --no-pareto    leave out the Pareto fronts
#   -h, --help         show this help
#
# Each figure is <observable>_<profile>__<scenario>.<format>, e.g.
# radial_profile_r100__single_photon.pdf, energy_resolution__single_photon.pdf
# or intrinsic_angle_theta_top4__single_photon.pdf;
# the metrics go to metrics.csv next to them, and the fitted parameters of the
# energy figures to fits.csv. A model that is the closure test of another
# (closure in models/<model>.yaml: the same model through another DDML
# backend) is also drawn against that one alone, in closure/<model>/, and
# closure.csv says for each scenario whether the two measured the same,
# number for number. metrics.csv also has each distribution's Wasserstein
# distance from Geant4 (W1), and the energy figures' and the pi0 categories'
# chi^2 (chi2, in their legends too). Where the timing benchmark has timed the
# models (timing_summary.sh), each scenario also has its Pareto fronts on each
# kind of node, pareto_js_<node>__<scenario>.<format> and
# pareto_wasserstein_<node>__<scenario>.<format>: every metric on one axis, a
# marker each, each model's JS or Wasserstein distance (or the chi^2) from
# Geant4, divided by the models' mean, against its time per event, and Geant4
# at its own distance, its second sample's, at its time, with pareto.csv. dashboard.html has
# every figure in one page, with a menu to show or hide each model in all of
# them, or in one by clicking its legend entry. It runs with this shell's
# python3 if that has numpy and matplotlib, and otherwise in the environment
# from setup.sh.

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
            # The timing table beside the .npz files' folder, which the
            # Pareto fronts read unless given --timing
            timing_dir=$(dirname -- "$(realpath -- "$2")")/timing
            if [[ $option == --dir && -d $timing_dir ]]; then bind_args+=(--bind "$timing_dir"); fi
            shift 2
            ;;
        --format)
            (($# >= 2)) || die "--format needs a list of formats"
            args+=(--format "$2")
            shift 2
            ;;
        --reference)
            (($# >= 2)) || die "--reference needs a simulator"
            args+=(--reference "$2")
            shift 2
            ;;
        --timing)
            (($# >= 2)) || die "--timing needs the timing benchmark's table"
            [[ -f $2 ]] || die "there is no file '$2' (timing_summary.sh makes the table)"
            args+=(--timing "$(realpath -- "$2")")
            bind_args+=(--bind "$(dirname -- "$(realpath -- "$2")")")
            shift 2
            ;;
        --no-closure) args+=(--no-closure); shift ;;
        --no-dashboard) args+=(--no-dashboard); shift ;;
        --no-pareto) args+=(--no-pareto); shift ;;
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
