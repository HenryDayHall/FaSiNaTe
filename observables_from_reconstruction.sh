#!/usr/bin/env bash
# Measure the benchmark observables (`observables` in the configuration) on the
# reconstructed events of each simulator whose reconstruction is finished, and
# gather the numbers every plot of them needs into one .npz file per simulator,
# for plot_observables.sh.
#
# Usage: ./observables_from_reconstruction.sh [options] [SIMULATOR ...]
#
#   SIMULATOR          measure only these: models in models/, geant4, or
#                      geant4_2, its second sample (default: all of them)
#   -n, --events N     measure the test sample of the first N events of each
#                      input (<input>_n<N>_REC.edm4hep.root, as
#                      reconstruct_simulation.sh -n N writes), writing
#                      <simulator>_n<N>.npz
#   -f, --force        measure again even where the .npz file is up to date
#   -c, --config FILE  merge FILE over the configuration (repeatable)
#       --dry-run      show what would be measured and what is done, then stop
#       --no-fetch     make everything here, fetching nothing published
#   -h, --help         show this help
#
# A simulator's reconstruction is finished when every input of every scenario
# the observables are measured on has its reconstructed file,
#   <paths.data_dir>/rec_output/<scenario>/<simulator>/<input>_REC.edm4hep.root
# Simulators that aren't finished are skipped, and said so. Each one measured
# writes
#   <paths.data_dir>/observables/<simulator>.npz
# with each profile per input file, as sums over events of each bin's energy
# and of its square, and the reconstructed energy of each event and the
# residuals of its shower's intrinsic angles (ddml_bench/measure.py lists the
# keys). A simulator whose
# .npz file already has every key the configuration asks for, with the same
# settings and from the same reconstructed files, is skipped, whether or not
# those files are still there (or ever were, if the .npz file was fetched):
# the .npz file records them, relative to paths.data_dir.
#
# Reading the EDM4hep files (with uproot) needs the environment from setup.sh:
# in a shell that has sourced it, this runs right there; anywhere else it runs
# in a clean shell that sources setup.sh, inside the container unless
# container.enabled is false. Run it from the repository root, or submit it
# from there with sbatch.

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
for arg in "$@"; do
    case $arg in -h | --help) usage; exit 0 ;; esac
done

# Under slurm this runs from a copy of the script, so then use the current directory.
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[[ -f $here/scripts/ddml-bench ]] || here=$PWD
exec "${FASINATE_PYTHON:-python3}" "${FASINATE_REPO_DIR:-$here}/scripts/ddml-bench" stage observables "$@"
