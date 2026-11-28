#!/usr/bin/env bash
# Simulate the benchmark scenarios with every model described in models/, or
# the ones named, running ddsim with DDML as in DDML's README: with the DDML
# build from scripts/build_ddml.sh, in the environment that setup.sh gives,
# inside the container unless container.enabled is false. A model with DDML
# presets runs with them (--ml-model); one with DDML model configurations
# written out (ddml.model_configs) runs scripts/ddsim_steer_models.py, which
# runs DDML's steering file with those (--ml-config) in place of presets.
#
# Usage: ./simulate_models.sh [options] [MODEL ...]
#
#   MODEL                simulate with these models (default: all of them)
#   -s, --scenario NAME  simulate this scenario's inputs (repeatable; default:
#                        every scenario, unless --input is given)
#   -i, --input FILE     simulate FILE too, as the scenario "custom" (repeatable)
#       --file FILE      simulate only FILE, one of the scenarios' inputs
#                        (repeatable)
#   -n, --events N       a test on the first N events of each input
#   -f, --force          simulate again even where the output exists, and also
#                        where nothing needs it (see below)
#   -c, --config FILE    merge FILE over the configuration (repeatable)
#       --dry-run        show what would be simulated and what is done, then stop
#       --no-fetch       make everything here, fetching nothing published
#   -h, --help           show this help
#
# Each simulation writes
#   <paths.data_dir>/sim_output/<scenario>/<model>/<input name>.root
# (<input name>_n<N>.root with --events), with its log next to it, and is
# skipped if that output exists, or if nothing still to be made needs it: if
# its reconstruction is there, or the measurements made from it are up to date
# (observables/<model>.npz records what it was made from), e.g. as that was
# fetched with the inputs. ddsim writes to a hidden file that is only
# renamed once ddsim succeeds, so an output is always complete. Each model is
# simulated in the detector it was trained for
# (fixed_at_training.detector.compact_file). Before simulating, the
# environment is checked: the DDML libraries, the ddml Python package and the
# plugin of each preset or configuration come from the configured build, each
# gives the model the particles it is for, the weights match their md5 sum,
# and the plugins load those weights. Make the scenarios' inputs first with
# ./generate_input.sh.
#
# Each model's preflight runs first, as ./preflight_models.sh runs it, where it
# hasn't passed as the model is now, and the model's simulations wait for it
# to pass: see <paths.data_dir>/preflight_output/summary.txt. -f forces the
# simulations again, not the preflight.
#
# Run it from the repository root, or submit it from there with sbatch. The
# environment is checked where the models run, so --dry-run also builds DDML
# first if needed.

#SBATCH --time 2-00:00:00
#SBATCH --nodes 1
#SBATCH --partition maxcpu
#SBATCH --job-name sim_models
#SBATCH --constraint='EPYC&7402'
#SBATCH --mail-type=FAIL

if [[ ${BASH_SOURCE[0]} != "$0" ]]; then
    echo "simulate_models.sh exits when something fails, so run it instead of sourcing it" >&2
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
exec "${FASINATE_PYTHON:-python3}" "${FASINATE_REPO_DIR:-$here}/scripts/ddml-bench" stage models "$@"
