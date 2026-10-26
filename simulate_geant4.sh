#!/usr/bin/env bash
# Simulate the benchmark scenarios with full Geant4, the reference that the
# models are compared to, as simulate_models.sh does for the models: ddsim with
# DDML's steering file from the build by scripts/build_ddml.sh, but without
# DDML's fast simulation (scripts/ddsim_steer_geant4.py), in the environment
# that setup.sh gives.
#
# Usage: ./simulate_geant4.sh [options]
#
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
#   <paths.data_dir>/sim_output/<scenario>/geant4/<input name>.edm4hep.root
# (<input name>_n<N>.edm4hep.root with --events), with its log next to it, and
# is skipped if that output exists, or if nothing still to be made needs it:
# if its reconstruction is there, or the measurements made from it are up to
# date (observables/geant4.npz records what it was made from), e.g. as that
# was fetched with the inputs. ddsim writes to a hidden file that is only
# renamed once ddsim succeeds, so an output is always complete. The inputs are
# simulated one after another; for many, or long, simulations, use
# slurm_pipeline.sh --stages geant4, which runs one input per slurm job. An
# input that isn't made yet is an error: make it with ./generate_input.sh.
#
# In a shell that has sourced setup.sh, this simulates right there. Anywhere
# else it first builds what setup.sh needs, if needed, then runs in a clean
# shell that sources setup.sh, inside the container unless container.enabled is
# false (scripts/run_in_environment.sh). Run it from the repository root, or
# submit it from there with sbatch.

#SBATCH --time 2-00:00:00
#SBATCH --nodes 1
#SBATCH --partition maxcpu
#SBATCH --job-name sim_geant4
#SBATCH --constraint='EPYC&7402'
#SBATCH --mail-type=FAIL

if [[ ${BASH_SOURCE[0]} != "$0" ]]; then
    echo "simulate_geant4.sh exits when something fails, so run it instead of sourcing it" >&2
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
exec "${FASINATE_PYTHON:-python3}" "${FASINATE_REPO_DIR:-$here}/scripts/ddml-bench" stage geant4 "$@"
