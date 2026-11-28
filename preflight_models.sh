#!/usr/bin/env bash
# Check every model described in models/, or the ones named, before it is
# simulated, with the DDML build from scripts/build_ddml.sh, in the environment
# that setup.sh gives, inside the container unless container.enabled is false.
# Its steps, in order, each only once those before it pass:
#
#   1. models/<model>.yaml is well formed and has every field needed, with
#      ddml.weights and ddml.weights_md5
#   2. the weights are there, and match that md5 sum
#   3. the model's own Python inference code, if it has any (the modules of
#      models/ its DDML configurations run), compiles and conforms to Black
#      (preflight.black in the configuration: its major version)
#   4. that code imports, from models/ as ddsim imports it, with its entry point
#   5. if DDML runs the model in the ECal barrel: DDML has what the model needs,
#      and each particle it gives the model there, shot at 100 GeV from just off
#      the interaction point, (2, 0, 0) mm, once straight up, (0, 1, 0), and
#      once tilted towards the beam, (0, 1, 0.5), deposits energy, from a shower
#      DDML has the model place (preflight.barrel); the hits are drawn
#
# Usage: ./preflight_models.sh [options] [MODEL ...]
#
#   MODEL                check these models (default: all of them)
#   -f, --force          check again, even the models that passed as they are
#   -c, --config FILE    merge FILE over the configuration (repeatable)
#       --dry-run        show which models would be checked, then stop
#   -h, --help           show this help
#
# Everything goes to <paths.data_dir>/preflight_output: summary.txt sums up
# every model, and <model>/ has the model's summary.txt (what each step found,
# and where the full logs are), preflight.log (every step's whole output:
# Black's diff, the import's traceback, the check of DDML), the plots of the
# barrel shots, barrel_<particle>.png, and barrel/ with their simulation and
# ddsim's log. A model that passes is checked again only when its description,
# code or weights, the preflight settings or the pinned software change; the
# simulations of a model (./simulate_models.sh, which runs this first, the
# pipeline and the low-cost mode) wait for it to pass.
#
# Run it from the repository root, or submit it from there with sbatch.

#SBATCH --time 02:00:00
#SBATCH --nodes 1
#SBATCH --partition maxcpu
#SBATCH --job-name preflight
#SBATCH --mail-type=FAIL

if [[ ${BASH_SOURCE[0]} != "$0" ]]; then
    echo "preflight_models.sh exits when something fails, so run it instead of sourcing it" >&2
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
exec "${FASINATE_PYTHON:-python3}" "${FASINATE_REPO_DIR:-$here}/scripts/ddml-bench" stage preflight "$@"
