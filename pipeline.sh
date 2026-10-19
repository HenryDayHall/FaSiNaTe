#!/usr/bin/env bash
# Run the whole benchmark pipeline here, one task after another: work out what
# each stage still has to do from what is already on disk, and do it, each task
# after the tasks that make what it reads. The tasks, as for slurm_pipeline.sh:
#
#   generate     one per input file still to make, with ILD_Particle_Gun.py
#   geant4       one per input still to simulate with Geant4
#   models       one per input and model still to simulate
#   reconstruct  one per simulation still to reconstruct
#   observables  one per simulator whose measurements aren't up to date, after
#                all its reconstructions
#   plot         one, after all the measurements, unless the plots are up to
#                date
#
# Usage: ./pipeline.sh [options] [SIMULATOR ...]
#
#   SIMULATOR            only these simulators: models in models/, or geant4
#                        (default: all of them)
#   -s, --scenario NAME  only this scenario's inputs (repeatable; default:
#                        every scenario)
#       --stages LIST    only these stages, separated by commas (default: all)
#   -n, --events N       a test run on the first N events of each input, with
#                        outputs named _n<N>
#   -f, --force STAGE    make the outputs of STAGE again where they exist, and
#                        everything made from them (repeatable, or a list
#                        separated by commas)
#   -c, --config FILE    merge FILE over the configuration (repeatable), e.g.
#                        --config configs/pipeline_test.yaml for a very light
#                        test run
#       --dry-run        show what each stage has done and what would be run,
#                        then stop
#   -h, --help           show this help
#
# A task is done when its output exists; the measurements when they are up to
# date with the reconstructions, and the plots with the measurements. A task
# that fails stops only the tasks that need its output; run this again to
# carry on. Logs go next to the outputs, as the stage scripts write them.
#
# In a shell that has sourced setup.sh, this runs right there. Anywhere else it
# first builds what setup.sh needs, if needed, then runs in a clean shell that
# sources setup.sh, inside the container unless container.enabled is false
# (scripts/run_in_environment.sh). On a slurm cluster, slurm_pipeline.sh runs
# the same tasks as jobs.

if [[ ${BASH_SOURCE[0]} != "$0" ]]; then
    echo "pipeline.sh exits when something fails, so run it instead of sourcing it" >&2
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
exec "${FASINATE_PYTHON:-python3}" "${FASINATE_REPO_DIR:-$here}/scripts/ddml-bench" run --executor local "$@"
