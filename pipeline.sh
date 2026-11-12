#!/usr/bin/env bash
# Run the whole benchmark pipeline here, one task after another: work out what
# each stage still has to do from what is already on disk, and do it, each task
# after the tasks that make what it reads. The tasks, as for slurm_pipeline.sh:
#
#   generate     one per input file still to make, with ILD_Particle_Gun.py
#   geant4       one per input still to simulate with Geant4, and one per
#                input for its second sample, geant4_2 (other seeds)
#   models       one per input and model still to simulate
#   reconstruct  one per simulation still to reconstruct
#   observables  one per simulator whose measurements aren't up to date, after
#                all its reconstructions
#   plot         one, after all the measurements, unless the plots are up to
#                date
#
# Usage: ./pipeline.sh [options] [SIMULATOR ...]
#
#   SIMULATOR            only these simulators: models in models/, geant4, or
#                        geant4_2, its second sample (default: all of them)
#   -s, --scenario NAME  only this scenario's inputs (repeatable; default:
#                        every scenario)
#       --stages LIST    only these stages, separated by commas (default: all)
#   -n, --events N       a test run on the first N events of each input, with
#                        outputs named _n<N>
#   -f, --force STAGE    make the outputs of STAGE, again where they exist and
#                        even where nothing needs them, and everything made
#                        from them (repeatable, or a list separated by commas)
#   -c, --config FILE    merge FILE over the configuration (repeatable), e.g.
#                        --config configs/pipeline_test.yaml for a very light
#                        test run
#       --dry-run        show what each stage has done and what would be run,
#                        then stop
#       --no-fetch       make everything here, fetching nothing published
#   -h, --help           show this help
#
# First, the published inputs and measurements (published in the
# configuration: a Zenodo record) that this run would make the same way, and
# that aren't here yet, are fetched instead (--dry-run says which). Nothing
# here is replaced, and a run says why a published file isn't used.
#
# A task is done when its output exists; the measurements when their .npz file
# is up to date by what it records (the settings, and the reconstructions it
# was made from, which needn't be there any more), and the plots when they are
# up to date with the measurements. A simulation or reconstruction is made
# only if something still to be made needs it, so none is made for
# measurements that are up to date, e.g. fetched with the inputs: adding a
# model then runs only its tasks and the plots. A task that fails stops only
# the tasks that need its output; run this again to carry on. Logs go next to
# the outputs, as the stage scripts write them.
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
