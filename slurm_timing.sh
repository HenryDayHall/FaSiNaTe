#!/usr/bin/env bash
# Time every simulator on the inputs of every scenario, on each kind of node
# of the configuration's timing section, as slurm jobs of one task each: the
# timing benchmark, apart from the pipeline's jobs (docs/timing.md). A task
# times one input with one simulator on one kind of node, simulating it as the
# pipeline does (the same ddsim command), on the input's first warmup_events +
# timing.events events (200 by default), and records each event's wall-clock
# and CPU time. By default:
#
#   cpu  every simulator (geant4 and every model in models/), the models'
#        inference on the CPU, on nodes with --constraint=EPYC&7402&512G
#   gpu  every model whose inference can run on the GPU (DDML's embedded
#        Python), the inference on the GPU, on nodes with
#        --constraint=A100-PCIE-80GB&GPUx1
#
# Usage: ./slurm_timing.sh [options] [SIMULATOR ...] [-- SBATCH_OPTION ...]
#
#   SIMULATOR            only these simulators: models in models/, or geant4
#                        (default: each kind of node's own, timing.devices)
#   -s, --scenario NAME  only this scenario's inputs (repeatable; default:
#                        every scenario that is enabled)
#   -d, --device NODE    only this kind of node, e.g. cpu or gpu (repeatable;
#                        default: every kind in timing.devices)
#       --file FILE      only this input of the scenarios (repeatable)
#   -n, --events N       time N events of each input (default: timing.events)
#       --warmup N       after N events that aren't counted (default:
#                        timing.warmup_events)
#   -f, --force          time again what is timed already
#   -c, --config FILE    merge FILE over the configuration, for this and every
#                        job it submits (repeatable)
#       --dry-run        show what is timed and what would be submitted, then
#                        stop
#   -h, --help           show this help
#
# Options after -- go to every sbatch call, e.g. -- --account=myproject; the
# rest come from the configuration: slurm.sbatch_args, then each kind of
# node's partition, constraint, time limit and sbatch_args (timing.devices).
#
# The results go to <paths.data_dir>/timing/<node>/<scenario>/<simulator>/,
# for each input <name>_t<N>.json (the statistics, the node, the software and
# what the inference ran on), .events.csv (each event's times) and .log (ddsim's
# output); ./timing_summary.sh tabulates them. A task is done when its .json
# records the settings it would be timed with. Run this again at any time:
# what is timed is skipped, and a task that a job of an earlier submission is
# still queued or running for isn't submitted again. Each submission gets a
# folder, <paths.data_dir>/timing/slurm/timing_<time>, with the job script,
# the jobs (jobs.tsv) and their slurm logs (logs/).
#
# Run this on a login node, from the repository root. The inputs must be
# there already (./generate_input.sh, or the pipeline, makes them). It works
# out what is done, and checks that the jobs' environment can run the rest, in
# that environment (building DDML first if needed), as slurm_pipeline.sh does.

if [[ ${BASH_SOURCE[0]} != "$0" ]]; then
    echo "slurm_timing.sh exits when something fails, so run it instead of sourcing it" >&2
    return 1
fi
set -euo pipefail

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "${BASH_SOURCE[0]}"; }
for arg in "$@"; do
    case $arg in
        --) break ;;
        -h | --help) usage; exit 0 ;;
    esac
done

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec "${FASINATE_PYTHON:-python3}" "${FASINATE_REPO_DIR:-$here}/scripts/ddml-timing" run --executor slurm "$@"
