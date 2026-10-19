#!/usr/bin/env bash
# Simulate the benchmark scenarios with full Geant4 on a slurm cluster, one
# input per job: what slurm_pipeline.sh --stages geant4 does, which this runs.
# Inputs whose output already exists are left out, so run this again to
# resubmit what failed or timed out.
#
# Usage: ./slurm_simulate_geant4.sh [options] [-- SBATCH_OPTION ...]
#
#   -s, --scenario NAME   simulate this scenario's inputs (repeatable; default:
#                         every scenario)
#   -n, --events N        a test on the first N events of each input
#   -f, --force           simulate again even where the output exists
#   -c, --config FILE     merge FILE over the configuration (repeatable)
#       --dry-run         show what would be submitted, then stop
#   -h, --help            show this help
#
# Options after -- go to sbatch, e.g. -- --time 3-00:00:00; the jobs' defaults
# are in slurm.stages.geant4 in the configuration. Simulating files outside the
# scenarios (--input) and limiting the jobs running at once (--max-running),
# which this did as a job array, are gone: use simulate_geant4.sh --input, and
# slurm's own limits. See slurm_pipeline.sh for the folder each submission gets.

if [[ ${BASH_SOURCE[0]} != "$0" ]]; then
    echo "slurm_simulate_geant4.sh exits when something fails, so run it instead of sourcing it" >&2
    return 1
fi
set -euo pipefail

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "${BASH_SOURCE[0]}"; }
die() { echo "slurm_simulate_geant4: $*" >&2; exit 1; }

args=(--stages geant4)
while (($#)); do
    case $1 in
        -s | --scenario | -n | --events | -c | --config)
            (($# >= 2)) || die "$1 needs a value (see --help)"
            args+=("$1" "$2")
            shift 2
            ;;
        -f | --force) args+=(--force geant4); shift ;;
        --dry-run) args+=(--dry-run); shift ;;
        -i | --input | -m | --max-running)
            die "$1 is gone: see --help"
            ;;
        -h | --help) usage; exit 0 ;;
        --) args+=("$@"); break ;;
        *) die "unknown option or argument '$1' (see --help)" ;;
    esac
done

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec "$here/slurm_pipeline.sh" "${args[@]}"
