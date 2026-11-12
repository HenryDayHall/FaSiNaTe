#!/usr/bin/env bash
# Reconstruct the simulations of the benchmark scenarios with ILD's standard
# reconstruction, run with Gaudi: ILDConfig's ILDReconstruction.py under k4run,
# as in the Key4hep tutorial (key4hep-doc, tutorials/gaudi_ild_reco), in the
# environment that setup.sh gives.
#
# Usage: ./reconstruct_simulation.sh [options] [SIMULATOR | FILE ...]
#
#   SIMULATOR            reconstruct this simulator's simulations: a model in
#                        models/, geant4, or geant4_2, its second sample
#                        (default: all of them)
#   FILE                 reconstruct this simulation file (a .root or
#                        .edm4hep.root file written by ddsim)
#   -s, --scenario NAME  reconstruct this scenario's simulations (repeatable;
#                        default: every scenario in the configuration; name
#                        custom for what was simulated with --input)
#   -n, --events N       reconstruct the simulations of the first N events of
#                        each input, as simulate_*.sh -n N made them
#       --compact FILE   reconstruct in this detector, instead of the one each
#                        simulation was simulated in (relative to k4geo unless
#                        absolute; it names the detector model too)
#   -f, --force          reconstruct again even where the output exists, and
#                        also where nothing needs it (see below)
#   -c, --config FILE    merge FILE over the configuration (repeatable)
#       --dry-run        show what would be reconstructed and what is done, then
#                        stop
#       --no-fetch       make everything here, fetching nothing published
#   -h, --help           show this help
#
# It reconstructs the simulations of the scenarios' inputs, by the models in
# models/ and by Geant4 (both its samples), each in the detector it was
# simulated in (the model's fixed_at_training.detector.compact_file, or
# geant4.compact). Each simulation
#   <paths.data_dir>/sim_output/<scenario>/<simulator>/<name>.root
# (.edm4hep.root for Geant4) is reconstructed to
#   <paths.data_dir>/rec_output/<scenario>/<simulator>/<name>_REC.edm4hep.root
# with <name>_AIDA.root, <name>_PfoAnalysis.root and the log, <name>.log, next
# to it. With -n N, <name> is <input>_n<N>, the simulation of the first N events
# of each input, reconstructed whole. A file given outside sim_output goes to
# rec_output/custom/files/, and is reconstructed in geant4.compact unless
# --compact says otherwise. A reconstruction is skipped if its _REC file
# exists, or if nothing still to be made needs it: if the measurements made
# from it are up to date (observables/<simulator>.npz records what it was made
# from), e.g. as they were fetched with the inputs; a simulation file given is
# reconstructed all the same. k4run writes to hidden files, only renamed once
# it succeeds, so the outputs are always complete.
#
# ILDConfig isn't part of Key4hep releases; setup.sh clones it to
# detector_configs.ildconfig.source_dir if it isn't there, with
# scripts/fetch_detector_configs.sh (pin detector_configs.ildconfig.commit for
# benchmark runs). Arguments for ILDReconstruction.py go in
# reconstruction.extra_args; --noBeamCalReco is always added, and
# --usingParticleGun for particle_gun scenarios.
#
# In a shell that has sourced setup.sh, this reconstructs right there.
# Anywhere else it first builds what setup.sh needs, if needed, then runs in a
# clean shell that sources setup.sh, inside the container unless
# container.enabled is false (scripts/run_in_environment.sh). Run it from the
# repository root, or submit it from there with sbatch.

#SBATCH --time 2-00:00:00
#SBATCH --nodes 1
#SBATCH --partition maxcpu
#SBATCH --job-name reconstruct
#SBATCH --constraint='EPYC&7402'
#SBATCH --mail-type=FAIL

if [[ ${BASH_SOURCE[0]} != "$0" ]]; then
    echo "reconstruct_simulation.sh exits when something fails, so run it instead of sourcing it" >&2
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
exec "${FASINATE_PYTHON:-python3}" "${FASINATE_REPO_DIR:-$here}/scripts/ddml-bench" stage reconstruct "$@"
