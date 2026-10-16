#!/usr/bin/env bash
# Reconstruct the simulations of the benchmark scenarios with ILD's standard
# reconstruction, run with Gaudi: ILDConfig's ILDReconstruction.py under k4run,
# as in the Key4hep tutorial (key4hep-doc, tutorials/gaudi_ild_reco), in the
# environment that setup.sh gives. This replaces the Marlin reconstruction of
# run_reco.sh and Di_Photon_rec_commands.sh.
#
# Usage: ./reconstruct_simulation.sh [options] [SIMULATOR | FILE ...]
#
#   SIMULATOR            reconstruct this simulator's simulations: a model in
#                        models/, or geant4 (default: all of them)
#   FILE                 reconstruct this simulation file (a .root or
#                        .edm4hep.root file written by ddsim)
#   -s, --scenario NAME  reconstruct this scenario's simulations (repeatable;
#                        default: every scenario in the configuration; name
#                        custom for what was simulated with --input)
#   -n, --events N       reconstruct only the first N events of each
#       --compact FILE   reconstruct in this detector, instead of the one each
#                        simulation was simulated in (relative to k4geo unless
#                        absolute; it names the detector model too)
#   -f, --force          reconstruct again even where the output exists
#       --dry-run        show what would be reconstructed and skipped, then stop
#   -h, --help           show this help
#
# It reconstructs every finished simulation it finds of the scenarios in the
# configuration by the models in models/ and by Geant4, the outputs of
# simulate_models.sh and simulate_geant4.sh, each in the detector it was
# simulated in (the model's fixed_at_training.detector.compact_file, or
# geant4.compact). Each simulation
#   <paths.data_dir>/sim_output/<scenario>/<simulator>/<name>.root
# (.edm4hep.root for Geant4) is reconstructed to
#   <paths.data_dir>/rec_output/<scenario>/<simulator>/<name>_REC.edm4hep.root
# with <name>_AIDA.root, <name>_PfoAnalysis.root and the log, <name>.log, next
# to it (<name>_n<N>... with --events). A file given outside sim_output goes to
# rec_output/custom/files/, and is reconstructed in geant4.compact unless
# --compact says otherwise. A reconstruction is skipped if its _REC file
# exists. k4run writes to hidden files, only renamed once it succeeds, so the
# outputs are always complete.
#
# ILDConfig isn't part of Key4hep releases, so this clones it to
# reconstruction.ildconfig.source_dir first (pin reconstruction.ildconfig.commit
# for benchmark runs). Arguments for ILDReconstruction.py go in
# reconstruction.extra_args; --noBeamCalReco is always added, and
# --usingParticleGun for particle_gun scenarios.
#
# In a shell that has sourced setup.sh, this reconstructs right there.
# Anywhere else it first builds DDML if needed, then runs itself in a clean
# shell that sources setup.sh, inside the container unless container.enabled is
# false (scripts/run_in_environment.sh). Run it from the repository root, or
# submit it from there with sbatch.

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
die() { echo "reconstruct_simulation: $*" >&2; exit 1; }

# ensure_ildconfig: clone ILDConfig to reconstruction.ildconfig.source_dir if
# it isn't there, and check out reconstruction.ildconfig.commit if that is set.
# An existing clone is otherwise left as it is, so results don't change behind
# your back; update it with git yourself.
ensure_ildconfig() {
    local dir=$FASINATE_RECONSTRUCTION_ILDCONFIG_SOURCE_DIR
    local repository=$FASINATE_RECONSTRUCTION_ILDCONFIG_REPOSITORY
    local branch=$FASINATE_RECONSTRUCTION_ILDCONFIG_BRANCH
    local commit=${FASINATE_RECONSTRUCTION_ILDCONFIG_COMMIT:-}
    local tmp

    command -v git >/dev/null || die "git is needed to get ILDConfig"
    if [[ ! -e $dir/.git ]]; then
        [[ ! -e $dir ]] || die "$dir exists but isn't a git clone of ILDConfig; move it away, or point reconstruction.ildconfig.source_dir somewhere else"
        echo "reconstruct_simulation: cloning ILDConfig ($branch) to $dir"
        mkdir -p -- "$(dirname "$dir")"
        # Into a temporary folder first, so that a half-done clone is never used
        tmp=$(mktemp -d "$dir.clone.XXXXXX")
        if ! git clone --quiet --branch "$branch" -- "$repository" "$tmp"; then
            rm -rf -- "$tmp"
            die "cloning $repository failed"
        fi
        if ! mv -T -- "$tmp" "$dir" 2>/dev/null; then
            rm -rf -- "$tmp"
            [[ -e $dir/.git ]] || die "could not move the ILDConfig clone to $dir"
        fi
    fi
    if [[ -n $commit && $(git -C "$dir" rev-parse HEAD) != "$commit"* ]]; then
        echo "reconstruct_simulation: checking out ILDConfig $commit (reconstruction.ildconfig.commit)"
        git -C "$dir" cat-file -e "$commit^{commit}" 2>/dev/null ||
            git -C "$dir" fetch --quiet origin "$commit" ||
            git -C "$dir" fetch --quiet origin ||
            die "could not fetch ILDConfig $commit"
        git -C "$dir" checkout --quiet --detach "$commit" ||
            die "could not check out ILDConfig $commit in $dir (local changes?)"
    fi
    echo "reconstruct_simulation: ILDConfig $(git -C "$dir" rev-parse HEAD) in $dir"
}

# reconstruct_one SCENARIO SIMULATOR INPUT BASE COMPACT EVENTS DONE ARGS: run
# ILDReconstruction.py on INPUT in a folder of its own, writing hidden files
# that become BASE_REC.edm4hep.root and so on once k4run succeeds, with the log
# in BASE.log. DONE is the output that marks it done; ARGS are more arguments
# for ILDReconstruction.py, shell-quoted.
reconstruct_one() {
    local scenario=$1 simulator=$2 input=$3 base=$4 compact=$5 events=$6 done_file=$7 args=$8
    local dir=${base%/*} name=${base##*/}
    local partial=$dir/.$name.${HOSTNAME%%.*}.$$
    local work=$partial.work log=$base.log
    local done_suffix=${done_file#"$base"}
    local status=0 start why entry file extra=()
    eval "extra=($args)"
    local cmd=(k4run "$work/ILDReconstruction.py"
        --inputFiles="$input" --outputFileBase="$partial" --compactFile="$compact"
        --num-events="$events" "${extra[@]}")

    mkdir -p -- "$dir" || return 1
    {
        echo "# reconstruct_simulation.sh: $simulator on $scenario"
        echo "# input     $input"
        echo "# output    ${base}_REC.edm4hep.root and the rest, once k4run succeeds"
        echo "# started   $(date '+%F %T') on $HOSTNAME"
        echo "# Key4hep   ${DDML_KEY4HEP_RELEASE:-unknown}"
        echo "# ILDConfig $ildconfig_commit, in $ildconfig_dir"
        printf '# command   (in a folder linking to %s)' "$ildconfig_dir"
        printf ' %q' "${cmd[@]}"
        echo
    } >"$log" || return 1
    echo "reconstruct_simulation:   log: $log"

    # ILDReconstruction.py finds its configuration relative to the folder it
    # runs in, so run it in a folder of links to ILDConfig's, which also keeps
    # anything it writes there apart from other reconstructions.
    _rec_partial=$partial _rec_work=$work _rec_log=$log
    rm -rf -- "$work"
    mkdir -p -- "$work" || return 1
    for entry in "$ildconfig_dir"/*; do
        ln -s -- "$entry" "$work/" || return 1
    done
    start=$SECONDS
    (cd "$work" && exec "${cmd[@]}") </dev/null >>"$log" 2>&1 || status=$?
    rm -rf -- "$work"

    if ((status != 0)); then
        why="k4run exited with status $status"
    elif [[ ! -s $partial$done_suffix ]]; then
        why="k4run wrote no ${done_suffix#_}"
    else
        # The outputs, keeping the one that marks it done for last
        why=
        for file in "$partial"_*; do
            [[ $file == "$partial$done_suffix" ]] && continue
            mv -f -- "$file" "$base${file#"$partial"}" || why="could not rename $file"
        done
        if [[ -z $why ]] && ! mv -f -- "$partial$done_suffix" "$done_file"; then
            why="could not rename $partial$done_suffix to $done_file"
        fi
        if [[ -z $why ]]; then
            _rec_partial='' _rec_work='' _rec_log=''
            echo "# finished  $(date '+%F %T'), after $((SECONDS - start)) s" >>"$log"
            echo "reconstruct_simulation:   done in $((SECONDS - start)) s"
            return 0
        fi
    fi
    _rec_partial='' _rec_work='' _rec_log=''
    rm -f -- "$partial"_*
    echo "# FAILED    $(date '+%F %T'): $why" >>"$log"
    echo "reconstruct_simulation:   FAILED: $why; the end of the log:" >&2
    tail -n 20 -- "$log" | sed 's/^/    /' >&2
    return 1
}

# reconstruction_interrupted STATUS: the trap for INT and TERM (e.g. from
# scancel), which removes the half-written outputs.
reconstruction_interrupted() {
    if [[ -n ${_rec_partial:-} ]]; then
        rm -rf -- "$_rec_work"
        rm -f -- "$_rec_partial"_*
        echo "# INTERRUPTED $(date '+%F %T')" >>"$_rec_log"
    fi
    echo "reconstruct_simulation: interrupted" >&2
    exit "$1"
}

# ---------------------------------------------------------------------------

plan_args=()
bind_args=()
plan=
dry_run=false
while (($#)); do
    case $1 in
        -s | --scenario)
            (($# >= 2)) || die "$1 needs a scenario name"
            plan_args+=(--scenario "$2")
            shift 2
            ;;
        -n | --events)
            (($# >= 2)) || die "$1 needs a number of events"
            [[ $2 =~ ^[1-9][0-9]*$ ]] || die "$1 needs a whole number of events, not '$2'"
            plan_args+=(--events "$2")
            shift 2
            ;;
        --compact)
            (($# >= 2)) || die "--compact needs a compact file"
            if [[ $2 == /* ]]; then
                [[ -f $2 ]] || die "compact file not found: $2"
                bind_args+=(--bind "$(dirname "$(realpath -- "$2")")")
            fi
            plan_args+=(--compact "$2")
            shift 2
            ;;
        -f | --force) plan_args+=(--force); shift ;;
        --dry-run) dry_run=true; shift ;;
        # The plan that this script works out before it runs itself in the
        # environment from setup.sh
        --plan)
            (($# >= 2)) || die "--plan needs the plan"
            plan=$2
            shift 2
            ;;
        -h | --help) usage; exit 0 ;;
        -*) die "unknown option '$1' (see --help)" ;;
        *)
            if [[ -f $1 ]]; then
                bind_args+=(--bind "$(dirname "$(realpath -- "$1")")")
            fi
            plan_args+=("$1")
            shift
            ;;
    esac
done

if [[ -z ${DDML_SOURCE_DIR:-} || -z ${KEY4HEP_STACK:-} ]]; then
    # Not in the environment from setup.sh yet. Work out the plan here, where
    # the configuration files are, so that mistakes show up early, and get
    # ILDConfig, then run this script again in that environment.
    # Under slurm this runs from a copy of the script, so then use the current directory.
    here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
    [[ -f $here/scripts/load_config.sh ]] || here=$PWD
    # shellcheck source=scripts/load_config.sh
    source "${FASINATE_REPO_DIR:-$here}/scripts/load_config.sh"
    if [[ -z $plan ]]; then
        plan=$("${FASINATE_PYTHON:-python3}" "$FASINATE_REPO_DIR/scripts/reconstruction_jobs.py" \
            plan "${plan_args[@]}") || exit 1
    fi
    ensure_ildconfig
    mode_args=()
    if $dry_run; then mode_args+=(--dry-run); fi
    exec "$FASINATE_REPO_DIR/scripts/run_in_environment.sh" "${bind_args[@]}" \
        --bind "$FASINATE_RECONSTRUCTION_ILDCONFIG_SOURCE_DIR" -- \
        "$FASINATE_REPO_DIR/reconstruct_simulation.sh" --plan "$plan" "${mode_args[@]}"
fi

# In the environment from setup.sh, which loaded the configuration too
[[ -n ${FASINATE_REPO_DIR:-} ]] ||
    die "this shell has DDML set up, but not the FaSiNaTe configuration; source setup.sh"
if [[ -z $plan ]]; then
    plan=$(python3 "$FASINATE_REPO_DIR/scripts/reconstruction_jobs.py" plan "${plan_args[@]}") ||
        exit 1
    ensure_ildconfig
fi

# Check the environment, and skip what is done already
step=(jobs)
if $dry_run; then step+=(--dry-run); fi
jobs=$(python3 "$FASINATE_REPO_DIR/scripts/reconstruction_jobs.py" "${step[@]}" <<<"$plan") || exit 1
lines=()
if [[ -n $jobs ]]; then mapfile -t lines <<<"$jobs"; fi

if $dry_run; then
    echo "reconstruct_simulation: dry run, so nothing was reconstructed"
    exit 0
fi
if ((${#lines[@]} == 0)); then
    echo "reconstruct_simulation: nothing to reconstruct"
    exit 0
fi

ildconfig_dir=$(python3 -c 'import json, sys; print(json.load(sys.stdin)["ildconfig_dir"])' <<<"$plan")
ildconfig_commit=$(git -C "$ildconfig_dir" rev-parse HEAD 2>/dev/null || echo unknown)

# One thread, as in the simulation
export OMP_NUM_THREADS=1
# Keep the Python output in step with the rest of the log
export PYTHONUNBUFFERED=1
trap 'reconstruction_interrupted 130' INT
trap 'reconstruction_interrupted 143' TERM

reconstructed=()
failed=()
i=0
for line in "${lines[@]}"; do
    i=$((i + 1))
    IFS=$'\t' read -r scenario simulator input base compact events done_file args <<<"$line"
    [[ -n $done_file ]] || die "can't read the job '$line'"
    echo "reconstruct_simulation: [$i/${#lines[@]}] $simulator on $scenario: $input"
    if reconstruct_one "$scenario" "$simulator" "$input" "$base" "$compact" "$events" \
        "$done_file" "${args:-}"; then
        reconstructed+=("$done_file")
    else
        failed+=("$done_file")
    fi
done

echo "reconstruct_simulation: reconstructed ${#reconstructed[@]} of ${#lines[@]}"
if ((${#failed[@]})); then
    echo "reconstruct_simulation: these failed (each has its log next to it):" >&2
    printf '  %s\n' "${failed[@]}" >&2
    exit 1
fi
