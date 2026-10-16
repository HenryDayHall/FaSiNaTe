#!/usr/bin/env bash
# Clone or update the detector configuration repositories that reconstruction
# needs and Key4hep releases don't include (ILDConfig, for ILD's standard
# reconstruction), as set in the `detector_configs` section of the
# configuration (configs/default.yaml, then configs/local.yaml and any other
# overrides; see README.md).
#
# Usage: scripts/fetch_detector_configs.sh [options]
#
#   --config FILE   merge FILE over the configuration (repeatable)
#   --set K=V       override one setting, e.g.
#                   --set detector_configs.ildconfig.update=false (repeatable)
#   --if-needed     do nothing for a clone that is already there, on its
#                   branch, or at its commit if one is set
#   --dry-run       print the settings and planned steps without running them
#   -h, --help      show this help
#
# For each entry NAME of detector_configs: clone `repository` (its `branch`)
# to `source_dir` if it isn't there; otherwise, if `update` is true, fetch it
# and fast-forward `branch`. If `commit` is set, check that out instead.
#
# setup.sh and the scripts that start a container call this with --if-needed,
# so an existing clone is left as it is there. Run it yourself to update the clones to the heads of
# their branches.

set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "${BASH_SOURCE[0]}"; }
die() { echo "fetch_detector_configs: $*" >&2; exit 1; }
say() { echo "fetch_detector_configs: $*"; }

config_args=()
if_needed=false
dry_run=false
while (($#)); do
    case $1 in
        --config | --set)
            (($# >= 2)) || die "$1 needs an argument"
            config_args+=("$1" "$2")
            shift 2
            ;;
        --if-needed) if_needed=true; shift ;;
        --dry-run) dry_run=true; shift ;;
        -h | --help) usage; exit 0 ;;
        *) die "unknown option '$1' (see --help)" ;;
    esac
done

settings=$("${FASINATE_PYTHON:-python3}" "$here/fasinate_config.py" --shell "${config_args[@]}") ||
    die "could not load the configuration (see the message above)"
eval "$settings"

# Print a command, and run it unless this is a dry run.
run() {
    printf '+'; printf ' %q' "$@"; printf '\n'
    $dry_run || "$@"
}

normalize_url() { # lower case, no scheme, user or .git; host:path -> host/path
    local url=${1,,}
    url=${url%/}
    url=${url%.git}
    url=${url#*://}
    url=${url#*@}
    echo "${url/://}"
}

# setting NAME KEY: detector_configs.NAME.KEY
setting() {
    local var=FASINATE_DETECTOR_CONFIGS_${1^^}_${2^^}
    echo "${!var:-}"
}

# is_current NAME: the clone is there, from the right repository, and on its
# branch (or at its commit, if one is set)
is_current() {
    local src repo branch commit origin
    src=$(setting "$1" source_dir) repo=$(setting "$1" repository)
    branch=$(setting "$1" branch) commit=$(setting "$1" commit)
    [[ -e $src/.git ]] || return 1
    origin=$(git -C "$src" remote get-url origin 2>/dev/null) || return 1
    [[ $(normalize_url "$origin") == "$(normalize_url "$repo")" ]] || return 1
    if [[ -n $commit ]]; then
        [[ $(git -C "$src" rev-parse HEAD) == "$commit"* ]]
    else
        [[ $(git -C "$src" symbolic-ref --quiet --short HEAD 2>/dev/null) == "$branch" ]]
    fi
}

# fetch NAME: clone or update detector_configs.NAME, as fetch_ddml in
# build_ddml.sh does for DDML
fetch() {
    local name=$1 src repo branch commit update origin
    src=$(setting "$name" source_dir) repo=$(setting "$name" repository)
    branch=$(setting "$name" branch) commit=$(setting "$name" commit)
    update=$(setting "$name" update)
    if [[ ! -e $src/.git ]]; then
        if [[ -e $src && -n $(ls -A "$src") ]]; then
            die "$src exists but is not a git clone; move it or change detector_configs.$name.source_dir"
        fi
        run mkdir -p "$(dirname "$src")"
        run git clone --branch "$branch" "$repo" "$src"
        $dry_run && return
    else
        origin=$(git -C "$src" remote get-url origin) || die "$src has no 'origin' remote"
        [[ $(normalize_url "$origin") == "$(normalize_url "$repo")" ]] ||
            die "$src is a clone of $origin, not $repo; change detector_configs.$name.source_dir or .repository"
        if [[ $update == true ]]; then
            [[ -z $(git -C "$src" status --porcelain --untracked-files=no) ]] ||
                die "$src has uncommitted changes; commit or stash them, or use --set detector_configs.$name.update=false"
            run git -C "$src" fetch origin "$branch"
        fi
    fi
    if [[ -n $commit ]]; then
        if ! git -C "$src" cat-file -e "$commit^{commit}" 2>/dev/null; then
            [[ $update == true ]] ||
                die "commit $commit is not in $src and detector_configs.$name.update is false"
            run git -C "$src" fetch origin "$commit" ||
                die "could not fetch $commit (use the full 40-character hash)"
        fi
        run git -C "$src" -c advice.detachedHead=false checkout --detach "$commit"
    elif [[ $update == true ]]; then
        run git -C "$src" checkout "$branch"
        run git -C "$src" merge --ff-only "origin/$branch" ||
            die "the $branch branch in $src has diverged from origin/$branch"
    fi
}

# ---------------------------------------------------------------------------

# The entries of detector_configs, from the settings they define
names=()
for var in "${!FASINATE_DETECTOR_CONFIGS_@}"; do
    if [[ $var == *_REPOSITORY ]]; then
        name=${var#FASINATE_DETECTOR_CONFIGS_}
        name=${name%_REPOSITORY}
        names+=("${name,,}")
    fi
done
((${#names[@]})) || { say "the configuration has no detector_configs"; exit 0; }

command -v git >/dev/null || die "git is needed"

# One at a time: run scripts submitted together may all get here.
if ! $dry_run && command -v flock >/dev/null; then
    lock_dir=$FASINATE_PATHS_SOFTWARE_DIR
    mkdir -p "$lock_dir"
    exec 9>"$lock_dir/.fetch_detector_configs.lock"
    if ! flock -n 9; then
        say "waiting for another fetch_detector_configs.sh to finish"
        flock 9
    fi
fi

for name in "${names[@]}"; do
    src=$(setting "$name" source_dir)
    if $if_needed && is_current "$name"; then
        say "$name $(git -C "$src" rev-parse HEAD) is already in $src"
        continue
    fi
    commit=$(setting "$name" commit)
    cat <<EOF
$name   $(setting "$name" repository), ${commit:-head of $(setting "$name" branch)}
  sources   $src
EOF
    fetch "$name"
    if $dry_run; then continue; fi
    head=$(git -C "$src" rev-parse HEAD)
    say "$name $head in $src"
    if [[ -z $commit ]]; then
        echo "To use exactly this later, pin it in configs/local.yaml:"
        echo
        printf 'detector_configs:\n  %s:\n    commit: "%s"\n' "$name" "$head"
    fi
done
