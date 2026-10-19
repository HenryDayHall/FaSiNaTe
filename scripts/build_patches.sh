#!/usr/bin/env bash
# Build the patches in the configuration (`patches` in configs/default.yaml):
# packages of the Key4hep release rebuilt from a newer version, for fixes the
# release doesn't have yet. Each is built against the Key4hep release DDML is
# built against, and setup.sh uses it in place of the release's own.
#
# Usage: scripts/build_patches.sh [options]
#
#   --config FILE   merge FILE over the configuration (repeatable)
#   --set K=V       override one setting, e.g.
#                   --set patches.marlintrkprocessors.ref=v02-12-10 (repeatable)
#   --if-needed     do nothing if the patches are built for this configuration
#                   and this Key4hep release
#   --clean         delete the patches' build and install folders first
#   --dry-run       print the settings and planned steps without running them
#   -h, --help      show this help
#
# For each patch with enabled: true, this
#   1. finds the release's own build of the package, by its library in
#      MARLIN_DLL, and its version (from Spack's record of it, or its path);
#   2. if that version is older than fixed_in: clones `repository` at `ref` to
#      <patches.base_dir>/<name>/source, then builds and installs it to
#      <name>/install there, in a clean shell with the Key4hep release from
#      ddml.env_file, inside the container unless container.enabled is false,
#      as scripts/build_ddml.sh builds DDML;
# then writes patches.env_file, which setup.sh sources after setting up
# Key4hep: it puts each patch's library in MARLIN_DLL in place of the
# release's, and says which patches the release doesn't need any more.
#
# A patch whose fix the release already has (its version is fixed_in or later)
# isn't built or used, and setup.sh says that its entry can be deleted from the
# configuration. setup.sh and the scripts that start a container call this
# with --if-needed; DDML has to be built first (scripts/build_ddml.sh).

set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "${BASH_SOURCE[0]}"; }
die() { echo "build_patches: $*" >&2; exit 1; }
say() { echo "build_patches: $*"; }

config_args=()
if_needed=false
clean=false
dry_run=false
while (($#)); do
    case $1 in
        --config | --set)
            (($# >= 2)) || die "$1 needs an argument"
            config_args+=("$1" "$2")
            shift 2
            ;;
        --if-needed) if_needed=true; shift ;;
        --clean) clean=true; shift ;;
        --dry-run) dry_run=true; shift ;;
        -h | --help) usage; exit 0 ;;
        *) die "unknown option '$1' (see --help)" ;;
    esac
done

settings=$("${FASINATE_PYTHON:-python3}" "$here/fasinate_config.py" --shell "${config_args[@]}") ||
    die "could not load the configuration (see the message above)"
eval "$settings"
# shellcheck source=scripts/environment.sh
source "$here/environment.sh"

read -r -a names <<<"${FASINATE_PATCHES_ENABLED:-}"
if ((${#names[@]} == 0)); then
    $if_needed || say "no patches are enabled in the configuration, so there's nothing to build"
    exit 0
fi
base=$FASINATE_PATCHES_BASE_DIR
env_file=$FASINATE_PATCHES_ENV_FILE

# setting NAME KEY: patches.NAME.KEY
setting() {
    local var=${1^^}
    var=FASINATE_PATCHES_${var//[^A-Z0-9_]/_}_${2^^}
    echo "${!var:-}"
}

# Print a command, and run it unless this is a dry run.
run() {
    printf '+'; printf ' %q' "$@"; printf '\n'
    $dry_run || "$@"
}

# ---------------------------------------------------------------------------
# These functions run in a fresh shell with a minimal environment (see
# run_clean), possibly inside the container, so they only rely on the
# FASINATE_* variables and their arguments.

# patch_probe NAME LIBRARY [NAME LIBRARY ...]: with the Key4hep release from
# ddml.env_file (without DDML), print for each patch its name, the release's
# library in MARLIN_DLL and that library's version, separated by tabs.
patch_probe() {
    local find_version='
import json, os, re, sys
args = sys.argv[1:]
entries = [e for e in os.environ.get("MARLIN_DLL", "").split(":") if e]
for name, library in zip(args[0::2], args[1::2]):
    path = next((e for e in entries if os.path.basename(e) == library), "")
    version = ""
    if path:
        real = os.path.realpath(path)
        # Spack records the version with the installed package...
        try:
            with open(os.path.join(os.path.dirname(os.path.dirname(real)), ".spack",
                                   "spec.json")) as stream:
                nodes = json.load(stream)["spec"]["nodes"]
            version = next(str(n["version"]) for n in nodes if n.get("name") == name)
        except Exception:
            # ... and in its folder: .../<name>/<version>-<hash>/lib/<library>
            pattern = r"/%s/([0-9]+(?:\.[0-9]+)*)-[a-z0-9]+/" % re.escape(name)
            match = re.search(pattern, real) or re.search(pattern, path)
            version = match.group(1) if match else ""
    print("%s\t%s\t%s" % (name, path, version))
'
    export FASINATE_ENV_NO_DDML=1
    # shellcheck disable=SC1090
    source "$FASINATE_DDML_ENV_FILE" >&2 || return 1
    python3 -c "$find_version" "$@"
}

# patch_build SOURCE BUILD INSTALL JOBS [CMAKE_ARG ...]: configure, build and
# install a package with the Key4hep release from ddml.env_file (without
# DDML), with ROOT's C++ standard, as Key4hep builds its packages.
patch_build() {
    local src=$1 build=$2 install=$3 jobs=$4 std
    shift 4
    export FASINATE_ENV_NO_DDML=1
    # shellcheck disable=SC1090
    source "$FASINATE_DDML_ENV_FILE" || return 1
    unset FASINATE_ENV_NO_DDML
    std=$(root-config --cflags 2>/dev/null | grep -o 'std=c++[0-9]*' | head -n 1) || std=
    std=${std#std=c++}
    if [[ -z $std ]]; then
        echo "build_patches: couldn't get ROOT's C++ standard from root-config" >&2
        return 1
    fi
    set -ex
    cmake -S "$src" -B "$build" -DCMAKE_INSTALL_PREFIX="$install" -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_STANDARD="$std" "$@"
    cmake --build "$build" -j "$jobs"
    cmake --build "$build" --target install
}

# ---------------------------------------------------------------------------

# Run one of the functions above in `bash --noprofile --norc` with a minimal
# environment (scripts/environment.sh), as build_ddml.sh does.
run_clean() {
    local func=$1
    shift
    fasinate_clean_env http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY \
        SSL_CERT_FILE SSL_CERT_DIR
    # 9>&-: don't hand the lock to anything that might outlive this script
    "${container[@]}" /usr/bin/env -i "${fasinate_env[@]}" \
        bash --noprofile --norc -c "$(declare -f "$func"); $func \"\$@\"" "$func" "$@" 9>&-
}

normalize_url() { # lower case, no scheme, user or .git; host:path -> host/path
    local url=${1,,}
    url=${url%/}
    url=${url%.git}
    url=${url#*://}
    url=${url#*@}
    echo "${url/://}"
}

# fetch_source NAME: clone the patch's repository, or fetch it, and check out
# `ref`, detached.
fetch_source() {
    local name=$1 src commit origin
    local repo ref
    repo=$(setting "$name" repository) ref=$(setting "$name" ref)
    src=$base/$name/source
    if [[ ! -e $src/.git ]]; then
        if [[ -e $src && -n $(ls -A "$src") ]]; then
            die "$src exists but is not a git clone; move it away"
        fi
        run git clone --quiet "$repo" "$src"
        $dry_run && return 0
    else
        origin=$(git -C "$src" remote get-url origin) || die "$src has no 'origin' remote"
        [[ $(normalize_url "$origin") == "$(normalize_url "$repo")" ]] ||
            die "$src is a clone of $origin, not $repo; move it away or change patches.$name.repository"
        [[ -z $(git -C "$src" status --porcelain --untracked-files=no) ]] ||
            die "$src has uncommitted changes; commit or stash them"
        run git -C "$src" fetch --quiet --tags origin || die "could not fetch $repo"
    fi
    if $dry_run; then
        say "  would check out $ref"
        return 0
    fi
    # A tag, a commit, or failing those a branch of origin
    commit=$(git -C "$src" rev-parse --quiet --verify "refs/tags/$ref^{commit}" ||
        git -C "$src" rev-parse --quiet --verify "$ref^{commit}" ||
        git -C "$src" rev-parse --quiet --verify "origin/$ref^{commit}") ||
        die "$repo has no tag, commit or branch '$ref' (patches.$name.ref)"
    run git -C "$src" -c advice.detachedHead=false checkout --quiet --detach "$commit"
}

remove_dir() {
    local dir=$1
    case $dir in
        / | "$HOME" | "$FASINATE_REPO_DIR" | "$base") die "refusing to delete $dir" ;;
    esac
    [[ $dir == "$base"/* ]] || die "refusing to delete $dir, which is outside $base"
    run rm -rf -- "$dir"
}

is_current() {
    [[ -f $env_file ]] && grep -qxF "# patches stamp: $FASINATE_PATCHES_STAMP" "$env_file" &&
        grep -qxF "# Key4hep: $release $platform" "$env_file"
}

# ---------------------------------------------------------------------------

# The Key4hep release the patches are built against: the one DDML is built
# against, as setup.sh sets it up.
if [[ ! -f $FASINATE_DDML_ENV_FILE ]]; then
    msg="DDML isn't built yet, and the patches are built against the Key4hep release it uses; run scripts/build_ddml.sh first"
    $dry_run || die "$msg"
    say "warning: $msg"
fi
release=$(sed -n 's/^_fasinate_k4_release=//p' "$FASINATE_DDML_ENV_FILE" 2>/dev/null) || release=
platform=$(sed -n 's/^_fasinate_k4_platform=//p' "$FASINATE_DDML_ENV_FILE" 2>/dev/null) || platform=

# Only one build at a time: run scripts submitted together may all get here.
if ! $dry_run; then
    mkdir -p "$base"
    if command -v flock >/dev/null; then
        exec 9>"$base/.build_patches.lock"
        if ! flock -n 9; then
            say "waiting for another build_patches.sh to finish"
            flock 9
        fi
    fi
fi

if $if_needed && ! $clean && is_current; then
    exit 0
fi

# Inside an Apptainer/Singularity container, build right there rather than
# starting another container, as build_ddml.sh does. The folders are mounted at
# the same paths as outside, so the paths in the builds and in the env file hold
# in both places.
built_in="a clean shell, without a container"
if [[ $FASINATE_CONTAINER_ENABLED == true ]] && fasinate_in_container; then
    built_in="the container build_patches.sh ran in"
fi
fasinate_container_command /cvmfs "$FASINATE_DDML_BIND_PATHS" "$base" || die "can't start the container"
container=("${fasinate_container[@]}")
if ((${#container[@]})); then built_in=$FASINATE_CONTAINER_IMAGE; fi

echo "Patches     ${names[*]}, in $base"
echo "Key4hep     ${release:-(not set up yet)} ${platform}, from $FASINATE_DDML_ENV_FILE"
echo "Built in    $built_in"
for name in "${names[@]}"; do
    echo "  $name: $(setting "$name" repository) at $(setting "$name" ref)," \
        "while the release's $(setting "$name" library) is older than $(setting "$name" fixed_in)"
done

if $dry_run; then
    say "would set up Key4hep to find each library in MARLIN_DLL and its version"
    for name in "${names[@]}"; do
        say "$name: if the release's version is older than $(setting "$name" fixed_in):"
        fetch_source "$name"
        if $clean; then
            remove_dir "$base/$name/build"
            remove_dir "$base/$name/install"
        fi
        say "  would build and install it to $base/$name/install"
    done
    say "would write $env_file"
    exit 0
fi

# What the release has
probe_args=()
for name in "${names[@]}"; do probe_args+=("$name" "$(setting "$name" library)"); done
say "setting up Key4hep to find the release's builds of the patched packages"
probe=$(run_clean patch_probe "${probe_args[@]}" </dev/null | tail -n "${#names[@]}") ||
    die "setting up Key4hep failed (see above)"

mapfile -t rows <<<"$probe"
((${#rows[@]} == ${#names[@]})) || die "expected ${#names[@]} line(s) from setting up Key4hep, got: $probe"
swaps=() notes=() applied=()
for row in "${rows[@]}"; do
    IFS=$'\t' read -r name path version <<<"$row"
    library=$(setting "$name" library)
    fixed=$(setting "$name" fixed_in)
    ref=$(setting "$name" ref)
    if [[ -z $path ]]; then
        die "$name: the release has no $library in MARLIN_DLL, so there's nothing to replace; check patches.$name.library"
    fi
    if [[ -z $version ]]; then
        say "warning: $name: can't tell the version of $path, so the patch isn't used;" \
            "set patches.$name.enabled to false to stop this warning"
        notes+=("patch $name isn't used: the version of the release's $library is unknown")
        continue
    fi
    if [[ $(printf '%s\n' "$fixed" "$version" | sort -V | head -n 1) == "$fixed" ]]; then
        say "$name: the release has $version, which has the fix (fixed_in $fixed), so the patch isn't needed"
        notes+=("patch $name isn't needed any more: Key4hep $release has $name $version (fixed_in $fixed); delete patches.$name from configs/default.yaml")
        continue
    fi

    say "$name: the release has $version, older than $fixed, so building $ref"
    fetch_source "$name"
    commit=$(git -C "$base/$name/source" rev-parse HEAD)
    if $clean; then
        remove_dir "$base/$name/build"
        remove_dir "$base/$name/install"
    fi
    mkdir -p "$base/$name/build" "$base/$name/install"
    # patches.<name>.cmake_args, if it's there
    extra=()
    arr=${name^^}
    arr=FASINATE_PATCHES_${arr//[^A-Z0-9_]/_}_CMAKE_ARGS
    if declare -p "$arr" >/dev/null 2>&1; then
        declare -n cmake_args=$arr
        extra=("${cmake_args[@]}")
        unset -n cmake_args
    fi
    if ! run_clean patch_build "$base/$name/source" "$base/$name/build" "$base/$name/install" \
        "$FASINATE_DDML_JOBS" "${extra[@]}" </dev/null; then
        die "building $name $ref failed (see above). To go on without this patch, set
  patches:
    $name:
      enabled: false
in configs/local.yaml"
    fi
    built=$(find "$base/$name/install" -name "$library" \( -type f -o -type l \) | head -n 1)
    [[ -n $built ]] || die "building $name installed no $library in $base/$name/install"
    swaps+=("$library"$'\t'"$built"$'\t'"$name $ref (${commit:0:12}) in place of the release's $version, until Key4hep has $fixed")
    applied+=("$name $ref (${commit:0:12})")
done

# The env file, written whole and then moved into place, so it's never half there
tmp=$(mktemp "$env_file.XXXXXX")
{
    echo "# Patches to the Key4hep release for FaSiNaTe, which setup.sh sources after"
    echo "# setting up Key4hep: newer builds of packages of the release, for fixes it"
    echo "# doesn't have yet (patches in the configuration)."
    echo "# Written by scripts/build_patches.sh on $(date -u '+%Y-%m-%d %H:%M UTC'); run it again rather than edit this."
    echo "# patches stamp: $FASINATE_PATCHES_STAMP"
    echo "# Key4hep: $release $platform"
    printf '_fasinate_patches_stamp=%q\n' "$FASINATE_PATCHES_STAMP"
    printf '_fasinate_patches_release=%q\n' "$release"
    printf 'export FASINATE_PATCHES_APPLIED=%q\n' "$(IFS=';'; echo "${applied[*]}")"
    cat <<'EOF'

# Built for this configuration and this Key4hep release?
if [ "${FASINATE_PATCHES_STAMP:-}" != "$_fasinate_patches_stamp" ]; then
    echo "patches: the patches in the configuration changed since they were built; run scripts/build_patches.sh" >&2
    unset _fasinate_patches_stamp _fasinate_patches_release
    return 1
fi
if [ "${DDML_KEY4HEP_RELEASE:-}" != "$_fasinate_patches_release" ]; then
    echo "patches: built against Key4hep $_fasinate_patches_release, but this shell has Key4hep ${DDML_KEY4HEP_RELEASE:-(none)}; run scripts/build_patches.sh" >&2
    unset _fasinate_patches_stamp _fasinate_patches_release
    return 1
fi
unset _fasinate_patches_stamp _fasinate_patches_release

# _fasinate_patch LIBRARY PATH DESCRIPTION: PATH in place of the release's
# LIBRARY in MARLIN_DLL
_fasinate_patch() {
    local entries entry dll="" found=""
    if [ ! -f "$2" ]; then
        echo "patches: $2 is missing; run scripts/build_patches.sh --clean" >&2
        return 1
    fi
    IFS=: read -r -a entries <<<"${MARLIN_DLL:-}"
    for entry in "${entries[@]}"; do
        if [ -n "$entry" ] && [ "${entry##*/}" = "$1" ]; then
            entry=$2
            found=1
        fi
        [ -z "$entry" ] || dll=${dll:+$dll:}$entry
    done
    if [ -z "$found" ]; then
        echo "patches: MARLIN_DLL has no $1 to replace; run scripts/build_patches.sh" >&2
        return 1
    fi
    export MARLIN_DLL=$dll
    echo "FaSiNaTe: patch: $3"
}

EOF
    for swap in "${swaps[@]}"; do
        IFS=$'\t' read -r library built description <<<"$swap"
        printf '_fasinate_patch %q %q %q || { unset -f _fasinate_patch; return 1; }\n' \
            "$library" "$built" "$description"
    done
    for note in "${notes[@]}"; do
        printf 'echo %q\n' "FaSiNaTe: $note"
    done
    echo "unset -f _fasinate_patch"
    echo "return 0"
} >"$tmp"
chmod 0644 "$tmp"
mv -f -- "$tmp" "$env_file"
say "wrote $env_file"
if ((${#applied[@]})); then
    say "setup.sh now uses: ${applied[*]}"
fi
