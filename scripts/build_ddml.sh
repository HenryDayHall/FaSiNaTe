#!/usr/bin/env bash
# Clone DDML (the python-inference branch by default) and build it against a
# Key4hep release, as set in the configuration (configs/default.yaml, then
# configs/local.yaml and any other overrides; see README.md).
#
# Usage: scripts/build_ddml.sh [options]
#
#   --config FILE   merge FILE over the configuration (repeatable)
#   --set K=V       override one setting, e.g. --set ddml.jobs=8 (repeatable)
#   --clean         delete the build and install directories first
#   --dry-run       print the settings and planned steps without running them
#   -h, --help      show this help
#
# Steps: clone or update DDML in ddml.source_dir, pull the container image if
# it is missing, write ddml.env_file, build and install DDML in a clean shell
# (inside the container unless container.enabled is false), then check that
# the env file loads this DDML. Re-run it after changing the DDML commit or
# the Key4hep release.

set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "${BASH_SOURCE[0]}"; }
die() { echo "build_ddml: $*" >&2; exit 1; }
say() { echo "build_ddml: $*"; }

config_args=()
clean=false
dry_run=false
while (($#)); do
    case $1 in
        --config | --set)
            (($# >= 2)) || die "$1 needs an argument"
            config_args+=("$1" "$2")
            shift 2
            ;;
        --clean) clean=true; shift ;;
        --dry-run) dry_run=true; shift ;;
        -h | --help) usage; exit 0 ;;
        *) die "unknown option '$1' (see --help)" ;;
    esac
done

settings=$("${FASINATE_PYTHON:-python3}" "$here/fasinate_config.py" --shell "${config_args[@]}") ||
    die "could not load the configuration (see the message above)"
eval "$settings"

src=$FASINATE_DDML_SOURCE_DIR
build=$FASINATE_DDML_BUILD_DIR
install=$FASINATE_DDML_INSTALL_DIR
env_file=$FASINATE_DDML_ENV_FILE
setup=$FASINATE_KEY4HEP_SETUP_SCRIPT
release=$FASINATE_KEY4HEP_RELEASE

# Print a command, and run it unless this is a dry run.
run() {
    printf '+'; printf ' %q' "$@"; printf '\n'
    $dry_run || "$@"
}

# ---------------------------------------------------------------------------
# These functions run in a fresh shell with a minimal environment (see
# run_clean), possibly inside the container, so they only rely on the
# FASINATE_* variables and their arguments.

# Set up Key4hep without choosing a release and print the stack it picked.
k4_probe() {
    # shellcheck disable=SC1090
    source "$FASINATE_KEY4HEP_SETUP_SCRIPT" >&2
    if [[ -z ${KEY4HEP_STACK:-} ]]; then
        echo "Key4hep setup failed" >&2
        return 1
    fi
    readlink -f "$KEY4HEP_STACK"
}

# Configure, build and install DDML. Arguments are extra CMake arguments.
ddml_build() {
    local python=OFF models=OFF
    if [[ $FASINATE_DDML_EMBEDDED_PYTHON == true ]]; then python=ON; fi
    if [[ $FASINATE_DDML_DOWNLOAD_MODELS == true ]]; then models=ON; fi
    export FASINATE_ENV_NO_DDML=1
    # shellcheck disable=SC1090
    source "$FASINATE_DDML_ENV_FILE" || return 1
    unset FASINATE_ENV_NO_DDML
    set -ex
    cmake -S "$FASINATE_DDML_SOURCE_DIR" -B "$FASINATE_DDML_BUILD_DIR" \
        -DCMAKE_INSTALL_PREFIX="$FASINATE_DDML_INSTALL_DIR" \
        -DDDML_ENABLE_EMBEDDED_PYINFERENCE="$python" \
        -DDOWNLOAD_MODELS="$models" \
        "$@"
    cmake --build "$FASINATE_DDML_BUILD_DIR" -j "$FASINATE_DDML_JOBS"
    cmake --build "$FASINATE_DDML_BUILD_DIR" --target install
}

# Source the env file the way the run scripts do and check what it provides.
ddml_check() {
    # shellcheck disable=SC1090
    source "$FASINATE_DDML_ENV_FILE" || return 1
    local lib=$DDML_INSTALL_DIR/lib entry found=()
    if [[ ! -f $lib/libDDML.so ]]; then
        echo "check: $lib/libDDML.so is missing" >&2
        return 1
    fi
    python -c '
import os, sys, ddml
found = os.path.realpath(os.path.dirname(ddml.__file__))
expected = os.path.realpath(os.path.join(os.environ["DDML_SOURCE_DIR"], "python", "ddml"))
print("check: ddml Python package from", found)
sys.exit(0 if found == expected else "check: expected it from " + expected)
' || return 1
    for entry in LibTorch:CaloCloudsTwoAngleModelPolyhedraBarrelTorchModel \
        ONNX:RegularGridGANPolyhedraBarrelONNXModel \
        Python:CaloCloudsTwoAngleModelPolyhedraBarrelPyEmbeddedModel; do
        if grep -qs "${entry#*:}" "$lib"/*.components "$lib/libDDML.so"; then
            found+=("${entry%%:*}")
        fi
    done
    echo "check: inference backends built: ${found[*]:-none}"
    if [[ $FASINATE_DDML_EMBEDDED_PYTHON == true && " ${found[*]} " != *" Python "* ]]; then
        echo "check: the embedded-Python backend was requested but not built" >&2
        return 1
    fi
    if [[ " ${found[*]} " != *" LibTorch "* ]]; then
        echo "check: warning: no LibTorch backend, which ddsim_steer_di_photon.py uses" >&2
    fi
}

# ---------------------------------------------------------------------------

# Run one of the functions above in `bash --noprofile --norc` with a minimal
# environment: the Key4hep setup can only run once per shell, and leftovers
# from the calling shell (Python environments especially) confuse it.
run_clean() {
    local func=$1 var
    shift
    local env_vars=(HOME="$HOME" USER="${USER:-$(id -un)}"
        PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
    for var in LOGNAME TERM http_proxy https_proxy HTTP_PROXY HTTPS_PROXY \
        no_proxy NO_PROXY SSL_CERT_FILE SSL_CERT_DIR; do
        if [[ -n ${!var:-} ]]; then env_vars+=("$var=${!var}"); fi
    done
    for var in "${!FASINATE_@}"; do # the configuration, except bash arrays
        if [[ $(declare -p "$var") != "declare -a"* ]]; then env_vars+=("$var=${!var}"); fi
    done
    "${container[@]}" /usr/bin/env -i "${env_vars[@]}" \
        bash --noprofile --norc -c "$(declare -f "$func"); $func \"\$@\"" "$func" "$@"
}

# The last four directories above KEY4HEP_STACK name the release:
# .../releases/<release>/<platform>/key4hep-stack/<version>/setup.sh
release_of() {
    local path=$1 _
    for _ in 1 2 3 4; do path=$(dirname "$path"); done
    basename "$path"
}

normalize_url() { # lower case, no scheme, user or .git; host:path -> host/path
    local url=${1,,}
    url=${url%/}
    url=${url%.git}
    url=${url#*://}
    url=${url#*@}
    echo "${url/://}"
}

fetch_ddml() {
    local repo=$FASINATE_DDML_REPOSITORY branch=$FASINATE_DDML_BRANCH
    local commit=$FASINATE_DDML_COMMIT update=$FASINATE_DDML_UPDATE origin
    if [[ ! -e $src/.git ]]; then
        if [[ -e $src && -n $(ls -A "$src") ]]; then
            die "$src exists but is not a git clone; move it or change ddml.source_dir"
        fi
        run git clone --branch "$branch" "$repo" "$src"
        $dry_run && return
    else
        origin=$(git -C "$src" remote get-url origin) || die "$src has no 'origin' remote"
        [[ $(normalize_url "$origin") == "$(normalize_url "$repo")" ]] ||
            die "$src is a clone of $origin, not $repo; change ddml.source_dir or ddml.repository"
        if [[ $update == true ]]; then
            [[ -z $(git -C "$src" status --porcelain --untracked-files=no) ]] ||
                die "$src has uncommitted changes; commit or stash them, or use --set ddml.update=false"
            run git -C "$src" fetch origin "$branch"
        fi
    fi
    if [[ -n $commit ]]; then
        if ! git -C "$src" cat-file -e "$commit^{commit}" 2>/dev/null; then
            [[ $update == true ]] || die "commit $commit is not in $src and ddml.update is false"
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

remove_dir() {
    local dir=$1
    case $dir in
        / | "$HOME" | "$FASINATE_REPO_DIR" | "$src") die "refusing to delete $dir" ;;
    esac
    [[ $src != "$dir"/* ]] || die "refusing to delete $dir, which contains the DDML sources"
    run rm -rf -- "$dir"
}

write_env_file() { # write_env_file COMMIT FILE
    local commit=$1 file=$2
    {
        echo "# Environment for running ddsim with the DDML build in $install"
        echo "# Written by scripts/build_ddml.sh on $(date -u '+%Y-%m-%d %H:%M UTC'); rebuild rather than edit it."
        echo "#   DDML     $FASINATE_DDML_REPOSITORY ($FASINATE_DDML_BRANCH) at $commit"
        echo "#   Key4hep  $setup -r $release"
        echo "#   Built in ${image:-a clean shell, without a container}"
        echo "#"
        printf '_fasinate_k4_setup=%q\n' "$setup"
        printf '_fasinate_k4_release=%q\n' "$release"
        printf 'export DDML_SOURCE_DIR=%q\n' "$src"
        printf 'export DDML_INSTALL_DIR=%q\n' "$install"
        printf 'export DDML_COMMIT=%q\n' "$commit"
        cat <<'EOF'

# Source this file in a clean shell before running ddsim. It sets up the Key4hep
# release DDML was built against (the Key4hep setup can only run once per shell),
# adds the Torch and ONNX Runtime libraries the way DDML's setup_env_nightly.sh
# does, then sets up DDML. With FASINATE_ENV_NO_DDML=1 it stops before DDML,
# which is how build_ddml.sh builds with it.

if [ -n "${KEY4HEP_STACK:-}" ]; then
    _fasinate_found=$(readlink -f "$KEY4HEP_STACK")
    for _fasinate_i in 1 2 3 4; do _fasinate_found=$(dirname "$_fasinate_found"); done
    _fasinate_found=$(basename "$_fasinate_found")
    if [ "$_fasinate_found" != "$_fasinate_k4_release" ]; then
        echo "DDML env: Key4hep $_fasinate_found is already set up in this shell, but DDML was built against $_fasinate_k4_release; use a clean shell" >&2
        return 1
    fi
else
    source "$_fasinate_k4_setup" -r "$_fasinate_k4_release"
    if [ -z "${KEY4HEP_STACK:-}" ]; then
        echo "DDML env: setting up Key4hep $_fasinate_k4_release failed" >&2
        return 1
    fi
fi

_fasinate_dir=$(python -c 'import os, torch; print(os.path.dirname(torch.__file__))' 2>/dev/null) || _fasinate_dir=
if [ -n "$_fasinate_dir" ]; then
    export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:+$CMAKE_PREFIX_PATH:}$_fasinate_dir/share/cmake"
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$_fasinate_dir/lib"
else
    echo "DDML env: warning: this Key4hep release has no torch" >&2
fi
_fasinate_dir=$(python -c 'import os, onnxruntime; print(os.path.dirname(onnxruntime.__file__))' 2>/dev/null) || _fasinate_dir=
if [ -n "$_fasinate_dir" ]; then
    _fasinate_dir=$(realpath "$_fasinate_dir/../../../..")
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$_fasinate_dir/lib64:$_fasinate_dir/lib"
fi

if [ -z "${FASINATE_ENV_NO_DDML:-}" ]; then
    if [ ! -f "$DDML_INSTALL_DIR/bin/thisDDML.sh" ]; then
        echo "DDML env: $DDML_INSTALL_DIR/bin/thisDDML.sh not found; run scripts/build_ddml.sh" >&2
        return 1
    fi
    source "$DDML_INSTALL_DIR/bin/thisDDML.sh"
    # Key4hep releases include their own, older DDML; make sure this one wins.
    export LD_LIBRARY_PATH="$DDML_INSTALL_DIR/lib:$LD_LIBRARY_PATH"
    # The python-inference branch doesn't install its ddml Python package, so
    # use it from the sources. python/examples has the reference module for the
    # embedded-Python backend (cc3_sf_2a_wrapper).
    export PYTHONPATH="$DDML_SOURCE_DIR/python:$DDML_SOURCE_DIR/python/examples${PYTHONPATH:+:$PYTHONPATH}"
fi

unset _fasinate_k4_setup _fasinate_k4_release _fasinate_found _fasinate_i _fasinate_dir
return 0
EOF
    } >"$file"
    chmod 0644 "$file"
}

# ---------------------------------------------------------------------------

container=()
image=
binds=
if [[ $FASINATE_CONTAINER_ENABLED == true ]]; then
    image=$FASINATE_CONTAINER_IMAGE
    if ! runtime=$(command -v apptainer || command -v singularity); then
        msg="container.enabled is true, but neither apptainer nor singularity is on PATH"
        msg+=" (use --set container.enabled=false to build without a container)"
        $dry_run || die "$msg"
        say "warning: $msg"
        runtime=apptainer
    fi
    if [[ $(basename "$runtime") == apptainer ]]; then var_prefix=APPTAINER; else var_prefix=SINGULARITY; fi
    export "${var_prefix}_TMPDIR=$FASINATE_CONTAINER_TMP_DIR" "${var_prefix}_CACHEDIR=$FASINATE_CONTAINER_CACHE_DIR"
    # The DDML directories are mounted at the same paths as outside, so the paths
    # baked into the build and the env file are valid in both places.
    binds=/cvmfs,$FASINATE_DDML_BIND_PATHS
    IFS=, read -r -a bind_dirs <<<"$FASINATE_DDML_BIND_PATHS"
    build_is_bound=false
    for dir in "${bind_dirs[@]}"; do
        if [[ $build == "$dir" || $build == "$dir"/* ]]; then build_is_bound=true; fi
    done
    $build_is_bound || binds+=",$build"
    container=("$runtime" exec --bind "$binds" "$image")
fi

cat <<EOF
DDML        $FASINATE_DDML_REPOSITORY, ${FASINATE_DDML_COMMIT:-head of $FASINATE_DDML_BRANCH}
  sources   $src
  build     $build
  install   $install
  env file  $env_file
Key4hep     $setup -r $release
Container   ${image:-none, building in a clean shell on this machine}${binds:+
  mounts    $binds}
EOF

if [[ ! -f $setup ]]; then
    msg="$setup not found; is CVMFS mounted here?"
    if $dry_run; then say "warning: $msg"; else die "$msg"; fi
fi

say "getting DDML"
fetch_ddml
commit=$(git -C "$src" rev-parse HEAD 2>/dev/null || echo unknown)

if $clean; then
    say "deleting the old build"
    remove_dir "$build"
    remove_dir "$install"
fi

# Everything mounted into the container has to exist before it starts.
run mkdir -p "$build" "$install" "$(dirname "$env_file")"
if [[ -n $image ]]; then
    run mkdir -p "$FASINATE_CONTAINER_TMP_DIR" "$FASINATE_CONTAINER_CACHE_DIR"
fi

if [[ -n $image && $image != *://* && ! -f $image ]]; then
    [[ -n $FASINATE_CONTAINER_PULL_FROM ]] || die "container image $image not found and container.pull_from is empty"
    say "pulling the container image"
    run mkdir -p "$(dirname "$image")"
    run "$runtime" pull "$image" "$FASINATE_CONTAINER_PULL_FROM"
fi

release_was_latest=false
if [[ -z $release || $release == latest ]]; then
    release_was_latest=true
    if $dry_run; then
        say "would find out which Key4hep release 'latest' is"
    else
        say "finding out which Key4hep release 'latest' is"
        stack=$(run_clean k4_probe | tail -n 1) || die "setting up Key4hep failed (see above)"
        release=$(release_of "$stack")
        [[ $release =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2} ]] ||
            die "could not tell the release from $stack; set key4hep.release instead"
        say "the latest Key4hep release is $release"
    fi
fi

if $dry_run; then
    say "would write $env_file, then build, install and check DDML"
    exit 0
fi

# Build with a new env file and only replace the old one once DDML is installed,
# so a failed attempt leaves the previous build usable.
new_env=$(mktemp "$env_file.XXXXXX")
write_env_file "$commit" "$new_env"
FASINATE_DDML_ENV_FILE=$new_env

say "building DDML"
if ! run_clean ddml_build "${FASINATE_DDML_CMAKE_ARGS[@]}"; then
    rm -f "$new_env"
    die "the build failed (see above); $env_file is unchanged"
fi
mv "$new_env" "$env_file"
FASINATE_DDML_ENV_FILE=$env_file
say "wrote $env_file"

say "checking the environment file"
run_clean ddml_check || die "$env_file doesn't give a working DDML (see above)"

say "built DDML $commit against Key4hep $release"
echo "Source $env_file in a clean shell to use it; run_sim.sh and run_all.sh do that in the container."
if [[ -z $FASINATE_DDML_COMMIT ]] || $release_was_latest; then
    echo "To rebuild exactly this later, pin it in configs/local.yaml:"
    echo
    if [[ -z $FASINATE_DDML_COMMIT ]]; then printf 'ddml:\n  commit: "%s"\n' "$commit"; fi
    if $release_was_latest; then printf 'key4hep:\n  release: "%s"\n' "$release"; fi
fi
