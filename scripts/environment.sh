# shellcheck shell=bash
# How FaSiNaTe's scripts start a clean shell, inside the container unless
# container.enabled is false: the one copy of it, which run_in_environment.sh,
# build_ddml.sh and build_patches.sh source after loading the configuration
# (the FASINATE_* variables, from scripts/load_config.sh).
#
#   fasinate_clean_env [VAR ...]
#       Set the array fasinate_env to a minimal environment for env -i: HOME,
#       USER, a plain PATH, LOGNAME and TERM, the configuration (FASINATE_*,
#       except bash arrays, which can't be passed on) and the VARs named, if
#       they are set. The Key4hep setup can only run once per shell, and
#       leftovers from the calling shell (Python environments especially)
#       confuse it.
#   fasinate_in_container
#       Succeed if this shell runs in an Apptainer/Singularity container.
#   fasinate_find_runtime
#       Set fasinate_runtime to apptainer's (or singularity's) path, pointing
#       its temporary and cache folders at container.tmp_dir and cache_dir;
#       fail if neither is on PATH.
#   fasinate_binds DIR ...
#       Print the folders (each argument may be a list separated by commas)
#       separated by commas, leaving out empty ones, repeats and folders inside
#       others.
#   fasinate_container_command DIR ...
#       Set the array fasinate_container to the command that runs the rest in
#       the container with those folders mounted at the same paths as outside,
#       so paths mean the same in there; empty if container.enabled is false or
#       this already runs in a container.
#
# shellcheck disable=SC2034 # the arrays set are for the scripts that source this

fasinate_clean_env() {
    local var
    fasinate_env=(HOME="$HOME" USER="${USER:-$(id -un)}"
        PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin)
    for var in LOGNAME TERM "$@"; do
        if [[ -n ${!var:-} ]]; then fasinate_env+=("$var=${!var}"); fi
    done
    for var in "${!FASINATE_@}"; do
        if [[ $(declare -p "$var") != "declare -a"* ]]; then fasinate_env+=("$var=${!var}"); fi
    done
}

fasinate_in_container() {
    [[ -n ${APPTAINER_CONTAINER:-}${SINGULARITY_CONTAINER:-} || -d /.singularity.d ]]
}

fasinate_find_runtime() {
    local prefix
    fasinate_runtime=$(command -v apptainer || command -v singularity) || return 1
    if [[ ${fasinate_runtime##*/} == apptainer ]]; then prefix=APPTAINER; else prefix=SINGULARITY; fi
    export "${prefix}_TMPDIR=$FASINATE_CONTAINER_TMP_DIR" "${prefix}_CACHEDIR=$FASINATE_CONTAINER_CACHE_DIR"
}

fasinate_binds() {
    local dirs=() dir other binds=
    IFS=, read -r -a dirs <<<"$(IFS=,; echo "$*")"
    for dir in "${dirs[@]}"; do
        [[ -n $dir ]] || continue
        for other in "${dirs[@]}"; do
            if [[ -n $other && $dir == "$other"/* ]]; then continue 2; fi
        done
        if [[ ,$binds, != *",$dir,"* ]]; then binds+=${binds:+,}$dir; fi
    done
    echo "$binds"
}

fasinate_container_command() {
    fasinate_container=()
    if [[ $FASINATE_CONTAINER_ENABLED != true ]] || fasinate_in_container; then
        return 0
    fi
    if ! fasinate_find_runtime; then
        echo "container.enabled is true, but neither apptainer nor singularity is on PATH" >&2
        return 1
    fi
    mkdir -p "$FASINATE_CONTAINER_TMP_DIR" "$FASINATE_CONTAINER_CACHE_DIR" || return 1
    fasinate_container=("$fasinate_runtime" exec --bind "$(fasinate_binds "$@")"
        "$FASINATE_CONTAINER_IMAGE")
}
