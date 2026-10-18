# Set up this shell for FaSiNaTe: the Key4hep release, ML libraries and DDML
# build chosen in the configuration (configs/default.yaml and
# configs/local.yaml; see README.md), the detector configurations
# (detector_configs, e.g. ILDConfig) that reconstruction needs, and the patches
# to the release (patches: newer builds of its packages, for fixes it doesn't
# have yet). Source it from bash, in a clean shell:
#
#   source setup.sh
#
# If DDML isn't built for this configuration yet, it first builds it with
# scripts/build_ddml.sh, which can take a while, it clones the detector
# configurations that aren't there yet with scripts/fetch_detector_configs.sh,
# and it builds the patches that aren't built yet with scripts/build_patches.sh.
# Inside a container started by the run_*.sh scripts it uses the configuration
# they pass in, and expects them to have done all three already.

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    echo "setup.sh sets up the shell it runs in, so source it: source $0" >&2
    exit 1
fi

_fasinate_repo=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

_fasinate_cleanup() {
    unset _fasinate_repo _fasinate_given _fasinate_var _fasinate_missing
    unset -f _fasinate_cleanup _fasinate_is_built
}

_fasinate_is_built() {
    [ -f "$FASINATE_DDML_ENV_FILE" ] && [ -f "$FASINATE_DDML_INSTALL_DIR/bin/thisDDML.sh" ] &&
        grep -qxF "# build stamp: $FASINATE_DDML_BUILD_STAMP" "$FASINATE_DDML_ENV_FILE"
}

# A run script passes its configuration into the container. Use that rather than
# reading the files again: the repository is mounted elsewhere there
# (/user_repo), which would move relative paths.
_fasinate_given=
if [ -n "${APPTAINER_CONTAINER:-}${SINGULARITY_CONTAINER:-}" ] || [ -d /.singularity.d ]; then
    if [ -n "${FASINATE_DDML_ENV_FILE:-}" ] && [ -n "${FASINATE_DDML_BUILD_STAMP:-}" ]; then
        _fasinate_given=1
    fi
fi
if [ -z "$_fasinate_given" ]; then
    # shellcheck source=scripts/load_config.sh
    source "$_fasinate_repo/scripts/load_config.sh" || { _fasinate_cleanup; return 1; }
fi

if ! _fasinate_is_built; then
    if [ -n "$_fasinate_given" ]; then
        echo "setup.sh: DDML isn't built for this configuration; build it outside the container with scripts/build_ddml.sh" >&2
        _fasinate_cleanup
        return 1
    fi
    echo "setup.sh: DDML isn't built for this configuration yet, so building it first" >&2
    "$_fasinate_repo/scripts/build_ddml.sh" --if-needed || { _fasinate_cleanup; return 1; }
fi

# The detector configurations: cloned if they aren't there, except in a
# container, which may have no network; the run scripts fetch them beforehand
# and mount them (FASINATE_DETECTOR_CONFIGS_BIND_PATHS).
if [ -n "$_fasinate_given" ]; then
    _fasinate_missing=
    for _fasinate_var in "${!FASINATE_DETECTOR_CONFIGS_@}"; do
        case $_fasinate_var in
            *_SOURCE_DIR)
                [ -e "${!_fasinate_var}/.git" ] || _fasinate_missing="$_fasinate_missing ${!_fasinate_var}"
                ;;
        esac
    done
    if [ -n "$_fasinate_missing" ]; then
        echo "setup.sh: the detector configurations aren't in:$_fasinate_missing; fetch them outside the container with scripts/fetch_detector_configs.sh, and mount FASINATE_DETECTOR_CONFIGS_BIND_PATHS" >&2
        _fasinate_cleanup
        return 1
    fi
else
    "$_fasinate_repo/scripts/fetch_detector_configs.sh" --if-needed >&2 || { _fasinate_cleanup; return 1; }
fi

# The patches to the release: built if needed, except in a container, where
# the run scripts build them beforehand and mount them
# (FASINATE_PATCHES_BIND_PATHS). Nothing to do once there are none.
if [ -n "${FASINATE_PATCHES_ENABLED:-}" ] && [ -z "$_fasinate_given" ]; then
    "$_fasinate_repo/scripts/build_patches.sh" --if-needed >&2 || { _fasinate_cleanup; return 1; }
fi

# shellcheck disable=SC1090
source "$FASINATE_DDML_ENV_FILE" || { _fasinate_cleanup; return 1; }
echo "FaSiNaTe: Key4hep $DDML_KEY4HEP_RELEASE, DDML $DDML_COMMIT in $DDML_INSTALL_DIR"

# After Key4hep, whose MARLIN_DLL they change
if [ -n "${FASINATE_PATCHES_ENABLED:-}" ]; then
    if [ ! -f "$FASINATE_PATCHES_ENV_FILE" ]; then
        echo "setup.sh: the patches aren't built; build them outside the container with scripts/build_patches.sh, and mount FASINATE_PATCHES_BIND_PATHS" >&2
        _fasinate_cleanup
        return 1
    fi
    # shellcheck disable=SC1090
    source "$FASINATE_PATCHES_ENV_FILE" || { _fasinate_cleanup; return 1; }
fi
_fasinate_cleanup
