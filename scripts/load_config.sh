# Source this file to load FaSiNaTe's configuration into FASINATE_* variables:
#
#   source scripts/load_config.sh || exit 1
#
# e.g. FASINATE_PATHS_DATA_DIR, FASINATE_DDML_ENV_FILE, FASINATE_CONTAINER_IMAGE.
# Settings come from configs/default.yaml, configs/local.yaml and the files in
# $FASINATE_CONFIG; run `python3 scripts/fasinate_config.py --help` for details.
# Set FASINATE_PYTHON to use a Python other than python3 (it needs PyYAML).

_fasinate_settings=$("${FASINATE_PYTHON:-python3}" "$(dirname "${BASH_SOURCE[0]}")/fasinate_config.py" --shell) || {
    unset _fasinate_settings
    echo "Could not load the FaSiNaTe configuration (see the message above)" >&2
    return 1
}
eval "$_fasinate_settings"
unset _fasinate_settings
