#!/usr/bin/env python3
"""Load FaSiNaTe's configuration.

Settings are merged in this order, later ones winning:

  1. configs/default.yaml
  2. configs/local.yaml, if it exists (git-ignored, for your own setup)
  3. the files listed in $FASINATE_CONFIG (separated by ':'), then --config FILE
  4. --set SECTION.KEY=VALUE

Override files may only set keys that exist in default.yaml, so typos are
caught. In string values, ${section.key} refers to another setting and $VAR or
${VAR} to an environment variable. Relative paths (keys ending in _dir, _file,
_path or _script, and container.image) are taken relative to the repository.

Command line:
  fasinate_config.py            print the merged configuration
  fasinate_config.py --get KEY  print one setting, e.g. --get paths.data_dir
  fasinate_config.py --shell    print bash assignments, FASINATE_<SECTION>_<KEY>=...
                                (lists become bash arrays, which can't be exported)

From Python:
  from fasinate_config import load_config
  config = load_config()
  config["paths"]["data_dir"]
"""

import argparse
import datetime
import hashlib
import json
import os
import re
import shlex
import sys

try:
    import yaml
except ImportError:
    yaml = None

REPO_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DEFAULT_FILE = os.path.join(REPO_DIR, "configs", "default.yaml")
LOCAL_FILE = os.path.join(REPO_DIR, "configs", "local.yaml")

_PATH_SUFFIXES = ("_dir", "_file", "_path", "_script")
_PATH_NAMES = ("image",)
_REFERENCE = re.compile(r"\$\{([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+)\}")
_ENV_VAR = re.compile(r"\$(?:[A-Za-z_]\w*|\{[A-Za-z_]\w*\})")


class ConfigError(Exception):
    pass


def load_config(extra_files=(), overrides=()):
    """Return the merged and resolved configuration as nested dicts.

    extra_files are merged after configs/local.yaml and $FASINATE_CONFIG;
    overrides are "section.key=value" strings, applied last.
    """
    config = _read_yaml(DEFAULT_FILE)
    files = [LOCAL_FILE] if os.path.exists(LOCAL_FILE) else []
    files += [f for f in os.environ.get("FASINATE_CONFIG", "").split(":") if f]
    files += list(extra_files)
    for path in files:
        _merge(config, _read_yaml(path), path)
    for item in overrides:
        _apply_override(config, item)
    return _resolve(config)


def shell_assignments(config):
    """Bash code that sets FASINATE_<SECTION>_<KEY> for every setting."""
    lines = ["export FASINATE_REPO_DIR=" + shlex.quote(REPO_DIR)]
    for name, value in _flatten(config):
        var = "FASINATE_" + re.sub(r"\W", "_", name).upper()
        if isinstance(value, list):
            items = " ".join(shlex.quote(_to_str(v)) for v in value)
            lines.append("%s=(%s)" % (var, items))
        else:
            lines.append("export %s=%s" % (var, shlex.quote(_to_str(value))))
    # Directories a container needs to see (at the same path) to use the DDML build.
    lines.append("export FASINATE_DDML_BIND_PATHS="
                 + shlex.quote(",".join(ddml_bind_paths(config))))
    # The same for the detector configuration clones (fetch_detector_configs.sh).
    lines.append("export FASINATE_DETECTOR_CONFIGS_BIND_PATHS="
                 + shlex.quote(",".join(detector_configs_bind_paths(config))))
    lines.append("export FASINATE_DDML_BUILD_STAMP=" + ddml_build_stamp(config))
    return "\n".join(lines)


def ddml_build_stamp(config):
    """Short hash of the settings that decide what DDML is built from and against.

    build_ddml.sh writes it into the env file, and setup.sh rebuilds when it no
    longer matches. Paths are left out, so the stamp doesn't change where the
    repository is mounted somewhere else (e.g. /user_repo in a container).
    """
    flat = dict(_flatten(config))
    image = flat["container.image"]
    values = [flat[key] for key in (
        "ddml.repository", "ddml.branch", "ddml.commit", "ddml.embedded_python",
        "ddml.download_models", "ddml.cmake_args", "key4hep.setup_script",
        "key4hep.release", "container.enabled")]
    values.append(image if "://" in image else os.path.basename(image))
    text = json.dumps(values, default=_to_str)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def ddml_bind_paths(config):
    """Top-level directories holding the DDML sources, install and env file."""
    ddml = config["ddml"]
    candidates = [ddml["source_dir"], ddml["install_dir"],
                  os.path.dirname(ddml["env_file"])]
    paths = []
    for path in candidates:
        if not any(_is_within(path, other) for other in candidates if other != path):
            if path not in paths:
                paths.append(path)
    return paths


def detector_configs_bind_paths(config):
    """The clones of detector_configs, without those inside the DDML folders."""
    ddml = ddml_bind_paths(config)
    paths = []
    for entry in (config.get("detector_configs") or {}).values():
        path = entry["source_dir"]
        if path in ddml or any(_is_within(path, other) for other in ddml):
            continue
        if path not in paths:
            paths.append(path)
    return paths


def _is_within(path, parent):
    return path.startswith(parent.rstrip(os.sep) + os.sep)


def _read_yaml(path):
    if yaml is None:
        raise ConfigError(
            "PyYAML is needed to read the configuration. Install it with "
            "'python3 -m pip install --user pyyaml', or set FASINATE_PYTHON "
            "to a Python interpreter that has it.")
    try:
        with open(path) as stream:
            data = yaml.safe_load(stream)
    except OSError as error:
        raise ConfigError("cannot read %s: %s" % (path, error.strerror))
    except yaml.YAMLError as error:
        raise ConfigError("%s: %s" % (path, error))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError("%s: expected sections of settings at the top level" % path)
    return data


def _merge(base, override, source, prefix=""):
    for key, value in override.items():
        name = prefix + str(key)
        if key not in base:
            raise ConfigError("%s: unknown setting '%s'" % (source, name))
        if isinstance(base[key], dict):
            if value is None:  # a section with everything commented out
                continue
            if not isinstance(value, dict):
                raise ConfigError("%s: '%s' is a section, not a single value" % (source, name))
            _merge(base[key], value, source, name + ".")
        elif isinstance(value, dict):
            raise ConfigError("%s: '%s' is a single value, not a section" % (source, name))
        else:
            base[key] = value


def _apply_override(config, item):
    name, sep, text = item.partition("=")
    if not sep or not name:
        raise ConfigError("--set expects SECTION.KEY=VALUE, not '%s'" % item)
    try:
        value = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise ConfigError("--set %s: %s" % (item, error))
    tree = value
    for key in reversed(name.split(".")):
        tree = {key: tree}
    _merge(config, tree, "--set")


def _flatten(tree, prefix=""):
    for key, value in tree.items():
        name = prefix + str(key)
        if isinstance(value, dict):
            for item in _flatten(value, name + "."):
                yield item
        else:
            yield name, value


def _to_str(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, datetime.date):  # YAML reads 2026-05-13 as a date
        return value.isoformat()
    return str(value)


def _is_path_key(name):
    leaf = name.rsplit(".", 1)[-1]
    return leaf.endswith(_PATH_SUFFIXES) or leaf in _PATH_NAMES


def _resolve(config):
    raw = dict(_flatten(config))
    done = {}

    def value_of(name, chain):
        if name in done:
            return done[name]
        if name in chain:
            raise ConfigError("circular reference: " + " -> ".join(chain + (name,)))
        if name not in raw:
            raise ConfigError("'%s' refers to unknown setting '%s'" % (chain[-1], name))
        value = raw[name]
        if isinstance(value, str):
            value = expand(name, value, chain + (name,))
        elif isinstance(value, list):
            value = [expand(name, v, chain + (name,)) if isinstance(v, str) else v
                     for v in value]
        elif isinstance(value, datetime.date):
            value = value.isoformat()
        done[name] = value
        return value

    def expand(name, text, chain):
        def reference(match):
            target = value_of(match.group(1), chain)
            if target is None or isinstance(target, list):
                raise ConfigError("'%s' refers to '%s', which is %s" % (
                    name, match.group(1), "empty" if target is None else "a list"))
            return _to_str(target)

        text = os.path.expandvars(_REFERENCE.sub(reference, text))
        unset = _ENV_VAR.search(text)
        if unset:
            raise ConfigError("'%s' uses %s, which is not set in the environment"
                              % (name, unset.group(0)))
        if _is_path_key(name) and text and "://" not in text:
            text = os.path.normpath(os.path.join(REPO_DIR, os.path.expanduser(text)))
        return text

    for name in raw:
        value_of(name, ())

    resolved = {}
    for name in raw:  # keep the order of default.yaml
        *sections, key = name.split(".")
        node = resolved
        for section in sections:
            node = node.setdefault(section, {})
        node[key] = done[name]
    return resolved


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Print FaSiNaTe's merged configuration.",
        epilog="See the module docstring for where settings come from.")
    parser.add_argument("--config", action="append", default=[], metavar="FILE",
                        help="merge FILE over the configuration (repeatable)")
    parser.add_argument("--set", action="append", default=[], dest="overrides",
                        metavar="SECTION.KEY=VALUE", help="override one setting (repeatable)")
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--get", metavar="SECTION.KEY", help="print a single setting")
    output.add_argument("--shell", action="store_true",
                        help="print bash assignments for every setting")
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config, args.overrides)
        if args.shell:
            print(shell_assignments(config))
        elif args.get:
            node = config
            for key in args.get.split("."):
                if not isinstance(node, dict) or key not in node:
                    raise ConfigError("unknown setting '%s'" % args.get)
                node = node[key]
            if isinstance(node, dict):
                print(yaml.safe_dump(node, sort_keys=False), end="")
            elif isinstance(node, list):
                print("\n".join(_to_str(v) for v in node))
            else:
                print(_to_str(node))
        else:
            print(yaml.safe_dump(config, sort_keys=False), end="")
    except ConfigError as error:
        print("fasinate config: %s" % error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
