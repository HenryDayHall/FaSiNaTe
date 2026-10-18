#!/usr/bin/env python3
"""Work out the simulations that simulate_models.sh runs.

Every model described in models/*.yaml simulates the input of every benchmark
scenario in the configuration, or of the ones named, or the files given. Each
simulation writes

  <paths.data_dir>/sim_output/<scenario>/<model>/<input name>.root

or <input name>_n<N>.root when only the first N events are simulated; files
given with --input are simulated as the scenario "custom". A simulation whose
output exists is done, and is skipped unless forced.

This runs in two steps:

  simulation_jobs.py plan [--scenario NAME] [--input FILE] [--file FILE]
                          [--events N] [--force] [MODEL ...]
      Before the simulation environment is set up: read the configuration and
      the model descriptions, check them, and print the plan as JSON.
      --scenario and --input can be repeated; without either, every scenario
      is simulated. With --file, it is just that input, as part of the one
      scenario given (as the jobs of slurm_pipeline.sh simulate).

  simulation_jobs.py jobs [--dry-run] < PLAN
      In the environment from setup.sh: check that it runs the DDML build from
      scripts/build_ddml.sh and has what each model needs from it, name the
      scenarios' input files, and print the simulations still to run, one per
      line, with these fields separated by tabs: model, scenario, input,
      output, compact file, presets (separated by commas) and the number of
      events for ddsim (-1 for all). What is run and skipped is reported on
      stderr.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys

from fasinate_config import REPO_DIR, ConfigError, load_config
from scenario_inputs import INPUT_FILES, scenario_commands

MODELS_DIR = os.path.join(REPO_DIR, "models")
# The scenario that files given with --input are simulated as.
CUSTOM_SCENARIO = "custom"
# Model and scenario names become folder names.
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+-]*$")
_MD5 = re.compile(r"^[0-9a-f]{32}$")
# A line of a .components file, as DD4hep's plugin service reads it.
_COMPONENT = re.compile(r"^\s*(?:(v[0-9]+)::)?([^:]+):(.*\S)\s*$")


class PlanError(Exception):
    pass


def say(message):
    print("simulate_models: " + message, file=sys.stderr)


# ---------------------------------------------------------------------------
# The plan, made outside the simulation environment


def model_names():
    """The models described in models/, by file name."""
    if not os.path.isdir(MODELS_DIR):
        return []
    return sorted(name[:-len(".yaml")] for name in os.listdir(MODELS_DIR)
                  if name.endswith(".yaml"))


def load_model(name):
    """What simulate_models.sh needs from models/<name>.yaml, after checking it."""
    path = os.path.join(MODELS_DIR, name + ".yaml")
    source = os.path.relpath(path, REPO_DIR)
    if not os.path.isfile(path):
        raise PlanError("there is no model '%s'; models/ has: %s"
                        % (name, ", ".join(model_names()) or "nothing"))
    if not _NAME.match(name):
        raise PlanError("%s: model names become folder names, so use only letters, "
                        "digits and _.+-" % source)
    try:
        import yaml
    except ImportError:
        raise PlanError("PyYAML is needed to read %s; install it with "
                        "'python3 -m pip install --user pyyaml'" % source)
    try:
        with open(path) as stream:
            data = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as error:
        raise PlanError("cannot read %s: %s" % (source, error))

    def get(keys, is_valid, what, required=True):
        node = data
        for key in keys.split("."):
            if not isinstance(node, dict) or key not in node:
                if required:
                    raise PlanError("%s: %s is missing; it should be %s" % (source, keys, what))
                return None
            node = node[key]
        if not is_valid(node):
            raise PlanError("%s: %s should be %s" % (source, keys, what))
        return node

    def text(value):
        return isinstance(value, str) and bool(value.strip())

    def section(value):
        return isinstance(value, dict) and bool(value)

    def names(value):
        return isinstance(value, list) and bool(value) and all(text(v) for v in value)

    get("representation", text, "how the model represents a shower, e.g. point_cloud")
    get("fixed_at_training", section, "the factors fixed when the model was trained")
    get("context", section, "the factors given to the model for each shower")
    model = {
        "name": name,
        "config": source,
        "compact_file": get("fixed_at_training.detector.compact_file", text,
                            "the detector's compact file, relative to k4geo"),
        "particle": get("fixed_at_training.particle.name", text,
                        "the Geant4 name of the particle the model was trained on"),
        "presets": get("ddml.presets", names,
                       "a list of DDML presets (from python/ddml/configs.py in DDML)"),
        "weights": get("ddml.weights", text,
                       "the path of the weights in the DDML clone", required=False),
        "weights_md5": get("ddml.weights_md5",
                           lambda v: isinstance(v, str) and bool(_MD5.match(v)),
                           "an md5 sum: 32 hex digits, in quotes", required=False),
    }
    if model["weights_md5"] and not model["weights"]:
        raise PlanError("%s: ddml.weights_md5 is set, but not ddml.weights" % source)
    return model


def make_plan(models=(), scenarios=(), inputs=(), events=None, force=False, file=None):
    """The simulations to consider, as a JSON-able dict."""
    config = load_config()
    names = []
    for name in models or model_names():
        name = os.path.basename(name)
        if name.endswith(".yaml"):
            name = name[:-len(".yaml")]
        if name not in names:
            names.append(name)
    if not names:
        raise PlanError("there are no model descriptions (*.yaml) in %s" % MODELS_DIR)
    if events is not None and events < 1:
        raise PlanError("the number of events must be at least 1, not %d" % events)

    groups = []
    if file is not None:
        # One input of one scenario, as geant4_jobs.py simulates with --file
        if len(scenarios) != 1 or inputs:
            raise PlanError("--file needs exactly one --scenario, the one the file belongs "
                            "to (custom for files outside the scenarios), and no --input")
        if not _NAME.match(scenarios[0]):
            raise PlanError("scenario names become folder names, so use only letters, "
                            "digits and _.+- in '%s'" % scenarios[0])
        if not os.path.isfile(file):
            raise PlanError("input file not found: %s" % file)
        # The real path, as that is what gets mounted in the container
        groups.append({"scenario": scenarios[0], "type": "files",
                       "files": [os.path.realpath(file)]})
    elif scenarios or not inputs:
        for name, kind, commands in scenario_commands(config, scenarios):
            if not _NAME.match(name):
                raise PlanError("scenario names become folder names, so use only letters, "
                                "digits and _.+- in '%s'" % name)
            if kind not in INPUT_FILES:
                raise PlanError("scenario %s: there's no way to tell which files a %s input "
                                "makes; add one to INPUT_FILES in scripts/scenario_inputs.py"
                                % (name, kind))
            groups.append({"scenario": name, "type": kind, "commands": commands})
    if inputs:
        if CUSTOM_SCENARIO in config["scenarios"]:
            raise PlanError("files given with --input are simulated as the scenario '%s', "
                            "but the configuration has a scenario of that name"
                            % CUSTOM_SCENARIO)
        files = []
        for path in inputs:
            if not os.path.isfile(path):
                raise PlanError("input file not found: %s" % path)
            # The real path, as that is what gets mounted in the container
            path = os.path.realpath(path)
            if path not in files:
                files.append(path)
        groups.append({"scenario": CUSTOM_SCENARIO, "type": "files", "files": files})

    return {
        "sim_dir": os.path.join(config["paths"]["data_dir"], "sim_output"),
        "events": events,
        "force": force,
        "models": [load_model(name) for name in names],
        "inputs": groups,
    }


def report_plan(plan):
    say("models: " + ", ".join(model["name"] for model in plan["models"]))
    for group in plan["inputs"]:
        count = len(group["files"] if group["type"] == "files" else group["commands"])
        say("scenario %s: %d input file%s%s" % (
            group["scenario"], count, "" if count == 1 else "s",
            " given" if group["type"] == "files" else ", from its %s" % group["type"]))
    if plan["events"] is not None:
        say("simulating only the first %d events of each" % plan["events"])


# ---------------------------------------------------------------------------
# The jobs, worked out in the simulation environment


def _same_path(first, second):
    return bool(first) and bool(second) and os.path.realpath(first) == os.path.realpath(second)


def _library_path():
    return [d for d in os.environ.get("LD_LIBRARY_PATH", "").split(":") if d]


def _plugin_factories():
    """{factory: (directory, library)} for the plugins DD4hep can load.

    DD4hep's plugin service reads the .components files in the folders on
    LD_LIBRARY_PATH, in order, and the first declaration of a factory wins.
    """
    found = {}
    for directory in _library_path():
        try:
            file_names = sorted(os.listdir(directory))
        except OSError:
            continue
        for file_name in file_names:
            path = os.path.join(directory, file_name)
            if not file_name.endswith(".components") or not os.path.isfile(path):
                continue
            try:
                with open(path, errors="replace") as stream:
                    lines = stream.read().splitlines()
            except OSError:
                continue
            for line in lines:
                match = _COMPONENT.match(line)
                if match and match.group(1) == "v2":
                    found.setdefault(match.group(3), (directory, match.group(2)))
    return found


def _first_on_library_path(library):
    """Where the dynamic loader finds a library named in a .components file."""
    if os.path.isabs(library):
        return library if os.path.isfile(library) else None
    for directory in _library_path():
        path = os.path.join(directory, library)
        if os.path.isfile(path):
            return path
    return None


def _module_origin(module):
    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, ValueError):
        return None
    return spec.origin if spec else None


def _md5(path):
    digest = hashlib.md5()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_environment(models):
    """Check that this environment runs the DDML build from
    scripts/build_ddml.sh and has what each model needs from it. Reports what
    each model runs, and returns each model's compact file."""
    env = os.environ
    source = env.get("DDML_SOURCE_DIR")
    install = env.get("DDML_INSTALL_DIR")
    if not source or not install:
        raise PlanError("DDML isn't set up in this environment; source setup.sh first")

    problems = []
    for var, setting in (("DDML_SOURCE_DIR", "FASINATE_DDML_SOURCE_DIR"),
                         ("DDML_INSTALL_DIR", "FASINATE_DDML_INSTALL_DIR")):
        if not _same_path(env[var], env.get(setting)):
            problems.append("%s is %s, but scripts/build_ddml.sh builds DDML in %s for "
                            "this configuration" % (var, env[var], env.get(setting) or "(unset)"))
    lib_dir = os.path.join(install, "lib")

    try:
        import ddml
        from ddml import configs as presets
    except Exception as error:  # ImportError, or anything the package raises
        raise PlanError("cannot import DDML's Python package, ddml: %s" % error)
    ddml_dir = os.path.dirname(os.path.realpath(ddml.__file__))
    if not _same_path(ddml_dir, os.path.join(source, "python", "ddml")):
        problems.append("the ddml Python package comes from %s, not from the DDML "
                        "build's sources in %s" % (ddml_dir, source))
    model_config = getattr(ddml, "ModelConfig", None)
    if model_config is None:
        raise PlanError("\n  ".join(["the ddml Python package in %s has no presets "
                                     "(ddml.ModelConfig); is it DDML's python-inference "
                                     "branch?" % ddml_dir] + problems))

    factories = _plugin_factories()
    k4geo = env.get("k4geo_DIR") or env.get("K4GEO")
    compact_files = {}
    for model in models:
        name = model["name"]
        compact = model["compact_file"]
        if not os.path.isabs(compact):
            compact = os.path.join(k4geo, compact) if k4geo else None
        if compact is None:
            problems.append("%s: k4geo_DIR isn't set, so its compact file can't be found" % name)
        elif not os.path.isfile(compact):
            problems.append("%s: compact file not found: %s" % (name, compact))
        compact_files[name] = compact

        for preset in model["presets"]:
            settings = getattr(presets, preset, None)
            if preset.startswith("_") or not isinstance(settings, model_config):
                problems.append("%s: DDML has no preset %s (see %s)" % (
                    name, preset, os.path.join(source, "python", "ddml", "configs.py")))
                continue
            plugin = settings.plugin.split("/")[0]
            if plugin not in factories:
                problems.append("%s: preset %s needs the plugin %s, which no .components "
                                "file on LD_LIBRARY_PATH declares; was DDML built with the "
                                "inference backend it needs?" % (name, preset, plugin))
            else:
                directory, library = factories[plugin]
                loaded = _first_on_library_path(library)
                if not (_same_path(directory, lib_dir) and loaded
                        and _same_path(os.path.dirname(loaded), lib_dir)):
                    problems.append("%s: the plugin %s would be loaded from %s, not from "
                                    "the DDML build in %s" % (
                                        name, plugin, loaded or os.path.join(directory, library),
                                        lib_dir))
                else:
                    say("%s: preset %s runs %s from %s" % (name, preset, plugin, loaded))
            particles = sorted(settings.applicable_particles)
            if particles != [model["particle"]]:
                problems.append("%s: preset %s gives %s to the model, but %s says it was "
                                "trained on %s" % (name, preset, ", ".join(particles) or "nothing",
                                                   model["config"], model["particle"]))
            module = settings.plugin_properties.get("PythonModule")
            if module:
                origin = _module_origin(module)
                if origin:
                    say("%s: preset %s runs the Python module %s" % (name, preset, origin))
                else:
                    problems.append("%s: preset %s runs the Python module %s, which isn't "
                                    "on PYTHONPATH" % (name, preset, module))

        if model["weights"]:
            weights = os.path.join(source, model["weights"])
            md5 = _md5(weights) if model["weights_md5"] and os.path.isfile(weights) else None
            if not os.path.isfile(weights):
                problems.append("%s: weights not found: %s (building DDML downloads them "
                                "when ddml.download_models is true)" % (name, weights))
            elif md5 and md5 != model["weights_md5"]:
                problems.append("%s: the md5 sum of %s is %s, not %s as in %s; if these "
                                "weights are intended, update weights_md5 there" % (
                                    name, weights, md5, model["weights_md5"], model["config"]))
            else:
                say("%s: weights %s%s" % (name, weights, ", md5 checked" if md5 else ""))

    if problems:
        raise PlanError("this environment can't run the models as described:\n  "
                        + "\n  ".join(problems))
    return compact_files


def make_jobs(plan, dry_run=False):
    """The simulations still to run, as tuples of strings (see the docstring)."""
    compact_files = check_environment(plan["models"])

    inputs = []  # (scenario, input files)
    for group in plan["inputs"]:
        if group["type"] == "files":
            files = group["files"]
        else:
            files = INPUT_FILES[group["type"]](group["commands"])
        inputs.append((group["scenario"], [os.path.abspath(f) for f in files]))

    events = plan["events"]
    suffix = "" if events is None else "_n%d" % events
    jobs = []
    missing = {}  # input file: scenario
    outputs = {}
    for model in plan["models"]:
        name = model["name"]
        for scenario, files in inputs:
            done, redo, new, absent = 0, [], [], 0
            for path in files:
                stem = os.path.splitext(os.path.basename(path))[0]
                output = os.path.join(plan["sim_dir"], scenario, name, stem + suffix + ".root")
                if output in outputs:
                    raise PlanError("%s and %s would both be simulated to %s"
                                    % (outputs[output], path, output))
                outputs[output] = path
                exists = os.path.exists(output)
                if exists and not plan["force"]:
                    done += 1
                elif not os.path.isfile(path):
                    missing[path] = scenario
                    absent += 1
                else:
                    (redo if exists else new).append((path, output))

            counts = []
            if done:
                counts.append("%d already simulated, so skipped" % done)
            if new:
                counts.append("%d to simulate" % len(new))
            if redo:
                counts.append("%d to simulate again (--force)" % len(redo))
            if absent:
                counts.append("%d not made yet" % absent)
            say("%s on %s, %d input%s: %s" % (name, scenario, len(files),
                                             "" if len(files) == 1 else "s", ", ".join(counts)))
            for path, output in new + redo:
                if dry_run:
                    say("  would simulate %s" % output)
                jobs.append((name, scenario, path, output, compact_files[name],
                             ",".join(model["presets"]), str(events or -1)))

    if missing:
        made_by_scenarios = sorted({s for s in missing.values() if s != CUSTOM_SCENARIO})
        say("these inputs don't exist%s:" % (
            "; make them with ./generate_input.sh " + " ".join(made_by_scenarios)
            if made_by_scenarios else ""))
        for path in missing:
            say("  " + path)
        if not dry_run:
            raise PlanError("some inputs are missing (see above), so nothing was simulated")

    for job in jobs:
        for field in job:
            if not field or "\t" in field or "\n" in field:
                raise PlanError("empty fields, tabs and line breaks aren't supported: %r"
                                % (job,))
    return jobs


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Work out the simulations that simulate_models.sh runs.",
        epilog="See the module docstring for the steps.")
    steps = parser.add_subparsers(dest="step")
    plan_parser = steps.add_parser("plan", help="print the plan as JSON")
    plan_parser.add_argument("models", nargs="*", metavar="MODEL",
                             help="models in models/ (default: all of them)")
    plan_parser.add_argument("--scenario", action="append", default=[], metavar="NAME",
                             help="simulate this scenario (repeatable)")
    plan_parser.add_argument("--input", action="append", default=[], metavar="FILE",
                             help="simulate this file, as the scenario 'custom' (repeatable)")
    plan_parser.add_argument("--file", metavar="FILE",
                             help="simulate only this input of the one --scenario given")
    plan_parser.add_argument("--events", type=int, metavar="N",
                             help="simulate only the first N events of each input")
    plan_parser.add_argument("--force", action="store_true",
                             help="simulate again where the output exists")
    jobs_parser = steps.add_parser("jobs", help="read the plan on stdin and print the jobs")
    jobs_parser.add_argument("--dry-run", action="store_true",
                             help="also list what would be simulated, and allow missing inputs")
    args = parser.parse_args(argv)

    try:
        if args.step == "plan":
            plan = make_plan(args.models, args.scenario, args.input, args.events, args.force,
                             args.file)
            report_plan(plan)
            print(json.dumps(plan))
        elif args.step == "jobs":
            plan = json.load(sys.stdin)
            # stdout is for the jobs only, but what this imports can print
            # (pyLCIO and ROOT do), so send everything else to stderr, at the
            # level of the file descriptor to catch C++ output too.
            sys.stdout.flush()
            stdout = os.dup(1)
            os.dup2(2, 1)
            try:
                jobs = make_jobs(plan, args.dry_run)
            finally:
                sys.stdout.flush()
                os.dup2(stdout, 1)
                os.close(stdout)
            for job in jobs:
                print("\t".join(job))
        else:
            parser.error("choose a step: plan or jobs")
    except (PlanError, ConfigError) as error:
        say(str(error))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
