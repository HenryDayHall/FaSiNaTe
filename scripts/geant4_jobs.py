#!/usr/bin/env python3
"""Work out the simulations that simulate_geant4.sh runs.

Full Geant4, the reference that the models are compared to, simulates the input
of every benchmark scenario in the configuration, or of the ones named, or the
files given. Each simulation writes

  <paths.data_dir>/sim_output/<scenario>/geant4/<input name>.edm4hep.root

or <input name>_n<N>.edm4hep.root when only the first N events are simulated;
files given with --input are simulated as the scenario "custom". A simulation
whose output exists is done, and is skipped unless forced.

This runs in two steps:

  geant4_jobs.py plan [--scenario NAME] [--input FILE] [--file FILE]
                      [--events N] [--force]
      Print the plan as JSON. With --file, it is just that input, as part of
      the one scenario given; that needs no configuration files, only the
      FASINATE_* variables, so it also works in the container. Otherwise it
      is the inputs of the scenarios (every scenario unless --scenario or
      --input is given) and the files given with --input.

  geant4_jobs.py jobs [--dry-run] < PLAN
      In the environment from setup.sh: check it, name the scenarios' input
      files, and print the simulations still to run, one per line, with these
      fields separated by tabs: scenario, input, output, log, compact file and
      the number of events for ddsim (-1 for all). What is run and skipped is
      reported on stderr.
"""

import argparse
import json
import os
import re
import shutil
import sys

from fasinate_config import REPO_DIR, ConfigError, load_config
from scenario_inputs import INPUT_FILES, scenario_commands

# What the outputs are named after: their folder in sim_output/<scenario>/
NAME = "geant4"
# ddsim writes EDM4hep for a .root output file
OUTPUT_EXTENSION = ".edm4hep.root"
# The scenario that files given with --input are simulated as
CUSTOM_SCENARIO = "custom"
# Scenario names become folder names.
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+-]*$")


class PlanError(Exception):
    pass


def say(message):
    print("simulate_geant4: " + message, file=sys.stderr)


def _setting(var, what):
    value = os.environ.get(var)
    if not value:
        raise PlanError("%s isn't set, so the configuration isn't loaded; run this through "
                        "simulate_geant4.sh (%s)" % (var, what))
    return value


def _check_name(scenario):
    if not _NAME.match(scenario):
        raise PlanError("scenario names become folder names, so use only letters, digits "
                        "and _.+- in '%s'" % scenario)


def _model_detectors():
    """{model: compact file} from models/*.yaml, for a consistency check."""
    try:
        import yaml
    except ImportError:
        return {}
    models_dir = os.path.join(REPO_DIR, "models")
    found = {}
    for file_name in sorted(os.listdir(models_dir)) if os.path.isdir(models_dir) else []:
        if not file_name.endswith(".yaml"):
            continue
        try:
            with open(os.path.join(models_dir, file_name)) as stream:
                data = yaml.safe_load(stream)
            found[file_name[:-len(".yaml")]] = data["fixed_at_training"]["detector"]["compact_file"]
        except (OSError, yaml.YAMLError, KeyError, TypeError):
            continue
    return found


def make_plan(scenarios=(), inputs=(), file=None, events=None, force=False):
    """The simulations to consider, as a JSON-able dict."""
    sim_dir = os.path.join(_setting("FASINATE_PATHS_DATA_DIR", "paths.data_dir"), "sim_output")
    compact = _setting("FASINATE_GEANT4_COMPACT", "geant4.compact")
    if events is not None and events < 1:
        raise PlanError("the number of events must be at least 1, not %d" % events)

    groups = []
    if file is not None:
        if len(scenarios) != 1 or inputs:
            raise PlanError("--file needs exactly one --scenario, the one the file belongs "
                            "to (custom for files outside the scenarios), and no --input")
        _check_name(scenarios[0])
        if not os.path.isfile(file):
            raise PlanError("input file not found: %s" % file)
        groups.append({"scenario": scenarios[0], "type": "files",
                       "files": [os.path.realpath(file)]})
    else:
        if scenarios or not inputs:
            config = load_config()
            for name, kind, commands in scenario_commands(config, scenarios):
                _check_name(name)
                if kind not in INPUT_FILES:
                    raise PlanError("scenario %s: there's no way to tell which files a %s "
                                    "input makes; add one to INPUT_FILES in "
                                    "scripts/scenario_inputs.py" % (name, kind))
                groups.append({"scenario": name, "type": kind, "commands": commands})
            if CUSTOM_SCENARIO in config["scenarios"] and inputs:
                raise PlanError("files given with --input are simulated as the scenario "
                                "'%s', but the configuration has a scenario of that name"
                                % CUSTOM_SCENARIO)
        if inputs:
            files = []
            for path in inputs:
                if not os.path.isfile(path):
                    raise PlanError("input file not found: %s" % path)
                # The real path, as that is what gets mounted in the container
                path = os.path.realpath(path)
                if path not in files:
                    files.append(path)
            groups.append({"scenario": CUSTOM_SCENARIO, "type": "files", "files": files})
        for model, model_compact in _model_detectors().items():
            if model_compact != compact:
                say("warning: models/%s.yaml simulates in %s, but geant4.compact is %s; "
                    "the comparison needs the same detector" % (model, model_compact, compact))

    return {"sim_dir": sim_dir, "compact": compact, "events": events, "force": force,
            "inputs": groups}


def report_plan(plan):
    for group in plan["inputs"]:
        count = len(group["files"] if group["type"] == "files" else group["commands"])
        say("scenario %s: %d input file%s%s" % (
            group["scenario"], count, "" if count == 1 else "s",
            " given" if group["type"] == "files" else ", from its %s" % group["type"]))
    if plan["events"] is not None:
        say("simulating only the first %d events of each" % plan["events"])


def check_environment(plan):
    """Check that this is the environment from setup.sh, with the DDML build
    from scripts/build_ddml.sh, whose steering file the Geant4 steering runs.
    Returns the compact file."""
    env = os.environ
    source = env.get("DDML_SOURCE_DIR")
    if not source:
        raise PlanError("DDML isn't set up in this environment; source setup.sh first")
    problems = []
    expected = env.get("FASINATE_DDML_SOURCE_DIR")
    if not (expected and os.path.realpath(source) == os.path.realpath(expected)):
        problems.append("DDML_SOURCE_DIR is %s, but scripts/build_ddml.sh builds DDML in %s "
                        "for this configuration" % (source, expected or "(unset)"))
    for steering in (os.path.join(source, "scripts", "ddsim_steer.py"),
                     os.path.join(REPO_DIR, "scripts", "ddsim_steer_geant4.py")):
        if not os.path.isfile(steering):
            problems.append("steering file not found: %s" % steering)
    if not shutil.which("ddsim"):
        problems.append("ddsim isn't on PATH")
    compact = plan["compact"]
    if not os.path.isabs(compact):
        k4geo = env.get("k4geo_DIR") or env.get("K4GEO")
        if not k4geo:
            problems.append("k4geo_DIR isn't set, so the compact file %s can't be found" % compact)
        compact = os.path.join(k4geo or "", compact)
    if not os.path.isfile(compact):
        problems.append("compact file not found: %s" % compact)
    if problems:
        raise PlanError("this environment can't run the simulation:\n  " + "\n  ".join(problems))
    return compact


def make_jobs(plan, dry_run=False):
    """The simulations still to run, as tuples of strings (see the docstring)."""
    compact = check_environment(plan)

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
    for scenario, files in inputs:
        done, redo, new, absent = 0, [], [], 0
        for path in files:
            stem = os.path.splitext(os.path.basename(path))[0] + suffix
            base = os.path.join(plan["sim_dir"], scenario, NAME, stem)
            output = base + OUTPUT_EXTENSION
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
                (redo if exists else new).append((path, output, base + ".log"))

        counts = []
        if done:
            counts.append("%d already simulated, so skipped" % done)
        if new:
            counts.append("%d to simulate" % len(new))
        if redo:
            counts.append("%d to simulate again (--force)" % len(redo))
        if absent:
            counts.append("%d not made yet" % absent)
        say("Geant4 on %s, %d input%s: %s" % (scenario, len(files),
                                             "" if len(files) == 1 else "s", ", ".join(counts)))
        for path, output, log in new + redo:
            if dry_run:
                say("  would simulate %s" % output)
            jobs.append((scenario, path, output, log, compact, str(events or -1)))

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
        description="Work out the simulations that simulate_geant4.sh runs.",
        epilog="See the module docstring for the steps.")
    steps = parser.add_subparsers(dest="step")
    plan_parser = steps.add_parser("plan", help="print the plan as JSON")
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
            plan = make_plan(args.scenario, args.input, args.file, args.events, args.force)
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
