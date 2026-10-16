#!/usr/bin/env python3
"""Work out the reconstructions that reconstruct_simulation.sh runs.

Every finished simulation of the benchmark scenarios in the configuration, by
every simulator (each model in models/ and Geant4), is reconstructed with ILD's
standard reconstruction, ILDConfig's ILDReconstruction.py, run with k4run. The
simulations are the files simulate_models.sh and simulate_geant4.sh write,

  <paths.data_dir>/sim_output/<scenario>/<simulator>/<name>.root
  <paths.data_dir>/sim_output/<scenario>/geant4/<name>.edm4hep.root

and each is reconstructed to

  <paths.data_dir>/rec_output/<scenario>/<simulator>/<name>_REC.edm4hep.root

with the other outputs of ILDReconstruction.py (<name>_AIDA.root,
<name>_PfoAnalysis.root) and the log (<name>.log) next to it, or <name>_n<N>...
when only the first N events are reconstructed. A simulation file given
directly is reconstructed to the same place if it is in sim_output, and to
rec_output/custom/files/ otherwise. A reconstruction whose _REC file exists is
done, and is skipped unless forced.

Each simulation is reconstructed in the detector it was simulated in: the
model's fixed_at_training.detector.compact_file, or geant4.compact.

This runs in two steps:

  reconstruction_jobs.py plan [--scenario NAME] [--compact FILE] [--events N]
                              [--force] [SIMULATOR_OR_FILE ...]
      Read the configuration, look for the simulations, and print the plan as
      JSON. Arguments are simulators (a model in models/, or geant4) or
      simulation files; without simulators, every simulator is reconstructed.
      Without --scenario, every scenario in the configuration is (sim_output/
      custom only when named).

  reconstruction_jobs.py jobs [--dry-run] < PLAN
      In the environment from setup.sh: check it, and print the
      reconstructions still to run, one per line, with these fields separated
      by tabs: scenario, simulator, input, output base (outputs are named
      <base>_REC.edm4hep.root and so on), compact file, the number of events
      for k4run (-1 for all), the file that marks it done, and the arguments for
      ILDReconstruction.py as one shell-quoted string. What is run and skipped
      is reported on stderr.
"""

import argparse
import json
import os
import re
import shlex
import shutil
import sys

from fasinate_config import ConfigError, load_config

GEANT4 = "geant4"
# The scenario that simulations given outside sim_output/ are reconstructed as,
# and their folder in it
CUSTOM_SCENARIO = "custom"
GIVEN_FILES = "files"
# What ddsim writes: EDM4hep, by the name's ending. LCIO (.slcio) simulations
# are left out, as nothing here writes them.
SIM_EXTENSIONS = (".edm4hep.root", ".root")
# The script in ILDConfig, and the folder it runs in
ILDCONFIG_PRODUCTION = os.path.join("StandardConfig", "production")
RECO_SCRIPT = "ILDReconstruction.py"
# Arguments of ILDReconstruction.py that reconstruct_simulation.sh sets itself
MANAGED_ARGS = ("--inputFiles", "--outputFileBase", "--compactFile", "--detectorModel",
                "--num-events", "--evt-max", "-n")
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+-]*$")


class PlanError(Exception):
    pass


def say(message):
    print("reconstruct_simulation: " + message, file=sys.stderr)


def _simulation_stem(file_name):
    """The name of a simulation file without its extension, or None."""
    if file_name.startswith("."):  # ddsim's unfinished outputs
        return None
    for extension in SIM_EXTENSIONS:
        if file_name.endswith(extension) and len(file_name) > len(extension):
            return file_name[:-len(extension)]
    return None


def _simulations_in(directory):
    """The finished simulations in a folder, sorted."""
    if not os.path.isdir(directory):
        return []
    found = []
    for file_name in sorted(os.listdir(directory)):
        path = os.path.join(directory, file_name)
        if _simulation_stem(file_name) and os.path.isfile(path):
            found.append(path)
    return found


def _model_compact_files():
    """{model: compact file} for the models in models/."""
    import simulation_jobs
    found = {}
    for name in simulation_jobs.model_names():
        try:
            found[name] = simulation_jobs.load_model(name)["compact_file"]
        except simulation_jobs.PlanError as error:
            raise PlanError(str(error))
    return found


def _extra_args(config):
    """ILDReconstruction.py's arguments from reconstruction.extra_args."""
    extra = config["reconstruction"]["extra_args"]
    if extra is None:
        extra = []
    if not isinstance(extra, list) or any(isinstance(a, (dict, list)) for a in extra):
        raise PlanError("reconstruction.extra_args should be a list of arguments")
    args = ["true" if a is True else "false" if a is False else str(a) for a in extra]
    for arg in args:
        if arg.split("=", 1)[0] in MANAGED_ARGS:
            raise PlanError("reconstruction.extra_args can't include %s: "
                            "reconstruct_simulation.sh sets it" % arg)
    return args


def _done_suffix(extra_args):
    """The output that marks a reconstruction done."""
    lcio = None
    for i, arg in enumerate(extra_args):
        if arg == "--lcioOutput" and i + 1 < len(extra_args):
            lcio = extra_args[i + 1]
        elif arg.startswith("--lcioOutput="):
            lcio = arg.split("=", 1)[1]
    return "_REC.slcio" if lcio == "only" else "_REC.edm4hep.root"


def make_plan(targets=(), scenarios=(), compact=None, events=None, force=False):
    """The reconstructions to consider, as a JSON-able dict."""
    config = load_config()
    data_dir = config["paths"]["data_dir"]
    sim_dir = os.path.join(data_dir, "sim_output")
    rec_dir = os.path.join(data_dir, "rec_output")
    if events is not None and events < 1:
        raise PlanError("the number of events must be at least 1, not %d" % events)
    extra = _extra_args(config)

    compact_files = _model_compact_files()
    if GEANT4 in compact_files:
        raise PlanError("models/%s.yaml: '%s' names the Geant4 reference, so it can't be a "
                        "model name" % (GEANT4, GEANT4))
    compact_files[GEANT4] = config["geant4"]["compact"]

    # Which scenarios are particle guns, for ILDReconstruction.py's --usingParticleGun
    kinds = {name: (settings or {}).get("input", {}).get("type")
             for name, settings in config["scenarios"].items()}
    for name in scenarios:
        if name not in kinds and name != CUSTOM_SCENARIO:
            raise PlanError("there is no scenario '%s'; the configuration has: %s (and %s, "
                            "for what was simulated with --input)"
                            % (name, ", ".join(kinds) or "none", CUSTOM_SCENARIO))
        if not _NAME.match(name):
            raise PlanError("'%s' isn't a scenario name" % name)

    simulators, files = [], []
    for target in targets:
        if os.path.isfile(target):
            path = os.path.realpath(target)
            if not _simulation_stem(os.path.basename(path)):
                raise PlanError("%s isn't a finished simulation: an EDM4hep file, named "
                                "*.root or *.edm4hep.root (not hidden)" % target)
            if path not in files:
                files.append(path)
        elif target in compact_files:
            if target not in simulators:
                simulators.append(target)
        else:
            raise PlanError("'%s' is neither a simulation file nor a simulator; the "
                            "simulators are: %s" % (target, ", ".join(compact_files)))

    groups = {}  # (scenario, simulator): group

    def group(scenario, simulator, folder=None):
        key = (scenario, simulator)
        if key not in groups:
            known = simulator in compact_files
            if compact:
                detector = compact
            elif known:
                detector = compact_files[simulator]
            else:
                detector = compact_files[GEANT4]
            if not compact and not known and simulator != GIVEN_FILES:
                say("warning: %s isn't a model in models/ or geant4, so its simulations are "
                    "reconstructed in geant4.compact, %s (use --compact to choose)"
                    % (simulator, detector))
            groups[key] = {
                "scenario": scenario,
                "simulator": simulator,
                "folder": folder or os.path.join(rec_dir, scenario, simulator),
                "compact": detector,
                "particle_gun": kinds.get(scenario) == "particle_gun",
                "files": [],
            }
        return groups[key]

    # Everything the configuration indicates, or the simulators and scenarios
    # named, unless only files were given
    if simulators or scenarios or not files:
        for scenario in scenarios or list(kinds):
            for simulator in simulators or list(compact_files):
                found = _simulations_in(os.path.join(sim_dir, scenario, simulator))
                group(scenario, simulator)["files"].extend(found)

    # Files given: placed by where they are in sim_output, if they are
    real_sim_dir = os.path.realpath(sim_dir)
    for path in files:
        parts = os.path.relpath(path, real_sim_dir).split(os.sep)
        if len(parts) == 3 and parts[0] != os.pardir:
            entry = group(parts[0], parts[1])
        else:
            entry = group(CUSTOM_SCENARIO, GIVEN_FILES,
                          os.path.join(rec_dir, CUSTOM_SCENARIO, GIVEN_FILES))
        if path not in entry["files"]:
            entry["files"].append(path)

    return {
        "sim_dir": sim_dir,
        "rec_dir": rec_dir,
        "ildconfig_dir": os.path.join(config["detector_configs"]["ildconfig"]["source_dir"],
                                      ILDCONFIG_PRODUCTION),
        "extra_args": extra,
        "events": events,
        "force": force,
        "groups": list(groups.values()),
    }


def report_plan(plan):
    total = 0
    for entry in plan["groups"]:
        count = len(entry["files"])
        total += count
        if count:
            say("%s on %s: %d simulation%s" % (entry["simulator"], entry["scenario"], count,
                                               "" if count == 1 else "s"))
    if not total:
        say("found no finished simulations to reconstruct in %s" % plan["sim_dir"])
    if plan["events"] is not None:
        say("reconstructing only the first %d events of each" % plan["events"])


# ---------------------------------------------------------------------------
# The jobs, worked out in the environment from setup.sh


def check_environment(plan):
    """Check that this environment can run ILDReconstruction.py, and return
    {compact file as in the plan: its path}."""
    problems = []
    if not shutil.which("k4run"):
        problems.append("k4run isn't on PATH; is this the environment from setup.sh?")
    script = os.path.join(plan["ildconfig_dir"], RECO_SCRIPT)
    if not os.path.isfile(script):
        problems.append("%s not found; get ILDConfig with scripts/fetch_detector_configs.sh"
                        % script)
    k4geo = os.environ.get("k4geo_DIR") or os.environ.get("K4GEO")
    compact_files = {}
    for entry in plan["groups"]:
        compact = entry["compact"]
        if compact in compact_files:
            continue
        path = compact
        if not os.path.isabs(path):
            if not k4geo:
                problems.append("k4geo_DIR isn't set, so the compact file %s can't be found"
                                % compact)
            path = os.path.join(k4geo or "", compact)
        if not os.path.isfile(path):
            problems.append("compact file not found: %s" % path)
        compact_files[compact] = path
    if problems:
        raise PlanError("this environment can't run the reconstruction:\n  "
                        + "\n  ".join(problems))
    return compact_files


def reco_args(plan, entry):
    """Arguments for ILDReconstruction.py, besides the input, output and detector."""
    args = list(plan["extra_args"])
    if not {"--runBeamCalReco", "--noBeamCalReco"} & set(args):
        args.append("--noBeamCalReco")
    if entry["particle_gun"] and "--usingParticleGun" not in args:
        args.append("--usingParticleGun")
    return args


def make_jobs(plan, dry_run=False):
    """The reconstructions still to run, as tuples of strings (see the docstring)."""
    compact_files = check_environment(plan)
    events = plan["events"]
    suffix = "" if events is None else "_n%d" % events
    done_suffix = _done_suffix(plan["extra_args"])
    jobs = []
    outputs = {}
    for entry in plan["groups"]:
        if not entry["files"]:
            continue
        args = " ".join(shlex.quote(a) for a in reco_args(plan, entry))
        done, redo, new, gone = 0, [], [], 0
        for path in entry["files"]:
            base = os.path.join(entry["folder"],
                                _simulation_stem(os.path.basename(path)) + suffix)
            if base in outputs:
                raise PlanError("%s and %s would both be reconstructed to %s"
                                % (outputs[base], path, base))
            outputs[base] = path
            exists = os.path.exists(base + done_suffix)
            if exists and not plan["force"]:
                done += 1
            elif not os.path.isfile(path):
                gone += 1
            else:
                (redo if exists else new).append((path, base))

        counts = []
        if done:
            counts.append("%d already reconstructed, so skipped" % done)
        if new:
            counts.append("%d to reconstruct" % len(new))
        if redo:
            counts.append("%d to reconstruct again (--force)" % len(redo))
        if gone:
            counts.append("%d no longer there" % gone)
        say("%s on %s, %d simulation%s: %s" % (
            entry["simulator"], entry["scenario"], len(entry["files"]),
            "" if len(entry["files"]) == 1 else "s", ", ".join(counts)))
        for path, base in new + redo:
            if dry_run:
                say("  would reconstruct %s%s" % (base, done_suffix))
            jobs.append((entry["scenario"], entry["simulator"], path, base,
                         compact_files[entry["compact"]], str(events or -1),
                         base + done_suffix, args))

    for job in jobs:
        for field in job[:-1]:
            if not field or "\t" in field or "\n" in field:
                raise PlanError("empty fields, tabs and line breaks aren't supported: %r"
                                % (job,))
        if "\t" in job[-1] or "\n" in job[-1]:
            raise PlanError("tabs and line breaks aren't supported in "
                            "reconstruction.extra_args")
    return jobs


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Work out the reconstructions that reconstruct_simulation.sh runs.",
        epilog="See the module docstring for the steps.")
    steps = parser.add_subparsers(dest="step")
    plan_parser = steps.add_parser("plan", help="print the plan as JSON")
    plan_parser.add_argument("targets", nargs="*", metavar="SIMULATOR_OR_FILE",
                             help="simulators (models in models/, or geant4) or simulation "
                                  "files (default: every simulator)")
    plan_parser.add_argument("--scenario", action="append", default=[], metavar="NAME",
                             help="reconstruct this scenario's simulations (repeatable)")
    plan_parser.add_argument("--compact", metavar="FILE",
                             help="reconstruct in this detector instead (relative to k4geo "
                                  "unless absolute)")
    plan_parser.add_argument("--events", type=int, metavar="N",
                             help="reconstruct only the first N events of each simulation")
    plan_parser.add_argument("--force", action="store_true",
                             help="reconstruct again where the output exists")
    jobs_parser = steps.add_parser("jobs", help="read the plan on stdin and print the jobs")
    jobs_parser.add_argument("--dry-run", action="store_true",
                             help="also list what would be reconstructed")
    args = parser.parse_args(argv)

    try:
        if args.step == "plan":
            plan = make_plan(args.targets, args.scenario, args.compact, args.events, args.force)
            report_plan(plan)
            print(json.dumps(plan))
        elif args.step == "jobs":
            plan = json.load(sys.stdin)
            jobs = make_jobs(plan, args.dry_run)
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
