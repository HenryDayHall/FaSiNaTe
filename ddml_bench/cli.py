"""The harness's command line, scripts/ddml-bench.

  ddml-bench run [options] [SIMULATOR ...] [-- SBATCH_OPTION ...]
      Run the pipeline, or some of its stages, with an executor: local, one
      task after another (pipeline.sh), or slurm, a job per task
      (slurm_pipeline.sh). See pipeline.sh --help for the options.

  ddml-bench stage STAGE [options] [ARGUMENT ...]
      Run one stage locally, with the options of its script: generate
      (generate_input.sh), geant4 (simulate_geant4.sh), models
      (simulate_models.sh), reconstruct (reconstruct_simulation.sh),
      observables (observables_from_reconstruction.sh) or plot.

  ddml-bench plot [options] [SIMULATOR ...]
      Plot measurements (plot_observables.sh); needs only numpy and
      matplotlib.

  ddml-bench manifest --out OUT [--stage] [options]
      Write the manifests of the data folder's records for publishing it
      (inputs, measurements, bulk), and with --stage the files to upload;
      needs only numpy (ddml_bench/manifest.py).

Outside the environment from setup.sh, run and stage work out the plan where
they are, then carry on in that environment (scripts/run_in_environment.sh)
with these steps, which they run themselves:

  ddml-bench local PLAN [--dry-run]   find what is done, and run the rest
  ddml-bench tasks PLAN TASKS         find what is done, and check that this
                                      environment can run the rest
  ddml-bench run-task DIR KEY         in a slurm job: run one task of
                                      DIR/tasks.json
"""

import argparse
import json
import os
import sys

from . import process
from .config import ConfigError
from .environment import in_environment, run_in_environment
from .executors import ExecutorError
from .fetch import FetchError
from .graph import PlanError, make_plan
from .layout import LayoutError
from .measure import ObservableError
from .models import ModelError, model_names
from .stages import STAGE_NAMES, StageError

ERRORS = (ConfigError, PlanError, StageError, ExecutorError, ModelError, LayoutError,
          ObservableError, FetchError)


class UsageError(Exception):
    pass


def say(message):
    print("ddml-bench: " + message, file=sys.stderr)


def positive_int(text):
    if not text.isdigit() or int(text) < 1:
        raise argparse.ArgumentTypeError("needs a whole number, at least 1, not '%s'" % text)
    return int(text)


def split_dashes(argv):
    """The arguments before --, and those after it."""
    if "--" in argv:
        index = argv.index("--")
        return argv[:index], argv[index + 1:]
    return argv, []


def parse(parser, argv, positional):
    """Parse argv, allowing positional arguments between the options (as the
    shell scripts always have), into the list attribute named positional."""
    args, rest = parser.parse_known_args(argv)
    unknown = [arg for arg in rest if arg.startswith("-")]
    if unknown:
        parser.error("unknown option '%s' (see --help)" % unknown[0])
    if rest:
        if positional is None:
            parser.error("unexpected argument '%s' (see --help)" % rest[0])
        getattr(args, positional).extend(rest)
    return args


def common_options(parser):
    parser.add_argument("-c", "--config", action="append", default=[], metavar="FILE",
                        help="merge FILE over the configuration, for this run and its jobs "
                             "(repeatable)")
    parser.add_argument("--dry-run", action="store_true",
                        help="show what is done and what would be run, then stop")
    parser.add_argument("--no-fetch", action="store_true",
                        help="make everything here, fetching nothing that is published "
                             "(published in the configuration)")


def use_config_files(files):
    """Merge the configuration files given over the configuration: they go,
    by their full paths, after those in FASINATE_CONFIG, which every job and
    every step in the environment reads."""
    entries = [os.path.abspath(f) for f in os.environ.get("FASINATE_CONFIG", "").split(":") if f]
    for path in files:
        if not os.path.isfile(path):
            raise UsageError("configuration file not found: %s" % path)
        entries.append(os.path.realpath(path))
    if entries:
        os.environ["FASINATE_CONFIG"] = ":".join(entries)
        print("ddml-bench: with the configuration in %s" % " ".join(entries))


def folders_of(paths):
    """The folders of files given, for mounting in the container."""
    found = []
    for path in paths:
        if path:
            folder = os.path.dirname(os.path.realpath(path))
            if folder not in found:
                found.append(folder)
    return found


def fetch_published(plan, args):
    """Fetch the published files that the plan's tasks would make, before
    anything works out what is done (ddml_bench/fetch.py); with --dry-run,
    note them in the plan instead."""
    from .fetch import prepare
    if args.no_fetch:
        plan["published"]["enabled"] = False
    prepare(plan, args.dry_run)


def run_locally(plan, dry_run=False, binds=()):
    """The local executor: in this environment if it is the one from setup.sh,
    or else in that environment."""
    if in_environment():
        from .executors import local
        from .graph import Graph
        return local.run(Graph(plan).find_status(), dry_run)
    # The plan goes in through a file in the data folder, which is mounted
    folder = os.path.join(plan["data_dir"], ".ddml-bench")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "plan.%s.%d.json" % (process.host(), os.getpid()))
    with open(path, "w") as stream:
        json.dump(plan, stream)
    try:
        return run_in_environment(["local", path] + (["--dry-run"] if dry_run else []), binds)
    finally:
        process.remove(path)


# ---------------------------------------------------------------------------
# The commands


def command_run(argv):
    before, sbatch_args = split_dashes(argv)
    parser = argparse.ArgumentParser(
        prog="ddml-bench run", description="Run the pipeline, or some of its stages.")
    parser.add_argument("simulators", nargs="*", metavar="SIMULATOR",
                        help="only these simulators: models in models/, geant4, or geant4_2, "
                             "its second sample")
    parser.add_argument("-s", "--scenario", action="append", default=[], metavar="NAME",
                        help="only this scenario (repeatable)")
    parser.add_argument("--stages", action="append", default=[], metavar="LIST",
                        help="only these stages, separated by commas (repeatable)")
    parser.add_argument("-n", "--events", type=positive_int, metavar="N",
                        help="a test on the first N events of each input")
    parser.add_argument("-f", "--force", action="append", default=[], metavar="STAGE",
                        help="make the outputs of STAGE again, even where nothing needs them, "
                             "and what is made from them (repeatable, or a list separated by "
                             "commas)")
    parser.add_argument("--executor", choices=("local", "slurm"), default="local",
                        help="run the tasks one after another here, or as slurm jobs")
    common_options(parser)
    args = parse(parser, before, "simulators")
    if sbatch_args and args.executor != "slurm":
        parser.error("options after -- go to sbatch, so they need --executor slurm")
    use_config_files(args.config)
    plan = make_plan(simulators=args.simulators, scenarios=args.scenario, stages=args.stages,
                     events=args.events, force=args.force, slurm=args.executor == "slurm")
    fetch_published(plan, args)
    if args.executor == "slurm":
        from .executors import slurm
        return slurm.run(plan, args.dry_run, sbatch_args)
    return run_locally(plan, args.dry_run)


def stage_parser(name):
    parser = argparse.ArgumentParser(prog="ddml-bench stage " + name,
                                     description="Run the %s stage." % name)
    positional = None
    if name == "generate":
        positional = "scenarios"
        parser.add_argument("scenarios", nargs="*", metavar="SCENARIO",
                            help="only these scenarios (default: all of them)")
    elif name == "models":
        positional = "simulators"
        parser.add_argument("simulators", nargs="*", metavar="MODEL",
                            help="only these models in models/ (default: all of them)")
    elif name == "reconstruct":
        positional = "targets"
        parser.add_argument("targets", nargs="*", metavar="SIMULATOR_OR_FILE",
                            help="only these simulators, or these simulation files")
    elif name == "observables":
        positional = "simulators"
        parser.add_argument("simulators", nargs="*", metavar="SIMULATOR",
                            help="only these simulators: models in models/, geant4, or "
                                 "geant4_2, its second sample")
    if name in ("geant4", "models", "reconstruct"):
        parser.add_argument("-s", "--scenario", action="append", default=[], metavar="NAME",
                            help="only this scenario (repeatable)")
    if name in ("geant4", "models"):
        parser.add_argument("-i", "--input", action="append", default=[], metavar="FILE",
                            help="simulate FILE too, as the scenario custom (repeatable)")
        parser.add_argument("--file", action="append", default=[], metavar="FILE",
                            help="only this input of the scenarios (repeatable)")
    if name != "generate":
        parser.add_argument("-n", "--events", type=positive_int, metavar="N",
                            help="the test sample of the first N events of each input")
    if name == "reconstruct":
        parser.add_argument("--compact", metavar="FILE",
                            help="reconstruct in this detector (relative to k4geo unless "
                                 "absolute)")
    parser.add_argument("-f", "--force", action="store_true",
                        help="make the outputs again where they exist, and where nothing "
                             "needs them")
    common_options(parser)
    return parser, positional


def command_stage(argv):
    if not argv or argv[0] not in STAGE_NAMES:
        raise UsageError("give a stage: %s" % ", ".join(STAGE_NAMES))
    name = argv[0]
    parser, positional = stage_parser(name)
    args = parse(parser, argv[1:], positional)
    use_config_files(args.config)
    simulators, simulations = [], []
    scenarios = getattr(args, "scenario", None) or getattr(args, "scenarios", None) or []
    if name == "models":
        for model in args.simulators:
            if model not in model_names():
                raise UsageError("there is no model '%s'; models/ has: %s"
                                 % (model, ", ".join(model_names()) or "nothing"))
        simulators = args.simulators
    elif name == "observables":
        simulators = args.simulators
    elif name == "reconstruct":
        for target in args.targets:
            (simulations if os.path.isfile(target) else simulators).append(target)
        if simulations and args.events:
            raise UsageError("-n picks the simulations of the first N events of each input "
                             "(simulate_*.sh -n N), so give simulation files without it")
    inputs = getattr(args, "input", [])
    files = getattr(args, "file", [])
    compact = getattr(args, "compact", None)
    plan = make_plan(simulators=simulators, scenarios=scenarios, stages=[name],
                     events=getattr(args, "events", None), force=[name] if args.force else [],
                     inputs=inputs, files=files, simulations=simulations, compact=compact)
    fetch_published(plan, args)
    binds = folders_of(list(inputs) + list(files) + simulations
                       + ([compact] if compact and os.path.isabs(compact) else []))
    return run_locally(plan, args.dry_run, binds)


def command_plot(argv):
    from .plot import main
    return main(argv)


def command_manifest(argv):
    from .manifest import main
    return main(argv)


def command_local(argv):
    parser = argparse.ArgumentParser(prog="ddml-bench local")
    parser.add_argument("plan")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    from .executors import local
    from .graph import Graph
    with open(args.plan) as stream:
        plan = json.load(stream)
    return local.run(Graph(plan).find_status(), args.dry_run)


def command_tasks(argv):
    parser = argparse.ArgumentParser(prog="ddml-bench tasks")
    parser.add_argument("plan")
    parser.add_argument("tasks")
    args = parser.parse_args(argv)
    from .executors import slurm
    slurm.write_tasks(args.plan, args.tasks)
    return 0


def command_run_task(argv):
    parser = argparse.ArgumentParser(prog="ddml-bench run-task")
    parser.add_argument("directory", metavar="DIR")
    parser.add_argument("key", metavar="KEY")
    args = parser.parse_args(argv)
    from .executors import slurm
    slurm.run_one(args.directory, args.key)
    return 0


COMMANDS = {
    "run": command_run,
    "stage": command_stage,
    "plot": command_plot,
    "manifest": command_manifest,
    "local": command_local,
    "tasks": command_tasks,
    "run-task": command_run_task,
}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__.strip())
        return 0 if argv else 2
    command = COMMANDS.get(argv[0])
    if command is None:
        say("there's no command '%s'; the commands are: %s" % (argv[0], ", ".join(COMMANDS)))
        return 2
    try:
        return command(argv[1:]) or 0
    except process.Interrupted as error:
        say("interrupted")
        return error.status
    except KeyboardInterrupt:
        say("interrupted")
        return 130
    except UsageError as error:
        say(str(error))
        return 2
    except ERRORS as error:
        say(str(error))
        return 1
