"""The timing benchmark's command line, scripts/ddml-timing.

  ddml-timing run [options] [SIMULATOR ...] [-- SBATCH_OPTION ...]
      Time the simulators on the scenarios' inputs: with --executor slurm, as
      a slurm job each, on the kinds of node of timing.devices
      (slurm_timing.sh); with --executor local, here, one task after another
      (timing.sh). See slurm_timing.sh --help for the options.

  ddml-timing summary [options]
      Tabulate the results (timing_summary.sh); needs only Python.

Outside the environment from setup.sh, run works out the tasks where it is,
then carries on in that environment (scripts/run_in_environment.sh) with these
steps, which it runs itself:

  ddml-timing prepare PLAN TASKS     find what is done, and check that this
                                     environment can run the rest
  ddml-timing local PLAN [--dry-run] find what is done, and time the rest here
  ddml-timing run-task DIR KEY       in a slurm job: time one task of
                                     DIR/tasks.json
"""

import argparse
import json
import os
import shutil
import sys

from ddml_bench import process
from ddml_bench.cli import UsageError, parse, positive_int, split_dashes, use_config_files
from ddml_bench.config import ConfigError, load_config
from ddml_bench.environment import in_environment
from ddml_bench.layout import LayoutError
from ddml_bench.models import ModelError

from .executors import ExecutorError
from .plan import PlanError, make_plan, timing_dir
from .runner import TimingError
from .settings import SettingsError

ERRORS = (ConfigError, PlanError, ExecutorError, TimingError, ModelError, LayoutError,
          SettingsError)


def say(message):
    print("ddml-timing: " + message, file=sys.stderr)


def whole(text):
    if not text.isdigit():
        raise argparse.ArgumentTypeError("needs a whole number, at least 0, not '%s'" % text)
    return int(text)


def command_run(argv):
    before, sbatch_args = split_dashes(argv)
    parser = argparse.ArgumentParser(
        prog="ddml-timing run", description="Time the simulators on the scenarios' inputs.")
    parser.add_argument("simulators", nargs="*", metavar="SIMULATOR",
                        help="only these simulators: models in models/, or geant4")
    parser.add_argument("-s", "--scenario", action="append", default=[], metavar="NAME",
                        help="only this scenario (repeatable)")
    parser.add_argument("-d", "--device", action="append", default=[], metavar="NODE",
                        help="only this kind of node of timing.devices (repeatable; default: "
                             "all of them with slurm, cpu here)")
    parser.add_argument("--file", action="append", default=[], metavar="FILE",
                        help="only this input of the scenarios (repeatable)")
    parser.add_argument("-n", "--events", type=positive_int, metavar="N",
                        help="time N events of each input (default: timing.events)")
    parser.add_argument("--warmup", type=whole, metavar="N",
                        help="after N events of warm-up (default: timing.warmup_events)")
    parser.add_argument("-f", "--force", action="store_true",
                        help="time again what is timed already")
    parser.add_argument("--executor", choices=("local", "slurm"), default="local",
                        help="time the tasks here, one after another, or as slurm jobs")
    parser.add_argument("-c", "--config", action="append", default=[], metavar="FILE",
                        help="merge FILE over the configuration, for this run and its jobs "
                             "(repeatable)")
    parser.add_argument("--dry-run", action="store_true",
                        help="show what is timed and what would be run, then stop")
    args = parse(parser, before, "simulators")
    if sbatch_args and args.executor != "slurm":
        parser.error("options after -- go to sbatch, so they need --executor slurm")
    use_config_files(args.config)
    devices = args.device or ([] if args.executor == "slurm" else ["cpu"])
    plan = make_plan(simulators=args.simulators, scenarios=args.scenario, devices=devices,
                     files=args.file, events=args.events, warmup=args.warmup, force=args.force)
    if not plan["tasks"]:
        say("nothing to time: no task has the simulators, scenarios and kinds of node given")
        return 1
    from . import executors
    if args.executor == "slurm":
        return executors.run_slurm(plan, args.dry_run, sbatch_args)
    if any(task["inference"] == "cuda" for task in plan["tasks"]):
        # The GPU goes into the container too, if there is one (apptainer --nv)
        os.environ.setdefault("APPTAINER_NV" if shutil.which("apptainer") else "SINGULARITY_NV",
                              "1")
    if in_environment():
        return executors.run_local(plan, args.dry_run)
    # The plan goes in through a file in the timing folder, which is mounted
    folder = plan["timing_dir"]
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, ".plan.%s.%d.json" % (process.host(), os.getpid()))
    with open(path, "w") as stream:
        json.dump(plan, stream)
    try:
        return executors.run_in_environment(["local", path] + (["--dry-run"] if args.dry_run
                                                               else []))
    finally:
        process.remove(path)


def command_summary(argv):
    parser = argparse.ArgumentParser(
        prog="ddml-timing summary",
        description="Tabulate the timing results: <paths.data_dir>/timing/summary.csv.")
    parser.add_argument("-d", "--dir", metavar="DIR",
                        help="the timing results' folder (default: paths.data_dir's timing/), "
                             "e.g. copied from elsewhere")
    parser.add_argument("-o", "--output", metavar="FILE",
                        help="write the table here (default: summary.csv in the results' folder)")
    parser.add_argument("-n", "--events", type=positive_int, metavar="N",
                        help="only the results of N events timed")
    parser.add_argument("-c", "--config", action="append", default=[], metavar="FILE",
                        help="merge FILE over the configuration (repeatable)")
    args = parser.parse_args(argv)
    from .summary import main
    if args.dir:
        root = args.dir
    else:
        use_config_files(args.config)
        root = timing_dir(load_config()["paths"]["data_dir"])
    if not os.path.isdir(root):
        raise UsageError("there are no timing results in %s yet (slurm_timing.sh makes them)"
                         % root)
    return main(root, args.output, args.events)


def command_prepare(argv):
    parser = argparse.ArgumentParser(prog="ddml-timing prepare")
    parser.add_argument("plan")
    parser.add_argument("tasks")
    args = parser.parse_args(argv)
    from .executors import prepare
    prepare(args.plan, args.tasks)
    return 0


def command_local(argv):
    parser = argparse.ArgumentParser(prog="ddml-timing local")
    parser.add_argument("plan")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    from .executors import run_local
    with open(args.plan) as stream:
        plan = json.load(stream)
    return run_local(plan, args.dry_run)


def command_run_task(argv):
    parser = argparse.ArgumentParser(prog="ddml-timing run-task")
    parser.add_argument("directory", metavar="DIR")
    parser.add_argument("key", metavar="KEY")
    args = parser.parse_args(argv)
    from .executors import run_task_in_job
    run_task_in_job(args.directory, args.key)
    return 0


COMMANDS = {
    "run": command_run,
    "summary": command_summary,
    "prepare": command_prepare,
    "local": command_local,
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
