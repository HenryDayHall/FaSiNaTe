#!/usr/bin/env python3
"""Work out, submit and run the slurm jobs of slurm_pipeline.sh.

slurm_pipeline.sh runs every stage of the benchmark pipeline (README.md) as
slurm jobs of one file each, each job waiting for the jobs that make what it
reads. It does so in these steps:

  pipeline_jobs.py plan [--scenario NAME] [--stages LIST] [--events N]
                        [--force STAGE] [SIMULATOR ...] > PLAN
      Where the configuration files are, before the simulation environment is
      set up: read the configuration and the model descriptions, check them
      and the options, and print the plan as JSON. --scenario, --stages and
      --force can be repeated, and stages given as lists separated by commas.

  pipeline_jobs.py tasks PLAN TASKS
      In the environment from setup.sh, which names the inputs (with pyLCIO)
      and reads the .npz files (with numpy): list the tasks of every stage,
      find which are done from what is on disk, work out what the others wait
      for, check that the environment can run them, and write it all to TASKS,
      as JSON.

  pipeline_jobs.py submit [--dry-run] DIR [-- SBATCH_OPTION ...]
      On a slurm login node: submit a job for each task still to do in
      DIR/tasks.json, waiting (sbatch --dependency=afterok) for the jobs of the
      tasks it needs, and record it in DIR/jobs.tsv. A task that a job of an
      earlier submission (another DIR in slurm.log_dir) is still queued or
      running for isn't submitted again: its job is waited for instead. Every
      job runs DIR/job.sh, which slurm_pipeline.sh writes.

  pipeline_jobs.py generate --type TYPE --output FILE [--force] -- COMMAND ...
      In a job, in the environment from setup.sh: make one input file with
      COMMAND, one of the commands from scripts/scenario_inputs.py, writing to
      a hidden folder next to FILE, then move what it made into place, FILE
      last, so that an input that exists is complete.

The tasks, by stage, each of them one job:

  generate     an input file of a scenario (ILD_Particle_Gun.py)
  geant4       the Geant4 simulation of an input (simulate_geant4.sh --file)
  models       a model's simulation of an input (simulate_models.sh --file)
  reconstruct  the reconstruction of a simulation (reconstruct_simulation.sh FILE)
  observables  the measurements of a simulator (observables_from_reconstruction.sh)
  plot         the figures and metrics.csv (plot_observables.sh)

A task is done when its output exists, unless it is forced. For the
observables, that is when the .npz file is up to date, as
observables_from_reconstruction.sh decides; for the plots, when metrics.csv is
newer than every .npz file. A task that isn't done waits for the tasks it needs
that aren't done either. As the stage scripts skip what exists, a task whose
output exists stays done even if something it needs is still to be made,
unless that is being made again (--force): then it is made again too.

With --events N, the simulations are of the first N events of each input, as
<input>_n<N>, and each is reconstructed whole, to <input>_n<N>_REC..., which is
what observables_from_reconstruction.sh -n N reads.
"""

import argparse
import getpass
import glob
import json
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile

import geant4_jobs
import observables
import reconstruction_jobs
import simulation_jobs
from fasinate_config import REPO_DIR, ConfigError, load_config
from scenario_inputs import INPUT_FILES, scenario_commands

STAGES = ("generate", "geant4", "models", "reconstruct", "observables", "plot")
GEANT4 = geant4_jobs.NAME
# For each input type, the option that makes its commands write into a given
# folder, so that a job can make one input in a hidden folder and then move it
# into place. A new input type in scripts/scenario_inputs.py needs one here
# before slurm_pipeline.sh can make its inputs.
SAVE_DIR_OPTIONS = {"particle_gun": "--save_dir"}
# The states squeue shows for jobs that are still to run or to finish
ACTIVE_STATES = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED",
                 "REQUEUED", "REQUEUE_HOLD", "REQUEUE_FED", "RESIZING", "SIGNALING",
                 "STAGE_OUT"}
# Statuses of a task
DONE, TODO, BLOCKED, UNSELECTED = "done", "todo", "blocked", "unselected"
# Scenario names become folder names.
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+-]*$")
# A record of a submitted job in jobs.tsv: job ID, stage, task, slurm log
RECORDS = "jobs.tsv"
RECORDS_HEADER = "# job\tstage\ttask\tlog\n"


class PipelineError(Exception):
    pass


# Errors that mean the configuration, the options or the environment aren't right
ERRORS = (PipelineError, ConfigError, geant4_jobs.PlanError, simulation_jobs.PlanError,
          reconstruction_jobs.PlanError, observables.ObservableError)


def say(message):
    print("slurm_pipeline: " + message, file=sys.stderr)


def _write_json(path, data):
    """Write data to path as JSON, whole or not at all."""
    partial = "%s.%d.partial" % (path, os.getpid())
    with open(partial, "w") as stream:
        json.dump(data, stream, indent=1)
        stream.write("\n")
    os.replace(partial, path)


def _read_json(path):
    try:
        with open(path) as stream:
            return json.load(stream)
    except (OSError, ValueError) as error:
        raise PipelineError("can't read %s: %s" % (path, error))


# ---------------------------------------------------------------------------
# The plan, made where the configuration files are


def stage_list(values, option):
    """The stages named in values (each a stage, a list of them separated by
    commas, or all), in the order of the pipeline."""
    named = set()
    for value in values:
        for name in value.split(","):
            name = name.strip()
            if name == "all":
                named.update(STAGES)
            elif name in STAGES:
                named.add(name)
            elif name:
                raise PipelineError("%s: there's no stage '%s'; the stages are %s, or all"
                                    % (option, name, ", ".join(STAGES)))
    return [stage for stage in STAGES if stage in named]


def make_plan(simulators=(), scenarios=(), stages=(), events=None, force=()):
    """What the tasks are made from, as a JSON-able dict."""
    config = load_config()
    data_dir = config["paths"]["data_dir"]
    if events is not None and events < 1:
        raise PipelineError("the number of events must be at least 1, not %d" % events)
    selected = stage_list(stages, "--stages") if stages else list(STAGES)
    if not selected:
        raise PipelineError("--stages names no stage")
    forced = stage_list(force, "--force")
    for stage in forced:
        if stage not in selected:
            raise PipelineError("--force %s: that stage isn't one of the --stages to run"
                                % stage)

    # Every scenario, even when only some are run, as the observables of a
    # simulator are measured on all of theirs
    groups = []
    for name, kind, commands in scenario_commands(config):
        if not _NAME.match(name):
            raise PipelineError("scenario names become folder names, so use only letters, "
                                "digits and _.+- in '%s'" % name)
        if kind not in INPUT_FILES:
            raise PipelineError("scenario %s: there's no way to tell which files a %s input "
                                "makes; add one to INPUT_FILES in scripts/scenario_inputs.py"
                                % (name, kind))
        if kind not in SAVE_DIR_OPTIONS and "generate" in selected:
            raise PipelineError("scenario %s: slurm_pipeline.sh can't make %s inputs one at a "
                                "time; add the option that says where they go to "
                                "SAVE_DIR_OPTIONS in scripts/pipeline_jobs.py, or leave out "
                                "the generate stage (--stages)" % (name, kind))
        groups.append({"scenario": name, "type": kind, "commands": commands})
    known = [group["scenario"] for group in groups]
    for name in scenarios:
        if name not in known:
            raise PipelineError("there is no scenario '%s'; the configuration has: %s"
                                % (name, ", ".join(known) or "none"))

    models = [simulation_jobs.load_model(name) for name in simulation_jobs.model_names()]
    names = [model["name"] for model in models]
    if GEANT4 in names:
        raise PipelineError("models/%s.yaml: '%s' names the Geant4 reference, so it can't be a "
                            "model name" % (GEANT4, GEANT4))
    for name in simulators:
        if name not in [GEANT4] + names:
            raise PipelineError("there's no simulator '%s'; the simulators are: %s"
                                % (name, ", ".join([GEANT4] + names)))
    compact = {GEANT4: config["geant4"]["compact"]}
    compact.update((model["name"], model["compact_file"]) for model in models)

    extra_args = reconstruction_jobs._extra_args(config)
    slurm = config.get("slurm") or {}
    for stage in STAGES:
        if not isinstance((slurm.get("stages") or {}).get(stage), dict):
            raise PipelineError("the configuration has no slurm.stages.%s, the slurm settings "
                                "of the %s jobs" % (stage, stage))

    return {
        "repo_dir": REPO_DIR,
        "config_files": [f for f in os.environ.get("FASINATE_CONFIG", "").split(":") if f],
        "data_dir": data_dir,
        "sim_dir": os.path.join(data_dir, "sim_output"),
        "rec_dir": os.path.join(data_dir, "rec_output"),
        "events": events,
        "stages": selected,
        "force": forced,
        "only_simulators": list(simulators),
        "only_scenarios": list(scenarios),
        "scenarios": groups,
        "models": models,
        "compact": compact,
        "ildconfig_dir": os.path.join(config["detector_configs"]["ildconfig"]["source_dir"],
                                      reconstruction_jobs.ILDCONFIG_PRODUCTION),
        "rec_done_suffix": reconstruction_jobs._done_suffix(extra_args),
        "observables": observables.make_plan(events=events),
        "slurm": slurm,
    }


# ---------------------------------------------------------------------------
# The tasks, worked out in the environment from setup.sh


def describe(task):
    """What a task makes, in words."""
    stage, name, simulator = task["stage"], task["name"], task["simulator"]
    if stage == "generate":
        return "the input %s" % name
    if stage in ("geant4", "models"):
        return "the %s simulation of %s" % ("Geant4" if simulator == GEANT4 else simulator, name)
    if stage == "reconstruct":
        return "the reconstruction of the %s simulation of %s" % (simulator, name)
    if stage == "observables":
        return "the measurements of %s" % simulator
    return "the plots"


def label(task):
    """A task, briefly, for the lists of jobs."""
    if task["stage"] == "generate":
        return "%s/%s" % (task["scenario"], task["name"])
    if task["stage"] in ("geant4", "models", "reconstruct"):
        return "%s %s/%s" % (task["simulator"], task["scenario"], task["name"])
    if task["stage"] == "observables":
        return task["simulator"]
    return "every .npz file"


def measurements(folder, suffix):
    """The .npz files plot_observables.py plots (with -n N for suffix _n<N>)."""
    found = []
    for path in sorted(glob.glob(os.path.join(folder, "*%s.npz" % suffix))):
        name = os.path.basename(path)[:-len(suffix + ".npz")]
        if name.startswith(".") or (suffix == "" and re.search(r"_n\d+$", name)):
            continue
        found.append(path)
    return found


class Tasks:
    """Every task of the pipeline, in an order where each comes after the
    tasks it needs, with what is done."""

    def __init__(self, plan):
        self.plan = plan
        self.events = plan["events"]
        self.suffix = "" if self.events is None else "_n%d" % self.events
        self.list = []
        self.by_key = {}
        self.writers = {}  # output: key
        self.extra = {}  # key: what only the status needs
        self._make()

    def _add(self, stage, output, needs, args, scenario=None, simulator=None, name=None):
        key = "%s:%s" % (stage, output)
        if output in self.writers:
            raise PipelineError("%s and %s would both write %s"
                                % (self.writers[output], key, output))
        self.writers[output] = key
        task = {"key": key, "stage": stage, "scenario": scenario, "simulator": simulator,
                "name": name, "output": output, "needs": needs, "args": args}
        self.list.append(task)
        self.by_key[key] = task
        return task

    def _make(self):
        plan = self.plan
        events = [] if self.events is None else ["--events", str(self.events)]

        # Stage 1: the inputs of each scenario, one task per file
        inputs = []  # (scenario, input file, its task)
        for group in plan["scenarios"]:
            commands = group["commands"]
            try:
                files = INPUT_FILES[group["type"]](commands)
            except ImportError as error:
                raise PipelineError("naming the inputs needs the environment from setup.sh "
                                    "(%s); slurm_pipeline.sh runs this step there" % error)
            if len(files) != len(commands):
                raise PipelineError("scenario %s: %d commands make %d files"
                                    % (group["scenario"], len(commands), len(files)))
            for path, command in zip(files, commands):
                path = os.path.abspath(path)
                key = "generate:" + path
                task = self.by_key.get(key)
                if task is None:
                    stem = os.path.splitext(os.path.basename(path))[0]
                    task = self._add("generate", path, [],
                                     ["--type", group["type"], "--output", path, "--"] + command,
                                     group["scenario"], None, stem)
                    task["scenarios"] = []
                task["scenarios"].append(group["scenario"])
                inputs.append((group["scenario"], path, task))

        # Stages 2 and 3: every input simulated by Geant4 and by each model
        simulations = []
        self.simulators = [GEANT4] + [model["name"] for model in plan["models"]]
        for simulator in self.simulators:
            for scenario, path, source in inputs:
                stem = os.path.splitext(os.path.basename(path))[0] + self.suffix
                folder = os.path.join(plan["sim_dir"], scenario, simulator)
                args = ["--scenario", scenario, "--file", path] + events
                if simulator == GEANT4:
                    stage = "geant4"
                    output = os.path.join(folder, stem + geant4_jobs.OUTPUT_EXTENSION)
                else:
                    stage = "models"
                    output = os.path.join(folder, stem + ".root")
                    args.append(simulator)
                simulations.append(self._add(stage, output, [source["key"]], args, scenario,
                                             simulator, stem))

        # Stage 4: each simulation reconstructed, as reconstruct_simulation.sh
        # names the reconstruction of a file in sim_output. Without --events,
        # so that a simulation of the first N events, <input>_n<N>, becomes
        # <input>_n<N>_REC...
        reconstructions = {}  # (simulator, scenario): [task]
        for simulation in simulations:
            base = os.path.join(plan["rec_dir"], simulation["scenario"],
                                simulation["simulator"], simulation["name"])
            task = self._add("reconstruct", base + plan["rec_done_suffix"], [simulation["key"]],
                             [simulation["output"]], simulation["scenario"],
                             simulation["simulator"], simulation["name"])
            reconstructions.setdefault((task["simulator"], task["scenario"]), []).append(task)

        # Stage 5: each simulator measured on the scenarios of the observables
        measured = plan["observables"]
        self.profiles = observables.wanted(measured)
        used = sorted({profile[1] for profile in self.profiles})
        expected = {scenario: observables.expected_files(measured, scenario) for scenario in used}
        measuring = []
        for simulator in self.simulators:
            needs = [task["key"] for scenario in used
                     for task in reconstructions.get((simulator, scenario), [])]
            output = os.path.join(measured["out_dir"], simulator + self.suffix + ".npz")
            task = self._add("observables", output, needs, events + [simulator],
                             None, simulator, simulator)
            self.extra[task["key"]] = {
                scenario: [os.path.join(measured["rec_dir"], scenario, simulator, name)
                           for name in expected[scenario]] for scenario in used}
            measuring.append(task)

        # Stage 6: the plots of everything measured
        output = os.path.join(measured["out_dir"], "plots", "metrics%s.csv" % self.suffix)
        self._add("plot", output, [task["key"] for task in measuring], events, name="plots")

    def _current(self, task):
        """(done, why not) from what is on disk, whatever the tasks it needs."""
        stage, output = task["stage"], task["output"]
        if stage == "observables":
            files = self.extra[task["key"]]
            paths = [path for scenario in sorted(files) for path in files[scenario]]
            absent = [path for path in paths if not os.path.isfile(path)]
            if absent:
                return False, "%d of its %d reconstructions aren't there" % (len(absent),
                                                                          len(paths))
            return observables.is_up_to_date(output, self.profiles, files)
        if stage == "plot":
            found = measurements(os.path.dirname(os.path.dirname(output)), self.suffix)
            if not found:
                return False, "there are no measurements to plot"
            if not os.path.isfile(output):
                return False, "not plotted yet"
            if max(os.path.getmtime(path) for path in found) > os.path.getmtime(output):
                return False, "measured again since the last plots"
            return True, ""
        if os.path.exists(output):
            return True, ""
        return False, "not made yet"

    def _selected(self, task):
        plan = self.plan
        if task["stage"] not in plan["stages"]:
            return False
        if plan["only_simulators"] and task["simulator"] is not None \
                and task["simulator"] not in plan["only_simulators"]:
            return False
        if plan["only_scenarios"] and task["stage"] not in ("observables", "plot"):
            scenarios = task.get("scenarios") or [task["scenario"]]
            if not set(scenarios) & set(plan["only_scenarios"]):
                return False
        return True

    def find_status(self):
        """Set the status of every task: done, todo (to be run), blocked (it
        needs what isn't done and won't be), or unselected (not chosen to run
        with --stages, --scenario or the simulators); for those to be run, the
        tasks they wait for, and whether they make their output again."""
        for task in self.list:
            needs = [self.by_key[key] for key in task["needs"]]
            waiting = [other for other in needs if other["status"] == TODO]
            remade = [other for other in waiting if other["redo"]]
            missing = [other for other in needs if other["status"] in (BLOCKED, UNSELECTED)]
            done, why = self._current(task)
            if task["stage"] in self.plan["force"]:
                done, why = False, "--force %s" % task["stage"]
            elif remade:
                done, why = False, "%s is made again" % describe(remade[0])
            elif waiting and task["stage"] in ("observables", "plot"):
                done, why = False, "%s is still to be made" % describe(waiting[0])

            task.update(status=DONE, why="", redo=False, after=[])
            if done:
                continue
            if not self._selected(task):
                task.update(status=UNSELECTED, why=why)
            elif missing and task["stage"] != "plot":
                # The plots are of whatever is measured, so only wait for what will be
                other = missing[0]
                task.update(status=BLOCKED, why="it needs %s, which %s" % (
                    describe(other), "isn't selected to run" if other["status"] == UNSELECTED
                    else "is blocked"))
            elif task["stage"] == "plot" and not waiting and not measurements(
                    os.path.dirname(os.path.dirname(task["output"])), self.suffix):
                task.update(status=BLOCKED, why="there is nothing measured to plot yet")
            else:
                redo = os.path.exists(task["output"]) and task["stage"] != "plot"
                if redo:
                    # The stage scripts skip outputs that exist, unless forced
                    task["args"] = ["--force"] + task["args"]
                task.update(status=TODO, why=why, redo=redo,
                            after=[other["key"] for other in waiting])

    def environment_problems(self):
        """What would make the tasks to run fail in this environment, by the
        checks the stage scripts make before they start."""
        todo = [task for task in self.list if task["status"] == TODO]
        stages = {task["stage"] for task in todo}
        problems = []
        if "geant4" in stages:
            try:
                geant4_jobs.check_environment({"compact": self.plan["compact"][GEANT4]})
            except geant4_jobs.PlanError as error:
                problems.append(str(error))
        models = sorted({task["simulator"] for task in todo if task["stage"] == "models"})
        if models:
            try:
                simulation_jobs.check_environment(
                    [model for model in self.plan["models"] if model["name"] in models])
            except simulation_jobs.PlanError as error:
                problems.append(str(error))
        simulators = sorted({task["simulator"] for task in todo
                             if task["stage"] == "reconstruct"})
        if simulators:
            check = {"ildconfig_dir": self.plan["ildconfig_dir"],
                     "groups": [{"compact": self.plan["compact"][name]} for name in simulators]}
            try:
                reconstruction_jobs.check_environment(check)
            except reconstruction_jobs.PlanError as error:
                problems.append(str(error))
        if "observables" in stages:
            for module in ("numpy", "uproot", "awkward"):
                try:
                    __import__(module)
                except ImportError as error:
                    problems.append("measuring the observables needs the Python module %s, "
                                    "which this environment doesn't have: %s" % (module, error))
        return problems


def make_tasks(plan):
    tasks = Tasks(plan)
    tasks.find_status()
    return {"events": plan["events"], "tasks": tasks.list,
            "problems": tasks.environment_problems()}


# ---------------------------------------------------------------------------
# Submitting, on a slurm login node


def earlier_jobs(log_dir, directory):
    """{task: job ID} from the earlier submissions in log_dir, the latest
    submission for each task."""
    found = {}
    for path in sorted(glob.glob(os.path.join(log_dir, "pipeline_*", RECORDS))):
        if os.path.realpath(os.path.dirname(path)) == os.path.realpath(directory):
            continue
        try:
            with open(path) as stream:
                lines = stream.read().splitlines()
        except OSError:
            continue
        for line in lines:
            fields = line.split("\t")
            if not line.startswith("#") and len(fields) >= 3:
                found[fields[2]] = fields[0]
    return found


def active_jobs():
    """{job ID: state} of this user's jobs that are still to run or finishing,
    or None if squeue can't tell."""
    user = os.environ.get("USER") or getpass.getuser()
    try:
        result = subprocess.run(["squeue", "-h", "-u", user, "-o", "%i|%T|%r"],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                universal_newlines=True)
    except OSError:
        return None
    if result.returncode != 0:
        return None
    jobs = {}
    for line in result.stdout.splitlines():
        fields = line.strip().split("|")
        if len(fields) < 2:
            continue
        reason = fields[2] if len(fields) > 2 else ""
        # A job waiting for a job that failed never runs
        if fields[1] in ACTIVE_STATES and reason != "DependencyNeverSatisfied":
            jobs[fields[0]] = fields[1]
    return jobs


def job_name(slurm, task):
    prefix = slurm.get("job_name_prefix") or "fasinate"
    stage = task["stage"]
    if stage == "models":
        part = task["simulator"]
    elif stage == "reconstruct":
        part = "reco-" + task["simulator"]
    elif stage == "observables":
        part = "obs-" + task["simulator"]
    else:
        part = stage
    return "%s-%s" % (prefix, part)


def log_name(number, task):
    parts = [task["stage"]]
    if task["stage"] in ("models", "reconstruct", "observables"):
        parts.append(task["simulator"])
    if task["stage"] in ("generate", "geant4", "models", "reconstruct"):
        parts += [task["scenario"], task["name"]]
    return "%03d-%s.out" % (number, re.sub(r"[^A-Za-z0-9_.+-]", "_", "-".join(parts)))


def sbatch_command(plan, task, after, log, job_script, extra):
    """The sbatch command of a task, waiting for the jobs in after."""
    slurm = plan["slurm"]
    settings = slurm["stages"][task["stage"]]
    command = ["sbatch", "--parsable", "--job-name=" + job_name(slurm, task),
               # slurm reads % in file names as a pattern
               "--output=" + log.replace("%", "%%"), "--chdir=" + plan["repo_dir"]]
    for option in ("time", "partition", "constraint"):
        if settings.get(option) not in (None, ""):
            command.append("--%s=%s" % (option, settings[option]))
    for args in (slurm.get("sbatch_args"), settings.get("sbatch_args"), extra):
        command += [str(arg) for arg in args or []]
    if after:
        command += ["--dependency=afterok:" + ":".join(after), "--kill-on-invalid-dep=yes"]
    return command + [job_script, task["stage"], task["output"]] + task["args"]


class Report:
    """Lines printed, and kept for report.txt."""

    def __init__(self):
        self.lines = []

    def __call__(self, line=""):
        print(line)
        sys.stdout.flush()
        self.lines.append(line)

    def save(self, path):
        with open(path, "w") as stream:
            stream.write("\n".join(self.lines) + "\n")


def report_stages(out, tasks, queued):
    out("slurm_pipeline: what each stage has done, and has to do:")
    out("  %-12s %6s %10s %8s %8s %13s" % ("stage", "done", "to submit", "queued", "blocked",
                                          "not selected"))
    for stage in STAGES:
        counts = {DONE: 0, TODO: 0, BLOCKED: 0, UNSELECTED: 0}
        waiting = 0
        for task in tasks:
            if task["stage"] == stage:
                counts[task["status"]] += 1
                waiting += task["key"] in queued
        out("  %-12s %6d %10d %8d %8d %13d" % (stage, counts[DONE], counts[TODO] - waiting,
                                              waiting, counts[BLOCKED], counts[UNSELECTED]))
    blocked = [task for task in tasks if task["status"] == BLOCKED]
    if blocked:
        out("slurm_pipeline: blocked, so not submitted:")
        for task in blocked:
            out("  %-12s %s: %s" % (task["stage"], label(task), task["why"]))


def submit(directory, dry_run=False, extra=()):
    plan = _read_json(os.path.join(directory, "plan.json"))
    found = _read_json(os.path.join(directory, "tasks.json"))
    tasks = found["tasks"]
    job_script = os.path.join(directory, "job.sh")
    records_path = os.path.join(directory, RECORDS)
    out = Report()

    # Jobs of earlier submissions that are still to run
    earlier = earlier_jobs(plan["slurm"]["log_dir"], directory)
    active = active_jobs()
    if active is None:
        if earlier:
            say("warning: squeue can't tell which jobs of earlier submissions are still queued, "
                "so this may submit jobs again that are queued already")
        active = {}
    queued = {}  # task: job ID
    for task in tasks:
        job = earlier.get(task["key"])
        if task["status"] == TODO and job in active:
            queued[task["key"]] = job

    report_stages(out, tasks, queued)
    to_submit = [task for task in tasks if task["status"] == TODO and task["key"] not in queued]
    if found["problems"]:
        for problem in found["problems"]:
            say(problem)
        if to_submit and not dry_run:
            raise PipelineError("the environment can't run the jobs (see above), so nothing "
                                "was submitted")
    if not to_submit:
        if queued:
            out("slurm_pipeline: nothing new to submit; %d job(s) of earlier submissions are "
                "still queued or running" % len(queued))
        elif any(task["status"] == BLOCKED for task in tasks):
            out("slurm_pipeline: nothing to submit, but some tasks are blocked (see above)")
        else:
            out("slurm_pipeline: everything selected is done, so there is nothing to submit")
        return 0

    ids = dict(queued)
    submitted = []
    example = None
    out("slurm_pipeline: %s %d job(s) (job, stage, what, after which jobs):"
        % ("would submit" if dry_run else "submitting", len(to_submit)))
    try:
        for number, task in enumerate(to_submit, 1):
            after = [ids[key] for key in task["after"]]
            log = os.path.join(directory, "logs", log_name(number, task))
            command = sbatch_command(plan, task, after, log, job_script, extra)
            if example is None or (after and "--kill-on-invalid-dep=yes" not in example):
                example = command
            if dry_run:
                job = "#%d" % number
            else:
                result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        universal_newlines=True)
                job = result.stdout.strip().split(";")[0]
                if result.returncode != 0 or not job.isdigit():
                    raise PipelineError("sbatch failed for %s: %s" % (
                        describe(task), (result.stderr or result.stdout).strip()))
                new = not os.path.exists(records_path)
                with open(records_path, "a") as stream:
                    if new:
                        stream.write(RECORDS_HEADER)
                    stream.write("%s\t%s\t%s\t%s\n" % (job, task["stage"], task["key"], log))
            ids[task["key"]] = job
            submitted.append(task)
            out("  %-10s %-12s %s%s" % (job, task["stage"], label(task),
                                        "  after " + ", ".join(after) if after else ""))
    except (PipelineError, OSError) as error:
        say(str(error))
        if submitted:
            say("the %d job(s) submitted before that are in %s, so running slurm_pipeline.sh "
                "again waits for them instead of submitting them again"
                % (len(submitted), records_path))
        out.save(os.path.join(directory, "report.txt"))
        return 1

    if queued:
        out("slurm_pipeline: and waiting for %d job(s) of earlier submissions: %s" % (
            len(queued), ", ".join(sorted(set(queued.values()), key=lambda j: (len(j), j)))))
    if dry_run:
        out("slurm_pipeline: dry run, so nothing was submitted; a job would be submitted as")
        out("  " + " ".join(shlex.quote(arg) for arg in example))
        if found["problems"]:
            say("but the environment can't run the jobs (see above), so slurm_pipeline.sh "
                "wouldn't submit them")
            return 1
        return 0
    out("slurm_pipeline: job script, slurm logs (logs/) and the list of jobs (%s) in %s"
        % (RECORDS, directory))
    for task in submitted:
        if task["stage"] == "plot":
            out("slurm_pipeline: the figures and metrics go to %s"
                % os.path.dirname(task["output"]))
    out("slurm_pipeline: follow them with: squeue -u %s -o '%%.12i %%.32j %%.9T %%.11M %%R'"
        % (os.environ.get("USER") or getpass.getuser()))
    out("slurm_pipeline: run slurm_pipeline.sh again, with the same options, to resubmit "
        "whatever fails or times out")
    out.save(os.path.join(directory, "report.txt"))
    return 0


# ---------------------------------------------------------------------------
# Making one input, in a job


def _exit_on_signal(signum, frame):
    raise SystemExit(128 + signum)


def generate(kind, output, command, force=False):
    option = SAVE_DIR_OPTIONS.get(kind)
    if option is None:
        raise PipelineError("can't make %s inputs one at a time (see SAVE_DIR_OPTIONS)" % kind)
    if not command:
        raise PipelineError("give the command that makes the input, after --")
    if os.path.exists(output) and not force:
        say("%s is there already, so it isn't made again" % output)
        return 0
    folder = os.path.dirname(output)
    os.makedirs(folder, exist_ok=True)
    # Removed again when the job is cancelled or times out (SIGTERM). Named
    # without the input's extension, as ILD_Particle_Gun.py names its .h5 file
    # by replacing ".slcio" anywhere in the path.
    signal.signal(signal.SIGTERM, _exit_on_signal)
    stem = os.path.splitext(os.path.basename(output))[0]
    work = tempfile.mkdtemp(dir=folder, prefix=".%s.%s.making." % (
        stem, socket.gethostname().split(".")[0]))
    try:
        command = list(command) + [option, work]
        say("making %s with" % output)
        say("  " + " ".join(shlex.quote(arg) for arg in command))
        status = subprocess.call(command, stdin=subprocess.DEVNULL)
        if status != 0:
            raise PipelineError("making %s failed: the command exited with status %d"
                                % (output, status))
        made = os.path.join(work, os.path.basename(output))
        if not os.path.isfile(made) or os.path.getsize(made) == 0:
            raise PipelineError("the command made no %s (it made: %s)" % (
                os.path.basename(output), ", ".join(sorted(os.listdir(work))) or "nothing"))
        # What else it made (the gun's .h5 file of the true momenta) first, and
        # the input itself last, as that marks it done
        for name in sorted(os.listdir(work)):
            if name != os.path.basename(output):
                os.replace(os.path.join(work, name), os.path.join(folder, name))
        os.replace(made, output)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    say("made %s" % output)
    return 0


# ---------------------------------------------------------------------------


def _after_dashes(args):
    return args[1:] if args[:1] == ["--"] else args


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Work out, submit and run the slurm jobs of slurm_pipeline.sh.",
        epilog="See the module docstring for the steps.")
    steps = parser.add_subparsers(dest="step")
    plan_parser = steps.add_parser("plan", help="print the plan as JSON")
    plan_parser.add_argument("simulators", nargs="*", metavar="SIMULATOR",
                             help="only these simulators: models in models/, or geant4")
    plan_parser.add_argument("--scenario", action="append", default=[], metavar="NAME",
                             help="only this scenario (repeatable)")
    plan_parser.add_argument("--stages", action="append", default=[], metavar="LIST",
                             help="only these stages, separated by commas (repeatable)")
    plan_parser.add_argument("--events", type=int, metavar="N",
                             help="simulate only the first N events of each input")
    plan_parser.add_argument("--force", action="append", default=[], metavar="STAGE",
                             help="make the outputs of STAGE again where they exist, and what "
                                  "uses them (repeatable)")
    tasks_parser = steps.add_parser("tasks", help="write the tasks and their status")
    tasks_parser.add_argument("plan", help="the plan, from the plan step")
    tasks_parser.add_argument("tasks", help="where the tasks go")
    submit_parser = steps.add_parser("submit", help="submit the tasks still to do")
    submit_parser.add_argument("--dry-run", action="store_true",
                               help="show what would be submitted, without submitting it")
    submit_parser.add_argument("directory", metavar="DIR",
                               help="the submission's folder, with plan.json and tasks.json")
    submit_parser.add_argument("sbatch_args", nargs=argparse.REMAINDER, metavar="SBATCH_OPTION",
                               help="after --: more options for every sbatch call")
    generate_parser = steps.add_parser("generate", help="make one input file")
    generate_parser.add_argument("--type", required=True, help="the scenario's input type")
    generate_parser.add_argument("--output", required=True, metavar="FILE",
                                 help="the input file that the command makes")
    generate_parser.add_argument("--force", action="store_true",
                                 help="make it again if it exists")
    generate_parser.add_argument("command", nargs=argparse.REMAINDER, metavar="COMMAND",
                                 help="after --: the command that makes it")
    args = parser.parse_args(argv)

    try:
        if args.step == "plan":
            plan = make_plan(args.simulators, args.scenario, args.stages, args.events, args.force)
            print(json.dumps(plan))
            return 0
        if args.step == "tasks":
            _write_json(args.tasks, make_tasks(_read_json(args.plan)))
            return 0
        if args.step == "submit":
            return submit(args.directory, args.dry_run, _after_dashes(args.sbatch_args))
        if args.step == "generate":
            return generate(args.type, os.path.abspath(args.output),
                            _after_dashes(args.command), args.force)
        parser.error("choose a step: plan, tasks, submit or generate")
    except ERRORS as error:
        say(str(error))
        return 1


if __name__ == "__main__":
    sys.exit(main())
