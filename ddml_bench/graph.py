"""The task graph: every task of the pipeline, what each needs, and what is done.

Every way of running the pipeline goes through it: the stage scripts, which
run one stage, and pipeline.sh and slurm_pipeline.sh, which run them all.

  make_plan(...)      Where the configuration files are, before the environment
                      is set up: read the configuration and the models, check
                      them and the options, and name every input. The plan is
                      a JSON-able dict, which the rest is made from.
  Graph(plan)         The tasks: each stage of the registry (ddml_bench/stages)
                      adds its own, in an order where each task comes after the
                      tasks it needs. The layout (ddml_bench/layout.py) names
                      every file.
  graph.find_status() In the environment from setup.sh (the observables' check
                      reads .npz files with numpy): what is done, what is to
                      run, and what each task to run waits for.

A task is done when its output exists, unless its stage is forced; the
measurements when they are up to date with the reconstructions, and the plots
with the measurements. A task that isn't done waits for the tasks it needs that
aren't done either. A task whose output exists stays done even if something it
needs is still to be made, unless that is being made again (--force): then it
is made again too.

Each task gets a status: done; todo (to run); blocked (it needs what isn't done
and won't be made this time); or unselected (not chosen by --stages, the
simulators, --scenario or the files given).
"""

import os
import sys

from . import measure
from .config import REPO_DIR, ConfigError, load_config
from .layout import (CUSTOM_SCENARIO, GEANT4, REC_SUFFIX, Layout, LayoutError, simulation_stem,
                     simulations_in)
from .models import ModelError, load_models, reference
from .scenarios import INPUT_TYPES, scenario_inputs
from .stages import BY_NAME, STAGE_NAMES, STAGES, StageError
from .stages import reconstruct

DONE, TODO, BLOCKED, UNSELECTED = "done", "todo", "blocked", "unselected"
STATUSES = (DONE, TODO, BLOCKED, UNSELECTED)


class PlanError(Exception):
    pass


def say(message):
    print("ddml-bench: " + message, file=sys.stderr)


def stage_list(values, option):
    """The stages named in values (each a stage, a list of them separated by
    commas, or all), in the order of the pipeline."""
    named = set()
    for value in values:
        for name in value.split(","):
            name = name.strip()
            if name == "all":
                named.update(STAGE_NAMES)
            elif name in STAGE_NAMES:
                named.add(name)
            elif name:
                raise PlanError("%s: there's no stage '%s'; the stages are %s, or all"
                                % (option, name, ", ".join(STAGE_NAMES)))
    return [stage for stage in STAGE_NAMES if stage in named]


def _real_files(paths, what):
    """The real paths of files given on the command line, each once."""
    found = []
    for path in paths:
        if not os.path.isfile(path):
            raise PlanError("%s not found: %s" % (what, path))
        real = os.path.realpath(path)
        if real not in found:
            found.append(real)
    return found


def make_plan(simulators=(), scenarios=(), stages=(), events=None, force=(), inputs=(),
              files=(), simulations=(), compact=None, slurm=False):
    """What the tasks are made from, as a JSON-able dict.

    simulators, scenarios: run only these (default: all of them)
    stages: run only these (lists separated by commas, or all; default: all)
    events: a test on the first N events of each input (default: all)
    force: stages whose outputs are made again where they exist
    inputs: input files simulated as the scenario custom
    files: run only on these inputs of the scenarios
    simulations: simulation files to reconstruct
    compact: reconstruct in this detector instead
    slurm: check the slurm settings too
    """
    try:
        config = load_config()
        layout = Layout(config["paths"]["data_dir"], events)
    except LayoutError as error:
        raise PlanError(str(error))
    selected = stage_list(stages, "--stages") if stages else list(STAGE_NAMES)
    if not selected:
        raise PlanError("--stages names no stage")
    forced = stage_list(force, "--force")
    for stage in forced:
        if stage not in selected:
            raise PlanError("--force %s: that stage isn't one of the --stages to run" % stage)

    # Every scenario, even when only some are run, as the observables of a
    # simulator are measured on all of theirs
    groups = scenario_inputs(config, layout)
    if "generate" in selected:
        for group in groups:
            if not INPUT_TYPES[group["type"]].save_dir_option:
                raise PlanError("scenario %s: %s inputs can't be made one at a time; add the "
                                "option that says where they go to INPUT_TYPES in "
                                "ddml_bench/scenarios.py, or leave out the generate stage "
                                "(--stages)" % (group["scenario"], group["type"]))
    configured = [group["scenario"] for group in groups]

    # Inputs given directly, simulated as the scenario custom
    custom_inputs = _real_files(inputs, "input file")
    if custom_inputs and CUSTOM_SCENARIO in configured:
        raise PlanError("files given with --input are simulated as the scenario '%s', but the "
                        "configuration has a scenario of that name" % CUSTOM_SCENARIO)
    # Simulations given directly, to reconstruct; with --scenario custom, those
    # that were simulated with --input
    given = _real_files(simulations, "simulation file")
    for path in given:
        if not simulation_stem(os.path.basename(path)):
            raise PlanError("%s isn't a finished simulation: an EDM4hep file named *.root or "
                            "*.edm4hep.root (not hidden)" % path)
    known = list(configured)
    if custom_inputs:
        known.append(CUSTOM_SCENARIO)
    elif CUSTOM_SCENARIO in scenarios and "reconstruct" in selected:
        custom_dir = os.path.join(layout.sim_dir, CUSTOM_SCENARIO)
        for folder in sorted(os.listdir(custom_dir)) if os.path.isdir(custom_dir) else []:
            for path in simulations_in(os.path.join(custom_dir, folder)):
                if os.path.realpath(path) not in given:
                    given.append(os.path.realpath(path))
        known.append(CUSTOM_SCENARIO)
    for name in scenarios:
        if name not in known:
            raise PlanError("there is no scenario '%s'; the configuration has: %s"
                            % (name, ", ".join(configured) or "none"))
    only_scenarios = list(dict.fromkeys(scenarios))
    if custom_inputs and only_scenarios and CUSTOM_SCENARIO not in only_scenarios:
        only_scenarios.append(CUSTOM_SCENARIO)
    elif custom_inputs and not only_scenarios:
        only_scenarios = [CUSTOM_SCENARIO]

    # The simulators
    try:
        models = load_models()
    except ModelError as error:
        raise PlanError(str(error))
    geant4 = reference(config)
    names = [GEANT4] + [model["name"] for model in models]
    for name in simulators:
        if name not in names:
            raise PlanError("there's no simulator '%s'; the simulators are: %s"
                            % (name, ", ".join(names)))
    if {"geant4", "models"} & set(selected):
        for model in models:
            if model["compact_file"] != geant4["compact_file"]:
                say("warning: models/%s.yaml simulates in %s, but geant4.compact is %s; the "
                    "comparison needs the same detector" % (model["name"],
                                                           model["compact_file"],
                                                           geant4["compact_file"]))

    # Only these inputs of the scenarios
    only_inputs = []
    for path in _real_files(files, "input file"):
        matches = [entry["file"] for group in groups for entry in group["inputs"]
                   if os.path.realpath(entry["file"]) == path]
        matches += [p for p in custom_inputs if p == path]
        if not matches:
            raise PlanError("%s isn't an input of any scenario; to simulate it anyway, give it "
                            "with --input, as the scenario %s" % (path, CUSTOM_SCENARIO))
        only_inputs += [p for p in matches if p not in only_inputs]

    # The reconstruction
    if compact and os.path.isabs(compact) and not os.path.isfile(compact):
        raise PlanError("compact file not found: %s" % compact)
    try:
        extra_args = reconstruct.extra_args(config)
    except StageError as error:
        raise PlanError(str(error))
    done_suffix = reconstruct.done_suffix(extra_args)
    reconstruction = {
        "extra_args": extra_args,
        "done_suffix": done_suffix,
        "compact": compact,
        "ildconfig_dir": os.path.join(config["detector_configs"]["ildconfig"]["source_dir"],
                                      reconstruct.ILDCONFIG_PRODUCTION),
        "particle_gun": [g["scenario"] for g in groups if INPUT_TYPES[g["type"]].particle_gun],
    }

    # The observables
    try:
        observables, measured = measure.check_observables(config)
    except measure.ObservableError as error:
        raise PlanError(str(error))
    for scenario in measured:
        if scenario not in configured:
            raise PlanError("the observables are measured on the scenario %s, which the "
                            "configuration doesn't have" % scenario)
    if done_suffix != REC_SUFFIX and "observables" in selected:
        raise PlanError("the observables are measured on the EDM4hep reconstructions, but "
                        "reconstruction.extra_args has --lcioOutput only")

    settings = config.get("slurm") or {}
    if slurm:
        for stage in STAGE_NAMES:
            if not isinstance((settings.get("stages") or {}).get(stage), dict):
                raise PlanError("the configuration has no slurm.stages.%s, the slurm settings of "
                                "the %s jobs" % (stage, stage))

    plan = {
        "repo_dir": REPO_DIR,
        "config_files": [f for f in os.environ.get("FASINATE_CONFIG", "").split(":") if f],
        "data_dir": layout.data_dir,
        "events": events,
        "stages": selected,
        "force": forced,
        "only_simulators": list(dict.fromkeys(simulators)),
        "only_scenarios": only_scenarios,
        "only_inputs": only_inputs,
        "scenarios": groups,
        "custom_inputs": custom_inputs,
        "given_simulations": given,
        "reference": geant4,
        "models": models,
        "compact": dict((sim["name"], sim["compact_file"]) for sim in [geant4] + models),
        "reconstruction": reconstruction,
        "observables": {"observables": observables, "scenarios": measured},
        "slurm": settings,
    }
    Graph(plan)  # names every file, and checks that no two tasks make the same one
    return plan


class Graph(object):
    """Every task of the pipeline, in an order where each comes after the tasks
    it needs."""

    def __init__(self, plan):
        self.plan = plan
        self.layout = Layout(plan["data_dir"], plan["events"])
        self.list = []
        self.by_key = {}
        self.writers = {}  # output: key
        # What the stages make, for the stages after them
        self.inputs = []  # (scenario, input file, its generate task or None)
        # The ddsim seeds each scenario's inputs are simulated with ([None]: once,
        # with ddsim's own)
        self.seeds = dict((group["scenario"], group.get("seeds") or [None])
                          for group in plan["scenarios"])
        self.simulations = []  # simulation tasks
        self.reconstructions = {}  # (simulator, scenario): [reconstruction tasks]
        self.measuring = []  # observables tasks
        self.given = set()  # keys of the reconstructions of simulations given directly
        try:
            for stage in STAGES:
                stage.add_tasks(self)
        except (StageError, LayoutError, ConfigError) as error:
            raise PlanError(str(error))

    def add(self, stage, output, needs, scenario=None, simulator=None, name=None, inputs=(),
            input_file=None, params=None, seed=None):
        """Add a task; its key is <stage>:<output>, as only one task makes a file.
        seed is the ddsim seed of the simulation it is or is made from, if it
        has one of its own (the scenario's simulation_seeds)."""
        key = "%s:%s" % (stage, output)
        if output in self.writers:
            raise PlanError("%s and %s would both write %s" % (self.writers[output], key, output))
        self.writers[output] = key
        task = {"key": key, "stage": stage, "scenario": scenario, "simulator": simulator,
                "name": name, "seed": seed, "inputs": list(inputs), "input_file": input_file,
                "output": output, "needs": list(needs), "params": params or {}}
        self.list.append(task)
        self.by_key[key] = task
        return task

    def describe(self, task):
        return BY_NAME[task["stage"]].describe(task)

    def selected(self, task):
        """Whether the options choose the task to run, if it isn't done."""
        plan = self.plan
        stage = BY_NAME[task["stage"]]
        if task["stage"] not in plan["stages"]:
            return False
        if task["key"] in self.given:
            return True
        if task["stage"] == "reconstruct" and plan["given_simulations"] \
                and not (plan["only_simulators"] or plan["only_scenarios"]):
            return False
        if plan["only_simulators"] and task["simulator"] is not None \
                and task["simulator"] not in plan["only_simulators"]:
            return False
        if plan["only_scenarios"] and not stage.gathers:
            scenarios = task.get("scenarios") or [task["scenario"]]
            if not set(scenarios) & set(plan["only_scenarios"]):
                return False
        if plan["only_inputs"] and task["input_file"] is not None \
                and task["input_file"] not in plan["only_inputs"]:
            return False
        return True

    def find_status(self):
        """Set the status of every task (see the module docstring); for those to
        run, the tasks they wait for (after), and whether they make their output
        again (redo)."""
        for task in self.list:
            stage = BY_NAME[task["stage"]]
            needs = [self.by_key[key] for key in task["needs"]]
            waiting = [other for other in needs if other["status"] == TODO]
            remade = [other for other in waiting if other["redo"]]
            missing = [other for other in needs if other["status"] in (BLOCKED, UNSELECTED)]
            done, why = stage.current(task)
            if task["stage"] in self.plan["force"]:
                done, why = False, "--force %s" % task["stage"]
            elif remade:
                done, why = False, "%s is made again" % self.describe(remade[0])
            elif waiting and stage.gathers:
                done, why = False, "%s is still to be made" % self.describe(waiting[0])

            task.update(status=DONE, why="", redo=False, after=[])
            if done:
                continue
            if not self.selected(task):
                task.update(status=UNSELECTED, why=why)
            elif missing and not stage.takes_what_there_is:
                other = missing[0]
                task.update(status=BLOCKED, why="it needs %s, which %s" % (
                    self.describe(other), "isn't selected to run" if other["status"] == UNSELECTED
                    else "is blocked"))
            elif not waiting and stage.nothing_to_use(task):
                task.update(status=BLOCKED, why=stage.nothing_to_use(task))
            else:
                # Made again where it exists, without checking again when it runs
                task.update(status=TODO, why=why, redo=os.path.exists(task["output"]),
                            after=[other["key"] for other in waiting])
        return self

    def tasks(self, status):
        return [task for task in self.list if task.get("status") == status]

    def environment_problems(self):
        """What would make the tasks to run fail in this environment, by each
        stage's checks."""
        problems = []
        for stage in STAGES:
            tasks = [t for t in self.tasks(TODO) if t["stage"] == stage.name]
            if tasks:
                problems += stage.check_environment(tasks)
        return problems
