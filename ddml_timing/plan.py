"""The timing tasks: one for each kind of node (timing.devices), input of an
enabled scenario and simulator timed on that kind of node; and what of each is
done.

A task times one input with one simulator on one kind of node, as the
pipeline simulates it (its compact file, steering file and options, and the
first of its scenario's simulation_seeds, if it has any), on its first
warmup_events + events events. Its results are named after the input's
simulation, as the pipeline names it (ddml_bench/layout.py), with _t<events>:

  <paths.data_dir>/timing/<device>/<scenario>/<simulator>/<name>_t<N>.json
      the task's settings, the node, the software, what the models' inference
      ran on, and the statistics of the times per event
  ....events.csv    each event's times (ddml_timing/events.py)
  ....log           ddsim's output, with what ran at the top
  ....failed.json   why the last attempt failed, until one succeeds

make_plan() needs only PyYAML, so it runs where the configuration is, before
the environment is set up. find_status() runs in the environment (from
setup.sh), as it asks DDML's Python package what each model's inference runs
with: a task is done when its .json records the settings the plan has for it
(identity: the events, the warm-up, where the inference runs, the CPU pinning,
the input and seed, and what the simulator is, as the pipeline's manifests
record it), to do when it isn't, skipped when its model's inference can't run
on its kind of node's device (a model DDML runs with LibTorch or ONNX Runtime,
on a GPU node), and blocked when its input isn't there, or nothing here could
run its inference on the GPU.
"""

import json
import os

from ddml_bench.config import REPO_DIR, ConfigError, load_config
from ddml_bench.layout import GEANT4, TIMING_DIR, Layout, LayoutError
from ddml_bench.models import MODELS_DIR, ModelError, load_models, reference
from ddml_bench.scenarios import scenario_inputs

from .settings import ALL, MODELS, SUBMISSIONS, SettingsError, timing_settings

DONE, TODO, SKIPPED, BLOCKED = "done", "todo", "skipped", "blocked"
STATUSES = (DONE, TODO, SKIPPED, BLOCKED)
# The version of the results' format; results of another aren't current
FORMAT = 1
RESULT, EVENTS, LOG, FAILED = ".json", ".events.csv", ".log", ".failed.json"


class PlanError(Exception):
    pass


def timing_dir(data_dir):
    return os.path.join(data_dir, TIMING_DIR)


def result_name(layout, input_file, seed, events):
    """The name of a task's results: its input's simulation, as the pipeline
    names it, with the number of events timed."""
    return "%s_t%d" % (layout.sample_name(input_file, seed), events)


def simulator_identity(simulator):
    """What a simulator is, as a task's results record it: its detector, and
    for a model its presets or configurations, weights and the sha256 sums of
    its description and own Python code (ddml_bench/models.py)."""
    if simulator["name"] == GEANT4:
        return {"name": GEANT4, "compact": simulator["compact_file"]}
    return {"name": simulator["name"], "compact": simulator["compact_file"],
            "presets": list(simulator["presets"]),
            "model_configs": list(simulator["model_configs"]),
            "weights_md5": simulator.get("weights_md5"),
            "description_sha256": simulator.get("description_sha256"),
            "python": dict((module, code["sha256"])
                           for module, code in sorted((simulator.get("python") or {}).items()))}


def device_simulators(device_name, device, names):
    """The simulators a kind of node times, of those named, in their order."""
    wanted = device["simulators"]
    if wanted == ALL:
        return list(names)
    if wanted == MODELS:
        return [name for name in names if name != GEANT4]
    unknown = [name for name in wanted if name not in names]
    if unknown:
        raise PlanError("timing.devices.%s.simulators has %s, but the simulators are %s"
                        % (device_name, ", ".join(unknown), ", ".join(names)))
    return [name for name in names if name in wanted]


def _real_files(paths):
    found = []
    for path in paths:
        if not os.path.isfile(path):
            raise PlanError("input file not found: %s" % path)
        if os.path.realpath(path) not in found:
            found.append(os.path.realpath(path))
    return found


def make_plan(simulators=(), scenarios=(), devices=(), files=(), events=None, warmup=None,
              force=False):
    """The timing tasks, as a JSON-able dict (with the settings they share).

    simulators, scenarios, devices: only these (default: all of them, each
        kind of node with its own simulators)
    files: only these inputs of the scenarios
    events, warmup: in place of timing.events and timing.warmup_events
    force: time again what is timed already
    """
    try:
        config = load_config()
        settings = timing_settings(config)
        layout = Layout(config["paths"]["data_dir"])
    except (SettingsError, LayoutError) as error:
        raise PlanError(str(error))
    if events is not None:
        settings["events"] = events
    if warmup is not None:
        settings["warmup_events"] = warmup

    every = scenario_inputs(config, layout)
    groups = [group for group in every if group["enabled"]]
    for name in scenarios:
        found = [group for group in every if group["scenario"] == name]
        if not found:
            raise PlanError("there is no scenario '%s'; the configuration has: %s"
                            % (name, ", ".join(group["scenario"] for group in groups)))
        if not found[0]["enabled"]:
            raise PlanError("the scenario %s is disabled (scenarios.%s.enabled is false)"
                            % (name, name))
    if scenarios:
        groups = [group for group in groups if group["scenario"] in scenarios]

    try:
        models = load_models()
    except ModelError as error:
        raise PlanError(str(error))
    every_simulator = [reference(config)] + models
    names = [simulator["name"] for simulator in every_simulator]
    for name in simulators:
        if name not in names:
            raise PlanError("there's no simulator '%s'; the simulators are: %s"
                            % (name, ", ".join(names)))
    for name in devices:
        if name not in settings["devices"]:
            raise PlanError("there's no kind of node '%s' in timing.devices; there are: %s"
                            % (name, ", ".join(settings["devices"])))
    chosen = [(name, device) for name, device in settings["devices"].items()
              if not devices or name in devices]

    only = _real_files(files)
    for path in only:
        if not any(os.path.realpath(entry["file"]) == path
                   for group in groups for entry in group["inputs"]):
            raise PlanError("%s isn't an input of the scenarios timed" % path)

    root = timing_dir(layout.data_dir)
    tasks = []
    for device_name, device in chosen:
        here = [simulator for simulator in every_simulator
                if simulator["name"] in device_simulators(device_name, device, names)
                and (not simulators or simulator["name"] in simulators)]
        for group in groups:
            for entry in group["inputs"]:
                if only and os.path.realpath(entry["file"]) not in only:
                    continue
                for simulator in here:
                    tasks.append(make_task(settings, device_name, device, group["scenario"],
                                           entry, simulator, layout, root))
    keys = [task["key"] for task in tasks]
    doubled = sorted({key for key in keys if keys.count(key) > 1})
    if doubled:
        raise PlanError("two timing tasks would write the same results: %s" % doubled[0])

    slurm = config.get("slurm") or {}
    return {
        "kind": "ddml-timing",
        "repo_dir": REPO_DIR,
        "config_files": [f for f in os.environ.get("FASINATE_CONFIG", "").split(":") if f],
        "data_dir": layout.data_dir,
        "timing_dir": root,
        "settings": settings,
        "force": bool(force),
        "slurm": {"sbatch_args": [str(a) for a in slurm.get("sbatch_args") or []],
                  "log_dir": os.path.join(root, SUBMISSIONS)},
        "tasks": tasks,
    }


def make_task(settings, device_name, device, scenario, entry, simulator, layout, root):
    """The task timing one input (an entry of a scenario's inputs) with one
    simulator on one kind of node."""
    events, warmup = settings["events"], settings["warmup_events"]
    seed = entry["seeds"][0]
    name = result_name(layout, entry["file"], seed, events)
    base = os.path.join(root, device_name, scenario, simulator["name"], name)
    model = None if simulator["name"] == GEANT4 else simulator
    return {
        "key": "/".join((device_name, scenario, simulator["name"], name)),
        "device": device_name,
        "inference": device["inference"],
        "also_cpu_only_models": device["also_cpu_only_models"],
        "scenario": scenario,
        "simulator": simulator["name"],
        "name": name,
        "input": entry["file"],
        "seed": seed,
        "events": events,
        "warmup_events": warmup,
        "result": base + RESULT,
        "events_file": base + EVENTS,
        "log": base + LOG,
        "failed": base + FAILED,
        "compact": simulator["compact_file"],
        "presets": list(model["presets"]) if model else [],
        "model_configs": list(model["model_configs"]) if model else [],
        "model": model,
        "identity": {"events": events, "warmup_events": warmup,
                     "inference": device["inference"], "pin_cpu": settings["pin_cpu"],
                     # relative to the data folder, so the results hold when it is copied
                     "input": layout.relative(entry["file"]), "seed": seed,
                     "simulator": simulator_identity(simulator)},
    }


def read_json(path):
    try:
        with open(path) as stream:
            return json.load(stream)
    except (OSError, ValueError):
        return None


def current(task):
    """(done, why not): whether the task's results are there, of the settings
    the plan has for it."""
    result = read_json(task["result"])
    if result is None:
        failed = read_json(task["failed"])
        if failed and failed.get("reason"):
            return False, "failed last time: %s" % failed["reason"].split("\n")[0]
        return False, "not timed yet"
    if result.get("format") != FORMAT:
        return False, "timed by another version of ddml-timing"
    recorded = result.get("identity") or {}
    differ = [key for key in sorted(set(task["identity"]) | set(recorded))
              if task["identity"].get(key) != recorded.get(key)]
    if differ:
        return False, "timed with other settings (%s)" % ", ".join(differ)
    return True, ""


# ---------------------------------------------------------------------------
# In the environment: what each model's inference runs with, and the statuses


def backend_of(plugin):
    """Which of DDML's inference backends a plugin (its factory) runs, in words."""
    factory = plugin.split("/")[0]
    if "PyEmbedded" in factory:
        return "embedded Python"
    if "ONNX" in factory:
        return "ONNX Runtime"
    if "Torch" in factory:
        return "LibTorch"
    if factory.startswith("LoadHDF5"):
        return "a shower library"
    return "its C++ plugin"


def model_backends(model):
    """[{plugin, backend, python_module}] of each of a model's DDML
    configurations, as ddsim makes them, with DDML's Python package (so in the
    environment); python_module is the module an embedded-Python plugin runs,
    or None."""
    import ddml
    from ddml import configs as presets
    from ddml_bench import ddml_configs
    model_config = getattr(ddml, "ModelConfig", None)
    configs = []
    for preset in model["presets"]:
        settings = getattr(presets, preset, None)
        if model_config is not None and isinstance(settings, model_config) \
                and not preset.startswith("_"):
            configs.append(settings)
    if model["model_configs"]:
        configs += ddml_configs.build(model["model_configs"])
    return [{"plugin": config.plugin, "backend": backend_of(config.plugin),
             "python_module": dict(config.plugin_properties).get("PythonModule")}
            for config in configs]


def cuda_problem():
    """Why no model's inference could run on a GPU with the PyTorch here (on
    a login node, before any job), or None: PyTorch isn't there, or was built
    without CUDA. Whether a GPU is there to use is checked in the job."""
    try:
        import torch
    except Exception as error:  # ImportError, or anything torch raises as it loads
        return ("PyTorch can't be imported here (%s), and a model's inference needs it to run on "
                "the GPU" % error)
    cuda = getattr(getattr(torch, "backends", None), "cuda", None)
    if cuda is None or not cuda.is_built():
        return ("PyTorch %s here was built without CUDA (torch.version.cuda is %s), so no model's "
                "inference can run on a GPU with it" % (
                    getattr(torch, "__version__", "?"),
                    getattr(getattr(torch, "version", None), "cuda", None)))
    return None


def find_status(plan):
    """The plan's tasks, each with its status and why, and for a model what
    its inference runs with (backends, python_modules, inference_on), in this
    environment."""
    tasks = [dict(task) for task in plan["tasks"]]
    backends = {}
    cuda = []  # the problem with CUDA here, once found
    for task in tasks:
        task["python_modules"], task["backends"], task["inference_on"] = [], [], "none"
        if task["model"] is not None:
            name = task["simulator"]
            if name not in backends:
                try:
                    backends[name] = model_backends(task["model"])
                except Exception as error:  # no DDML here, or a configuration it rejects
                    backends[name] = error
            if isinstance(backends[name], Exception) or not backends[name]:
                task["status"], task["why"] = BLOCKED, (
                    "can't tell what its inference runs with: %s" % (
                        backends[name] or "DDML here has none of its presets (%s)"
                        % ", ".join(task["presets"])))
                continue
            task["backends"] = backends[name]
            modules = [b["python_module"] for b in backends[name] if b["python_module"]]
            cpu_only = sorted({b["backend"] for b in backends[name] if not b["python_module"]})
            if task["inference"] == "cuda":
                if not modules:
                    if not task["also_cpu_only_models"]:
                        task["status"], task["why"] = SKIPPED, (
                            "DDML runs its inference with %s, on the CPU only"
                            % " and ".join(cpu_only or ["its C++ plugin"]))
                        continue
                    task["inference_on"] = "cpu (%s)" % " and ".join(cpu_only)
                else:
                    if not cuda:
                        cuda.append(cuda_problem())
                    if cuda[0]:
                        task["status"], task["why"] = BLOCKED, cuda[0]
                        continue
                    task["python_modules"] = modules
                    task["inference_on"] = "cuda" + (" (and %s on the CPU)" % " and ".join(
                        cpu_only) if cpu_only else "")
            else:
                task["python_modules"] = modules
                task["inference_on"] = "cpu"
        if not os.path.isfile(task["input"]):
            task["status"], task["why"] = BLOCKED, (
                "its input isn't there: %s (./generate_input.sh makes the guns' inputs; put inputs "
                "made elsewhere there)" % task["input"])
            continue
        done, why = current(task)
        if done and not plan.get("force"):
            task["status"], task["why"] = DONE, ""
        else:
            task["status"], task["why"] = TODO, "forced" if done else why
    return tasks


def environment_problems(tasks):
    """What in this environment would stop the tasks to do: the pipeline's
    own checks of the Geant4 and models stages (the DDML build, ddsim, the
    compact files, each model's plugins, weights and modules)."""
    from ddml_bench.stages.simulate import Geant4, check_models
    todo = [task for task in tasks if task["status"] == TODO]
    problems = []
    compacts = sorted({task["compact"] for task in todo if task["model"] is None})
    if compacts:
        problems += Geant4().check_environment([{"params": {"compact": c}} for c in compacts])
    models = []
    for task in todo:
        if task["model"] is not None and task["model"]["name"] not in [m["name"] for m in models]:
            models.append(task["model"])
    if models:
        problems += [p for p in check_models(models) if p not in problems]
    return problems


def models_path(task):
    """The folder ddsim imports a model's own inference code from (models/,
    put first by scripts/ddsim_steer_models.py), for a model with DDML model
    configurations written out; None otherwise."""
    return MODELS_DIR if task.get("model_configs") else None


__all__ = ["BLOCKED", "DONE", "SKIPPED", "TODO", "ConfigError", "PlanError", "current",
           "environment_problems", "find_status", "make_plan", "models_path"]
