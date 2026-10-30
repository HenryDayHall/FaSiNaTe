"""Stages 2 and 3: simulate every input with Geant4, and with each model in
models/, one task per input and simulator.

Both run ddsim as DDML's README does: with DDML's steering file from the DDML
build that setup.sh sets up, from its scripts/ folder (presets can give model
paths relative to it), on one thread. A scenario with simulation_seeds has
each input simulated once with each seed (ddsim's --random.seed), or with
one_seed_per_input once with its own seed, each a task of its own; otherwise
ddsim chooses a seed.
A model with presets runs with them (--ml-model); one with DDML model
configurations written out (ddml.model_configs) runs scripts/ddsim_steer_models.py,
which runs DDML's steering file with the configurations made from its --ml-config options
(ddml_bench/ddml_configs.py) in place of presets. Geant4, the reference, runs
scripts/ddsim_steer_geant4.py, which runs DDML's steering file without DDML's
fast simulation. So every other setting (physics list, crossing angle,
filters, ...) is the same for every simulator.

ddsim writes to a hidden file next to the output, renamed once ddsim
succeeds, so an output that exists is complete; its log goes next to it.
"""

import hashlib
import importlib.util
import os
import re
import shutil
import socket
import sys
import time

from .. import ddml_configs, process
from ..config import REPO_DIR
from .base import Stage, StageError, compact_problems, resolve_compact

GEANT4_STEERING = os.path.join(REPO_DIR, "scripts", "ddsim_steer_geant4.py")
# For the models with DDML model configurations written out, not presets
MODELS_STEERING = os.path.join(REPO_DIR, "scripts", "ddsim_steer_models.py")
# A line of a .components file, as DD4hep's plugin service reads it
_COMPONENT = re.compile(r"^\s*(?:(v[0-9]+)::)?([^:]+):(.*\S)\s*$")


class Simulate(Stage):
    # A simulation whose input isn't there, and won't be made, is an error
    blocked_is_error = True
    # Made for the reconstructions, and those for the measurements
    intermediate = True

    def simulators(self, graph):
        """The simulators this stage runs, as models.py describes them."""
        raise NotImplementedError

    def params(self, simulator, layout, scenario, path, events, seed):
        return {"log": layout.simulation_log(scenario, simulator["name"], path, seed),
                "compact": simulator["compact_file"],
                "events": -1 if events is None else events,
                "seed": seed}

    def add_tasks(self, graph):
        layout = graph.layout
        for simulator in self.simulators(graph):
            for scenario, path, source, seeds in graph.inputs:
                # Once with each of the input's seeds, or once with ddsim's own
                for seed in seeds:
                    task = graph.add(
                        self.name, layout.simulation(scenario, simulator["name"], path, seed),
                        [source["key"]] if source else [], scenario=scenario,
                        simulator=simulator["name"], name=layout.sample_name(path, seed),
                        inputs=[path], input_file=path, seed=seed,
                        params=self.params(simulator, layout, scenario, path,
                                           graph.plan["events"], seed))
                    graph.simulations.append(task)

    @staticmethod
    def seed_args(params):
        """ddsim's seed for the simulation, if it has one of its own (otherwise
        ddsim draws one, and writes it in the log)."""
        seed = params.get("seed")
        return [] if seed is None else ["--random.seed=%d" % seed]

    def command(self, task, compact, partial):
        raise NotImplementedError

    def header(self, task):
        raise NotImplementedError

    def run(self, task):
        source = os.environ.get("DDML_SOURCE_DIR")
        if not source:
            raise StageError("DDML isn't set up in this environment; source setup.sh first")
        params = task["params"]
        output, log = task["output"], params["log"]
        folder, name = os.path.split(output)
        partial = os.path.join(folder, ".%s.%d.%s" % (process.host(), os.getpid(), name))
        compact = resolve_compact(params["compact"])
        if compact is None:
            raise StageError("k4geo_DIR isn't set, so the compact file %s can't be found"
                             % params["compact"])
        command = self.command(task, compact, partial)
        cwd = os.path.join(source, "scripts")

        os.makedirs(folder, exist_ok=True)
        seed = params.get("seed")
        with open(log, "w") as stream:
            for line in self.header(task) + [
                    "# input     %s" % task["inputs"][0],
                    "# output    %s, once ddsim succeeds" % output,
                    "# seed      %s" % (seed if seed is not None
                                        else "ddsim's own, chosen at random (see below)"),
                    "# started   %s on %s" % (process.now(), socket.gethostname()),
                    "# Key4hep   %s" % (os.environ.get("DDML_KEY4HEP_RELEASE") or "unknown"),
                    self.ddml_line(source),
                    "# command   (in %s) %s" % (cwd, process.quote(command))]:
                stream.write(line + "\n")
        print("%s:   log: %s" % (self.name, log))

        start = time.time()
        with process.interruptible():
            try:
                status = process.run_logged(command, cwd, log)
            except process.Interrupted:
                process.remove(partial)
                process.append(log, "# INTERRUPTED %s" % process.now())
                raise
        elapsed = int(round(time.time() - start))
        if status != 0:
            why = "ddsim exited with status %d" % status
        elif not os.path.isfile(partial) or os.path.getsize(partial) == 0:
            why = "ddsim wrote no output"
        else:
            try:
                os.replace(partial, output)
                why = None
            except OSError as error:
                why = "could not rename %s to %s: %s" % (partial, output, error.strerror)
        if why is None:
            process.append(log, "# finished  %s, after %d s" % (process.now(), elapsed))
            print("%s:   done in %d s" % (self.name, elapsed))
            return
        process.remove(partial)
        process.append(log, "# FAILED    %s: %s" % (process.now(), why))
        raise StageError("%s; the end of its log, %s:\n%s" % (why, log, process.log_tail(log)))

    def describe(self, task):
        return "the %s simulation of %s" % (task["simulator"], task["name"])

    def label(self, task):
        return "%s %s/%s" % (task["simulator"], task["scenario"], task["name"])


def _ddml_problems(install=True):
    """Problems with the DDML build this environment sets up: it should be the
    one scripts/build_ddml.sh builds for this configuration (its sources, and
    unless install is false, its installation)."""
    env = os.environ
    if not env.get("DDML_SOURCE_DIR"):
        return ["DDML isn't set up in this environment; source setup.sh first"]
    problems = []
    pairs = [("DDML_SOURCE_DIR", "FASINATE_DDML_SOURCE_DIR")]
    if install:
        pairs.append(("DDML_INSTALL_DIR", "FASINATE_DDML_INSTALL_DIR"))
    for var, setting in pairs:
        if not _same_path(env.get(var), env.get(setting)):
            problems.append("%s is %s, but scripts/build_ddml.sh builds DDML in %s for this "
                            "configuration" % (var, env.get(var) or "(unset)",
                                               env.get(setting) or "(unset)"))
    return problems


class Geant4(Simulate):
    name = "geant4"

    def simulators(self, graph):
        return [graph.plan["reference"]]

    def command(self, task, compact, partial):
        params = task["params"]
        # The command the models run, with the Geant4 steering file in place of
        # DDML's (it runs DDML's, leaving out the fast simulation)
        return (["ddsim", "--steeringFile", GEANT4_STEERING, "--compactFile", compact,
                 "--inputFiles", task["inputs"][0], "--outputFile", partial,
                 "--numberOfEvents=%d" % params["events"]] + self.seed_args(params))

    def header(self, task):
        return ["# ddml-bench geant4: Geant4 on %s" % task["scenario"]]

    def ddml_line(self, source):
        return "# DDML      %s, steering from %s" % (
            os.environ.get("DDML_COMMIT") or "unknown",
            os.path.join(source, "scripts", "ddsim_steer.py"))

    def check_environment(self, tasks):
        """This environment should have the DDML build from scripts/build_ddml.sh,
        whose steering file the Geant4 steering runs, ddsim and the detector."""
        problems = _ddml_problems(install=False)
        source = os.environ.get("DDML_SOURCE_DIR")
        if source:
            for steering in (os.path.join(source, "scripts", "ddsim_steer.py"), GEANT4_STEERING):
                if not os.path.isfile(steering):
                    problems.append("steering file not found: %s" % steering)
        if not shutil.which("ddsim"):
            problems.append("ddsim isn't on PATH")
        problems += compact_problems(sorted({t["params"]["compact"] for t in tasks}), "Geant4")
        return problems

    def describe(self, task):
        return "the Geant4 simulation of %s" % task["name"]

    def job_name(self, task):
        return "geant4"

    def log_name(self, task):
        return [self.name, task["scenario"], task["name"]]


class Models(Simulate):
    name = "models"

    def simulators(self, graph):
        return graph.plan["models"]

    def params(self, simulator, layout, scenario, path, events, seed):
        params = Simulate.params(self, simulator, layout, scenario, path, events, seed)
        params.update(presets=list(simulator["presets"]),
                      model_configs=list(simulator["model_configs"]), model=simulator)
        return params

    def command(self, task, compact, partial):
        params = task["params"]
        if params.get("model_configs"):
            # DDML's steering file with these configurations instead of presets
            steering = MODELS_STEERING
            options = []
            for entry in params["model_configs"]:
                options += [ddml_configs.OPTION, ddml_configs.option(entry)]
        else:
            # DDML's README names ddsim_steer_cc3.py, but that only exists on
            # DDML's henry-all-showers branch, and imports get_presets_from_args,
            # which DDML has since replaced; apart from the particle gun, which
            # an input file overrides, it matches ddsim_steer.py
            steering = os.path.join(os.environ["DDML_SOURCE_DIR"], "scripts", "ddsim_steer.py")
            options = []
            for preset in params["presets"]:
                options += ["--ml-model", preset]
        return (["ddsim", "--steeringFile", steering, "--compactFile", compact] + options
                + ["--inputFiles", task["inputs"][0], "--outputFile", partial,
                   "--numberOfEvents=%d" % params["events"]] + self.seed_args(params))

    def header(self, task):
        return ["# ddml-bench models: %s on %s" % (task["simulator"], task["scenario"])]

    def ddml_line(self, source):
        return "# DDML      %s, installed in %s" % (os.environ.get("DDML_COMMIT") or "unknown",
                                                     os.environ.get("DDML_INSTALL_DIR"))

    def check_environment(self, tasks):
        models = []
        for task in tasks:
            if task["params"]["model"]["name"] not in [m["name"] for m in models]:
                models.append(task["params"]["model"])
        return check_models(models)

    def job_name(self, task):
        return task["simulator"]

    def log_name(self, task):
        return [self.name, task["simulator"], task["scenario"], task["name"]]


# ---------------------------------------------------------------------------
# What the models need from the DDML build


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


def say(message):
    print("models: " + message, file=sys.stderr)


def check_models(models):
    """Problems running the models in this environment: it should run the DDML
    build from scripts/build_ddml.sh, and have what each model needs from it.
    Reports what each model runs."""
    problems = _ddml_problems()
    if not os.environ.get("DDML_SOURCE_DIR") or not os.environ.get("DDML_INSTALL_DIR"):
        return problems or ["DDML isn't set up in this environment; source setup.sh first"]
    source = os.environ["DDML_SOURCE_DIR"]
    lib_dir = os.path.join(os.environ["DDML_INSTALL_DIR"], "lib")

    try:
        import ddml
        from ddml import configs as presets
    except Exception as error:  # ImportError, or anything the package raises
        return problems + ["cannot import DDML's Python package, ddml: %s" % error]
    ddml_dir = os.path.dirname(os.path.realpath(ddml.__file__))
    if not _same_path(ddml_dir, os.path.join(source, "python", "ddml")):
        problems.append("the ddml Python package comes from %s, not from the DDML build's "
                        "sources in %s" % (ddml_dir, source))
    model_config = getattr(ddml, "ModelConfig", None)
    if model_config is None:
        return problems + ["the ddml Python package in %s has no presets (ddml.ModelConfig); "
                           "is it DDML's python-inference branch?" % ddml_dir]

    factories = _plugin_factories()
    for model in models:
        name = model["name"]
        problems += compact_problems([model["compact_file"]], name)
        # What DDML is to run: (what, its ModelConfig), for each preset, or each
        # configuration as the steering file makes it (ddml_bench/ddml_configs.py)
        configs = []
        for preset in model["presets"]:
            settings = getattr(presets, preset, None)
            if preset.startswith("_") or not isinstance(settings, model_config):
                problems.append("%s: DDML has no preset %s (see %s)" % (
                    name, preset, os.path.join(source, "python", "ddml", "configs.py")))
                continue
            configs.append(("preset %s" % preset, settings))
        if model["model_configs"]:
            try:
                built = ddml_configs.build(model["model_configs"])
            except Exception as error:  # ModelConfigError, or no g4units
                problems.append("%s: can't make its DDML model configurations (%s): %s"
                                % (name, model["config"], error))
                built = []
            for number, settings in enumerate(built, 1):
                configs.append(("model configuration %d" % number, settings))
                say("%s: model configuration %d: %s" % (name, number,
                                                       ddml_configs.describe(settings)))

        loads = []  # (what, the file its ModelPath names)
        for what, settings in configs:
            plugin = settings.plugin.split("/")[0]
            if plugin not in factories:
                problems.append("%s: %s needs the plugin %s, which no .components file on "
                                "LD_LIBRARY_PATH declares; was DDML built with the inference "
                                "backend it needs?" % (name, what, plugin))
            else:
                directory, library = factories[plugin]
                loaded = _first_on_library_path(library)
                if not (_same_path(directory, lib_dir) and loaded
                        and _same_path(os.path.dirname(loaded), lib_dir)):
                    problems.append("%s: the plugin %s would be loaded from %s, not from the "
                                    "DDML build in %s" % (
                                        name, plugin, loaded or os.path.join(directory, library),
                                        lib_dir))
                else:
                    say("%s: %s runs %s from %s" % (name, what, plugin, loaded))
            particles = sorted(settings.applicable_particles)
            if particles != model["particles"]:
                problems.append("%s: %s gives %s to the model, but %s says %s" % (
                    name, what, ", ".join(particles) or "nothing", model["config"],
                    "it was trained on %s" % model["particle"]
                    if model["particles"] == [model["particle"]]
                    else "DDML gives it %s (ddml.particles)" % ", ".join(model["particles"])))
            module = settings.plugin_properties.get("PythonModule")
            if module:
                origin = _module_origin(module)
                if origin:
                    say("%s: %s runs the Python module %s" % (name, what, origin))
                else:
                    problems.append("%s: %s runs the Python module %s, which isn't on "
                                    "PYTHONPATH" % (name, what, module))
            model_path = settings.plugin_properties.get("ModelPath")
            if isinstance(model_path, str) and model_path:
                # As the plugin opens it, from DDML's scripts/ folder
                loads.append((what, os.path.normpath(os.path.join(source, "scripts",
                                                                  model_path))))
        if model["particles"] != [model["particle"]]:
            say("%s: trained on %s, but DDML gives it %s (ddml.particles in %s)" % (
                name, model["particle"], ", ".join(model["particles"]), model["config"]))

        weights = os.path.join(source, model["weights"]) if model["weights"] else None
        if weights:
            md5 = _md5(weights) if model["weights_md5"] and os.path.isfile(weights) else None
            if not os.path.isfile(weights):
                problems.append("%s: weights not found: %s (building DDML downloads them when "
                                "ddml.download_models is true, those DDML's build doesn't when "
                                "they are in ddml.extra_models; scripts/build_ddml.sh --clean "
                                "builds again)" % (name, weights))
            elif md5 and md5 != model["weights_md5"]:
                problems.append("%s: the md5 sum of %s is %s, not %s as in %s; if these weights "
                                "are intended, update weights_md5 there" % (
                                    name, weights, md5, model["weights_md5"], model["config"]))
            else:
                say("%s: weights %s%s" % (name, weights, ", md5 checked" if md5 else ""))
        # The weights whose md5 sum is checked should be what the plugins load
        for what, path in loads:
            if weights and not _same_path(path, weights):
                problems.append("%s: %s loads %s, but the weights %s checks are %s; make "
                                "ModelPath and ddml.weights the same file" % (
                                    name, what, path, model["config"], weights))
            elif not os.path.isfile(path):
                problems.append("%s: %s loads %s, which isn't there" % (name, what, path))
    return problems
