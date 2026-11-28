"""The checks of a model before it is simulated: the preflight stage
(ddml_bench/stages/preflight.py, ./preflight_models.sh), whose task for each
model in models/ its simulations wait for. In order, each only once those
before it pass:

  1. description  models/<name>.yaml is well formed YAML (no key twice in a
                  mapping) and has every field the pipeline needs
                  (ddml_bench/models.py's load_model()), and ddml.weights,
                  ddml.weights_md5 and, if set, a known ddml.weights_in
  2. weights      the weights are there, with that md5 sum
  3. Python code  the model's own inference code, the modules of models/ its
                  DDML model configurations run (plugin_properties
                  PythonModule), if it has any: each compiles, and conforms to
                  Black, whichever the environment has (the Key4hep
                  release's), its version noted in the summary
  4. import       each of those modules imports, from models/ as ddsim imports
                  it, in a Python of its own (preflight.import_timeout), with
                  its entry point (EntryPoint, run_inference by default)
  5. barrel       DDML has what the model needs (the models stage's check of
                  the environment), and if one of its presets or configurations
                  places showers in the ECal barrel (preflight.barrel.detector),
                  each particle it gives to the model there is shot from
                  preflight.barrel.position, once in each direction of
                  preflight.barrel.directions, with preflight.barrel.energy,
                  each shot an event, all in one ddsim run as the models stage
                  runs it: every shot must have DDML place a shower of the
                  model in the barrel (DDML's line for each, which the timing
                  benchmark counts too) and deposit energy in the hits of
                  preflight.barrel.hit_collections. Geant4 simulates whatever
                  the model doesn't take, so energy alone wouldn't show that
                  the model ran. The hits of each particle's shots are drawn,
                  for a look by eye.

Everything goes to <paths.data_dir>/preflight_output/<model>/ (made anew each
time): summary.txt (and summary.json), preflight.log with the whole output
of every step (Black's diff, the import's traceback, the check of DDML),
barrel_<particle>.png, and barrel/ with the shots' input, ddsim's log and its
simulation. preflight_output/summary.txt sums up every model's last
preflight. A model that passes gets passed_<digest>.json, which marks its task
done until what it was checked as changes: digest() of its description, its
own code, its weights, the preflight settings and the software pinned.

Step 1 needs only PyYAML, so it also runs before a plan is made
(check_description()); the rest runs in the environment from setup.sh.
"""

import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time

from .config import REPO_DIR

# Change this when the checks change, so that every model is checked again
VERSION = 1
STEPS = ((1, "description"), (2, "weights"), (3, "Python code"), (4, "import"), (5, "barrel"))
PASSED, FAILED, SKIPPED, NOT_RUN = "passed", "FAILED", "skipped", "not run"
SUMMARY, SUMMARY_JSON, LOG, BARREL = "summary.txt", "summary.json", "preflight.log", "barrel"
WEIGHTS_IN = ("ddml", "fasinate")
# The PDG codes of the particles DDML can give a model, by their Geant4 names
GEANT4_PDG = {"gamma": 22, "e-": 11, "e+": -11, "mu-": 13, "mu+": -13, "pi+": 211,
              "pi-": -211, "pi0": 111, "kaon+": 321, "kaon-": -321, "kaon0L": 130,
              "kaon0S": 310, "proton": 2212, "anti_proton": -2212, "neutron": 2112,
              "anti_neutron": -2112}
# The mass (GeV) and charge of those the particle package may lack
_KNOWN = {22: (0.0, 0.0)}
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class PreflightError(Exception):
    """The preflight settings are wrong."""


class StepFailed(Exception):
    pass


class StepSkipped(Exception):
    pass


# ---------------------------------------------------------------------------
# The settings, and what a model is checked as


def settings(config):
    """The preflight section of the configuration, checked."""
    section = config.get("preflight")
    if not isinstance(section, dict):
        raise PreflightError("the configuration has no preflight section")
    timeout = section.get("import_timeout")
    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or timeout <= 0:
        raise PreflightError("preflight.import_timeout should be a number of seconds")
    barrel = section.get("barrel")
    if not isinstance(barrel, dict):
        raise PreflightError("preflight.barrel should be the settings of the barrel shots")

    def vector(value, what):
        if not (isinstance(value, list) and len(value) == 3 and all(
                isinstance(v, (int, float)) and not isinstance(v, bool) for v in value)):
            raise PreflightError("preflight.barrel.%s should be [x, y, z]" % what)
        return [float(v) for v in value]

    directions = barrel.get("directions")
    if not isinstance(directions, dict) or not directions:
        raise PreflightError("preflight.barrel.directions should be {name: [x, y, z]}")
    found = {}
    for name, value in directions.items():
        if not isinstance(name, str) or not _IDENTIFIER.match(name):
            raise PreflightError("preflight.barrel.directions: %r isn't a name" % (name,))
        found[name] = vector(value, "directions.%s" % name)
        if not any(found[name]):
            raise PreflightError("preflight.barrel.directions.%s is no direction" % name)
    energy = barrel.get("energy")
    if not isinstance(energy, (int, float)) or isinstance(energy, bool) or energy <= 0:
        raise PreflightError("preflight.barrel.energy should be the shots' energy in GeV")
    collections = barrel.get("hit_collections")
    if not (isinstance(collections, list) and collections
            and all(isinstance(c, str) and c for c in collections)):
        raise PreflightError("preflight.barrel.hit_collections should be a list of "
                             "collections of simulated calorimeter hits")
    detector = barrel.get("detector")
    if not isinstance(detector, str) or not detector:
        raise PreflightError("preflight.barrel.detector should name DDML's ECal barrel")
    seed = barrel.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise PreflightError("preflight.barrel.seed should be ddsim's seed for the shots")
    return {"version": VERSION,
            "import_timeout": float(timeout),
            "barrel": {"detector": detector, "energy": float(energy),
                       "position": vector(barrel.get("position"), "position"),
                       "directions": found, "hit_collections": list(collections),
                       "seed": seed}}


def digest(model, preflight, pins):
    """What a model is checked as, in 16 hex digits: its description (as
    read, without what only the plots use), its own Python code, its weights
    and their md5 sum, the preflight settings and the software pinned (pins:
    the plan's published pins). Its task is done once it passes as that."""
    what = {"version": VERSION, "description": model.get("description_sha256"),
            "python": dict((module, code["sha256"])
                           for module, code in sorted((model.get("python") or {}).items())),
            "weights": [model.get("weights"), model.get("weights_in"),
                        model.get("weights_md5")],
            "preflight": preflight, "pins": pins}
    text = json.dumps(what, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Step 1, which needs only PyYAML


def _unique_loader():
    """A YAML loader (PyYAML's safe one) that refuses a key twice in a
    mapping, which PyYAML's own takes, keeping the last."""
    import yaml

    class Loader(yaml.SafeLoader):
        pass

    def mapping(loader, node, deep=False):
        seen = set()
        for key_node, _ in node.value:
            key = loader.construct_object(key_node, deep=True)
            try:
                twice = key in seen
                seen.add(key)
            except TypeError:  # a key that can't be hashed, which construct_mapping reports
                continue
            if twice:
                raise yaml.constructor.ConstructorError(
                    "while reading a mapping", node.start_mark,
                    "the key %r is in it twice" % (key,), key_node.start_mark)
        return loader.construct_mapping(node, deep=deep)

    Loader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    return Loader


def check_description(name):
    """Step 1: (the model as load_model() reads it, lines of detail); raises
    StepFailed with what is wrong."""
    from .models import MODELS_DIR, ModelError, load_model
    path = os.path.join(MODELS_DIR, name + ".yaml")
    source = os.path.relpath(path, REPO_DIR)
    try:
        import yaml
    except ImportError:
        raise StepFailed("PyYAML is needed to read %s" % source)
    try:
        with open(path) as stream:
            yaml.load(stream, Loader=_unique_loader())
    except OSError as error:
        raise StepFailed("cannot read %s: %s" % (source, error))
    except yaml.YAMLError as error:
        raise StepFailed("%s isn't well formed YAML: %s" % (source, error))
    try:
        model = load_model(name)
    except ModelError as error:
        raise StepFailed(str(error))
    missing = [key for key in ("weights", "weights_md5") if not model[key]]
    if missing:
        raise StepFailed("%s: %s missing; preflight checks the model's weights, so it needs "
                         "both ddml.weights, their path, and ddml.weights_md5, their md5 sum"
                         % (source, " and ".join("ddml." + key for key in missing)
                            + (" is" if len(missing) == 1 else " are")))
    if model["weights_in"] not in WEIGHTS_IN:
        raise StepFailed("%s: ddml.weights_in should be %s (where the weights are: DDML's "
                         "clone, the default, or this repository), not %r"
                         % (source, " or ".join(WEIGHTS_IN), model["weights_in"]))
    return model, ["%s is well formed, with every field needed" % source]


# ---------------------------------------------------------------------------
# The log, and the steps in the environment


class Log(object):
    """preflight.log: everything each step did, with a line of each on the
    task's output too."""

    def __init__(self, path):
        self.path = path
        open(path, "w").close()

    def write(self, text):
        with open(self.path, "a") as stream:
            stream.write(text if text.endswith("\n") else text + "\n")

    def say(self, text):
        print("preflight: " + text)
        sys.stdout.flush()
        self.write(text)


def weights_file(model):
    """Where a model's weights are: in DDML's clone ($DDML_SOURCE_DIR), or in
    this repository (ddml.weights_in: fasinate)."""
    if model["weights_in"] == "fasinate":
        return os.path.join(REPO_DIR, model["weights"])
    source = os.environ.get("DDML_SOURCE_DIR")
    if not source:
        raise StepFailed("DDML isn't set up in this environment, so its clone, where %s's "
                         "weights are, can't be found" % model["name"])
    return os.path.join(source, model["weights"])


def md5_of(path):
    digest_ = hashlib.md5()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest_.update(block)
    return digest_.hexdigest()


def check_weights(model, log):
    """Step 2."""
    path = weights_file(model)
    if not os.path.isfile(path):
        raise StepFailed("the weights aren't there: %s%s" % (
            path, "" if model["weights_in"] == "fasinate" else
            " (building DDML downloads them when ddml.download_models is true, those its "
            "build doesn't when they are in ddml.extra_models)"))
    found = md5_of(path)
    log.write("weights %s, md5 %s" % (path, found))
    if found != model["weights_md5"]:
        raise StepFailed("the md5 sum of %s is %s, not %s as in %s; if these weights are "
                         "intended, update ddml.weights_md5 there"
                         % (path, found, model["weights_md5"], model["config"]))
    return ["%s, md5 %s" % (os.path.relpath(path, REPO_DIR) if model["weights_in"] == "fasinate"
                            else path, found)]


def own_code(model):
    """[(module, its file)] of the model's own Python code."""
    return [(module, os.path.join(REPO_DIR, code["path"]))
            for module, code in sorted((model.get("python") or {}).items())]


def black_command():
    """The command running Black here, or None."""
    try:
        if importlib.util.find_spec("black") is not None:
            return [sys.executable, "-m", "black"]
    except (ImportError, ValueError):
        pass
    found = shutil.which("black")
    return [found] if found else None


def _run(command, log, timeout=None, cwd=None):
    """(exit status, output) of a command, its output in the log too."""
    log.write("$ " + " ".join(command))
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                universal_newlines=True, timeout=timeout, cwd=cwd,
                                stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired as error:
        output = error.output or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", "replace")
        log.write(output)
        return None, output
    log.write(result.stdout or "(no output)")
    return result.returncode, result.stdout or ""


def check_code(model, preflight, log, record=None):
    """Step 3."""
    code = own_code(model)
    if not code:
        raise StepSkipped("the model runs no Python module of models/ (DDML's own code)")
    for module, path in code:
        with open(path, "rb") as stream:
            text = stream.read()
        try:
            ast.parse(text, filename=path)
            compile(text, path, "exec")
        except (SyntaxError, ValueError) as error:
            raise StepFailed("%s doesn't compile: %s" % (os.path.relpath(path, REPO_DIR), error))
    command = black_command()
    if command is None:
        raise StepFailed("Black isn't found here, so the code can't be checked against it")
    status, output = _run(command + ["--version"], log, timeout=120)
    match = re.search(r"\b(\d+)\.(\d+)(?:\.\d+)?", output or "")
    if status != 0 or not match:
        raise StepFailed("Black doesn't run here (see %s)" % LOG)
    version = match.group(0)
    if record is not None:
        record["black"] = version
    log.write("checked with Black %s (%s)" % (version, " ".join(command)))
    paths = [path for _, path in code]
    status, output = _run(command + ["--check", "--diff", "--quiet"] + paths, log,
                          timeout=300, cwd=REPO_DIR)
    if status != 0:
        changed = [os.path.relpath(p, REPO_DIR) for p in paths
                   if ("+++ %s" % p) in output or ("--- %s" % p) in output]
        raise StepFailed("%s %s reformatted by Black %s (its diff is in %s): run black %s"
                         % (", ".join(changed or [os.path.relpath(p, REPO_DIR) for p in paths]),
                            "would be" if status == 1 else "can't be", version, LOG,
                            " ".join(changed or [os.path.relpath(p, REPO_DIR) for p in paths])))
    return ["%s compile%s and conform%s to Black %s" % (
        ", ".join(os.path.relpath(p, REPO_DIR) for p in paths), "" if len(paths) > 1 else "s",
        "" if len(paths) > 1 else "s", version)]


def entry_points(model):
    """{module: [the names DDML calls in it]} of the model's own code: each
    configuration's EntryPoint, from its preset's where it sets none, and
    run_inference where neither does."""
    presets = None
    try:
        from ddml import configs as presets
    except Exception:  # no DDML here; its presets then give none
        presets = None
    found = {}
    for entry in model["model_configs"]:
        properties = entry.get("plugin_properties") or {}
        module = properties.get("PythonModule")
        if not module or module not in (model.get("python") or {}):
            continue
        name = properties.get("EntryPoint")
        if name is None and entry.get("preset") and presets is not None:
            base = getattr(presets, entry["preset"], None)
            name = dict(getattr(base, "plugin_properties", None) or {}).get("EntryPoint")
        name = name or "run_inference"
        if name not in found.setdefault(module, []):
            found[module].append(name)
    return found


# Run in a Python of its own: import the module from models/, as ddsim does
# (scripts/ddsim_steer_models.py puts models/ first), and find its entry points
_IMPORT = """\
import importlib, json, os, sys
models, module, expected, names = json.loads(sys.argv[1])
sys.path.insert(0, models)
imported = importlib.import_module(module)
where = os.path.realpath(getattr(imported, "__file__", "") or "")
if where != os.path.realpath(expected):
    sys.exit("preflight: %s was imported from %s, not %s" % (module, where, expected))
for name in names:
    if not callable(getattr(imported, name, None)):
        sys.exit("preflight: %s has no function %s, which DDML calls" % (module, name))
print("preflight: imported %s from %s, with %s" % (module, where, ", ".join(names)))
"""


def check_import(model, preflight, log):
    """Step 4."""
    code = own_code(model)
    if not code:
        raise StepSkipped("the model runs no Python module of models/")
    names = entry_points(model)
    models_dir = os.path.join(REPO_DIR, "models")
    source = os.environ.get("DDML_SOURCE_DIR")
    cwd = os.path.join(source, "scripts") if source and os.path.isdir(
        os.path.join(source, "scripts")) else REPO_DIR
    details = []
    for module, path in code:
        wanted = names.get(module) or ["run_inference"]
        argument = json.dumps([models_dir, module, path, wanted])
        status, output = _run([sys.executable, "-c", _IMPORT, argument], log,
                              timeout=preflight["import_timeout"], cwd=cwd)
        if status is None:
            raise StepFailed("importing %s took longer than preflight.import_timeout (%g s)"
                             % (module, preflight["import_timeout"]))
        if status != 0:
            lines = [line for line in output.strip().splitlines() if line.strip()]
            raise StepFailed("%s can't be imported: %s (the traceback is in %s)"
                             % (module, lines[-1] if lines else "exit status %d" % status, LOG))
        details.append("%s imports, with %s" % (module, ", ".join(wanted)))
    return details


# ---------------------------------------------------------------------------
# Step 5, the barrel shots


def ddml_configs_of(model):
    """[(what, DDML's ModelConfig)] of the model's presets or configurations,
    with DDML's Python package."""
    from . import ddml_configs
    import ddml
    from ddml import configs as presets
    model_config = getattr(ddml, "ModelConfig", None)
    found = []
    for preset in model["presets"]:
        settings_ = getattr(presets, preset, None)
        if preset.startswith("_") or model_config is None \
                or not isinstance(settings_, model_config):
            raise StepFailed("DDML has no preset %s" % preset)
        found.append(("preset %s" % preset, settings_))
    if model["model_configs"]:
        try:
            built = ddml_configs.build(model["model_configs"])
        except Exception as error:  # ModelConfigError, or no g4units
            raise StepFailed("its DDML model configurations can't be made: %s" % error)
        found += [("model configuration %d" % n, s) for n, s in enumerate(built, 1)]
    return found


def particle_properties(name):
    """(PDG code, mass in GeV, charge) of a particle, by its Geant4 name."""
    if name not in GEANT4_PDG:
        raise StepFailed("DDML gives the model %s in the barrel, which preflight can't shoot "
                         "yet: add its PDG code to GEANT4_PDG in ddml_bench/preflight.py" % name)
    pdg = GEANT4_PDG[name]
    if pdg in _KNOWN:
        return (pdg,) + _KNOWN[pdg]
    import particle
    found = particle.Particle.from_pdgid(pdg)
    if found.mass is None or found.charge is None:
        raise StepFailed("the particle package has no mass or charge for %s (%d)" % (name, pdg))
    return pdg, found.mass / 1000.0, float(found.charge)


def shots(particles, barrel):
    """The barrel shots, an event each: each particle in each direction."""
    import math
    found = []
    for name in particles:
        pdg, mass, charge = particle_properties(name)
        momentum = math.sqrt(max(barrel["energy"] ** 2 - mass ** 2, 0.0))
        for direction, vector in barrel["directions"].items():
            length = math.sqrt(sum(v * v for v in vector))
            found.append({"event": len(found), "particle": name, "pdg": pdg, "mass": mass,
                          "charge": charge, "direction": direction, "vector": vector,
                          "momentum": [momentum * v / length for v in vector],
                          "position": list(barrel["position"]), "energy": barrel["energy"]})
    return found


def write_shots(path, events):
    """The shots as an LCIO file of MC particles, as the particle guns write
    theirs."""
    from array import array
    from pyLCIO import EVENT, IMPL, IOIMPL
    writer = IOIMPL.LCFactory.getInstance().createLCWriter()
    writer.open(path, EVENT.LCIO.WRITE_NEW)
    run = IMPL.LCRunHeaderImpl()
    run.parameters().setValue("Generator", "ddml_bench/preflight.py")
    writer.writeRunHeader(run)
    for shot in events:
        collection = IMPL.LCCollectionVec(EVENT.LCIO.MCPARTICLE)
        event = IMPL.LCEventImpl()
        event.setEventNumber(shot["event"])
        event.addCollection(collection, "MCParticle")
        particle = IMPL.MCParticleImpl()
        particle.setGeneratorStatus(1)
        particle.setMass(shot["mass"])
        particle.setPDG(shot["pdg"])
        particle.setMomentum(array("f", shot["momentum"]))
        particle.setCharge(shot["charge"])
        particle.setVertex(array("d", shot["position"]))
        particle.setEndpoint(array("d", shot["position"]))
        collection.addElement(particle)
        writer.writeEvent(event)
    writer.close()


# ddsim's line as each event starts, and DDML's for each shower of a model it
# places in the barrel (which the timing benchmark counts too,
# ddml_timing/events.py)
EVENT_START = re.compile(r"\+\+\+ Initializing event (\d+)\.")
BARREL_SHOWER = re.compile(r"PolyhedraBarrelGeometry::localToGlobal\s+INFO\b")


def showers_per_event(log_path, count):
    """How many showers DDML placed in the barrel in each of the first count
    events, from ddsim's log."""
    showers = []
    with open(log_path, errors="replace") as stream:
        for line in stream:
            if EVENT_START.search(line):
                showers.append(0)
            elif BARREL_SHOWER.search(line) and showers:
                showers[-1] += 1
    found = showers[:count]
    return found + [0] * (count - len(found))


def read_hits(path, collections):
    """([{collection: (x, y, z, energy) lists}] of each event, the collections
    the file has) of an EDM4hep simulation, with podio."""
    from podio import root_io
    reader = root_io.Reader(path)
    events, available = [], set()
    for frame in reader.get("events"):
        names = set(str(name) for name in frame.getAvailableCollections())
        available |= names
        found = {}
        for name in collections:
            if name not in names:
                continue
            x, y, z, energy = [], [], [], []
            for hit in frame.get(name):
                position = hit.getPosition()
                x.append(float(position.x))
                y.append(float(position.y))
                z.append(float(position.z))
                energy.append(float(hit.getEnergy()))
            found[name] = (x, y, z, energy)
        events.append(found)
    return events, sorted(available)


def _file_name(particle):
    return re.sub(r"[^A-Za-z0-9_+-]", "_", particle)


# Each panel shows this far (mm) about the shower's centre
VIEW_MM = 150.0


def plot_shots(path, particle, results, barrel):
    """A figure of one particle's shots: each shot's hits from the side (z, y)
    and from along the beam (x, y), coloured by their energy."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    # One hue, light to dark, for the hits' energies (a magnitude)
    ramp = LinearSegmentedColormap.from_list("energy", ["#9ecae9", "#3182bd", "#08306b"])
    rows = len(results)
    figure, axes = plt.subplots(rows, 2, figsize=(11, 5.2 * rows), squeeze=False)
    for row, result in enumerate(results):
        hits = [values for values in result["hits"].values()]
        x, y, z, energy = (np.array(sum((list(h[i]) for h in hits), []), dtype=float)
                           for i in range(4))
        vector = ", ".join("%g" % v for v in result["vector"])
        mark = "" if result["passed"] else "  [FAILED]"
        title = "%s %s (%s): %.3g GeV in %d hits, %d shower(s) placed by the model%s" % (
            particle, result["direction"], vector, result["energy_deposited"],
            result["hit_count"], result["showers"], mark)
        for column, (horizontal, name) in enumerate(((z, "z [mm]"), (x, "x [mm]"))):
            axis = axes[row][column]
            axis.set_xlabel(name)
            axis.set_ylabel("y [mm]")
            axis.grid(True, color="#e6e6e6", linewidth=0.6)
            axis.set_axisbelow(True)
            positive = energy > 0
            if not positive.any():
                axis.text(0.5, 0.5, "no energy deposited", ha="center", va="center",
                          transform=axis.transAxes, color="#555555")
                continue
            order = np.argsort(energy[positive])
            low, high = energy[positive].min(), energy[positive].max()
            points = axis.scatter(horizontal[positive][order], y[positive][order],
                                  c=energy[positive][order], s=6, linewidths=0, cmap=ramp,
                                  norm=LogNorm(vmin=low, vmax=max(high, low * 1.0001)))
            # About the shower's energy-weighted centre, as backscatter can
            # reach far from it
            weights = energy[positive]
            centre = [np.average(values[positive], weights=weights)
                      for values in (horizontal, y)]
            axis.set_xlim(centre[0] - VIEW_MM, centre[0] + VIEW_MM)
            axis.set_ylim(centre[1] - VIEW_MM, centre[1] + VIEW_MM)
            axis.set_aspect("equal")
            outside = int(np.sum((np.abs(horizontal[positive] - centre[0]) > VIEW_MM)
                                 | (np.abs(y[positive] - centre[1]) > VIEW_MM)))
            if outside:
                axis.text(0.01, 0.01, "%d hit(s) outside this view" % outside, fontsize=8,
                          color="#555555", transform=axis.transAxes)
            if column == 1:
                figure.colorbar(points, ax=axis, label="hit energy [GeV]")
        axes[row][0].set_title(title, loc="left", fontsize=9)
    figure.suptitle("Preflight barrel shots of %s: %g GeV from (%s) mm, hits of %s" % (
        particle, barrel["energy"], ", ".join("%g" % v for v in barrel["position"]),
        ", ".join(barrel["hit_collections"])), fontsize=10)
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)


def check_barrel(model, preflight, folder, log, seed=None):
    """Step 5: (details, plots)."""
    from .models import python_summary
    from .stages.base import resolve_compact
    from .stages.simulate import Models, check_models
    from . import process
    barrel = preflight["barrel"]

    # DDML has what the model needs, as the models stage checks it
    captured = io.StringIO()
    with contextlib.redirect_stderr(captured):
        problems = check_models([model])
    log.write(captured.getvalue() or "(the check of DDML printed nothing)")
    if problems:
        for problem in problems:
            log.write("problem: " + problem)
        raise StepFailed("this environment can't run the model: %s" % "; ".join(problems))
    details = ["DDML has what the model needs"]

    configs = ddml_configs_of(model)
    in_barrel = [(what, config) for what, config in configs
                 if config.geometry.ecal.detector == barrel["detector"]]
    if not in_barrel:
        raise StepSkipped("DDML has the model place no showers in the ECal barrel (%s); %s"
                          % (barrel["detector"], details[0]))
    particles = sorted(set(p for _, config in in_barrel for p in config.applicable_particles))
    log.write("in the barrel: %s, for %s" % (", ".join(what for what, _ in in_barrel),
                                             ", ".join(particles)))
    events = shots(particles, barrel)
    work = os.path.join(folder, BARREL)
    os.makedirs(work, exist_ok=True)
    source = os.path.join(work, "shots.slcio")
    write_shots(source, events)
    with open(os.path.join(work, "shots.json"), "w") as stream:
        json.dump(events, stream, indent=1)
    output = os.path.join(work, "shots.root")
    partial = os.path.join(work, ".%s.%d.shots.root" % (process.host(), os.getpid()))
    ddsim_log = os.path.join(work, "ddsim.log")
    compact = resolve_compact(model["compact_file"])
    if compact is None:
        raise StepFailed("k4geo_DIR isn't set, so the compact file %s can't be found"
                         % model["compact_file"])
    task = {"inputs": [source], "params": {
        "presets": list(model["presets"]), "model_configs": list(model["model_configs"]),
        "events": -1, "seed": barrel["seed"] if seed is None else seed}}
    command = Models().command(task, compact, partial)
    cwd = os.path.join(os.environ["DDML_SOURCE_DIR"], "scripts")
    with open(ddsim_log, "w") as stream:
        stream.write("# ddml-bench preflight: %s, %d barrel shot(s)\n# python    %s\n"
                     "# command   (in %s) %s\n" % (
                         model["name"], len(events), python_summary(model.get("python") or {}),
                         cwd, process.quote(command)))
    log.write("ddsim, its output in %s:\n  (in %s) %s" % (ddsim_log, cwd, process.quote(command)))
    log.say("%s: simulating %d barrel shot(s) (%s) with ddsim" % (
        model["name"], len(events), ", ".join(particles)))
    with process.interruptible():
        try:
            status = process.run_logged(command, cwd, ddsim_log)
        except process.Interrupted:
            process.remove(partial)
            raise
    if status != 0 or not os.path.isfile(partial) or os.path.getsize(partial) == 0:
        process.remove(partial)
        log.write(process.log_tail(ddsim_log, 40))
        raise StepFailed("ddsim %s (its log: %s):\n%s" % (
            "exited with status %d" % status if status != 0 else "wrote no output",
            os.path.relpath(ddsim_log, folder), process.log_tail(ddsim_log)))
    os.replace(partial, output)
    showers = showers_per_event(ddsim_log, len(events))
    hits, available = read_hits(output, barrel["hit_collections"])
    log.write("the simulation's collections: %s" % (", ".join(available) or "none"))
    if len(hits) != len(events):
        raise StepFailed("ddsim simulated %d of the %d shots (its log: %s)"
                         % (len(hits), len(events), os.path.relpath(ddsim_log, folder)))
    results, failures = [], []
    for shot, count, found in zip(events, showers, hits):
        energy = sum(sum(values[3]) for values in found.values())
        number = sum(len(values[3]) for values in found.values())
        passed = count > 0 and energy > 0
        result = dict(shot, showers=count, energy_deposited=energy, hit_count=number,
                      passed=passed, hits=found)
        results.append(result)
        line = "%s %s (%s): %d shower(s) placed by the model, %.4g GeV in %d hits" % (
            shot["particle"], shot["direction"], ", ".join("%g" % v for v in shot["vector"]),
            count, energy, number)
        log.write(line + ("" if passed else "  <- FAILED"))
        if not passed:
            why = []
            if count == 0:
                why.append("DDML placed no shower of the model (Geant4 simulated it)")
            if energy <= 0:
                why.append("no energy was deposited in %s" % ", ".join(barrel["hit_collections"]))
            failures.append("%s %s: %s" % (shot["particle"], shot["direction"], "; ".join(why)))
        details.append(line)
    if failures and not set(available) & set(barrel["hit_collections"]):
        failures.append("the simulation has none of preflight.barrel.hit_collections (%s), "
                        "only %s" % (", ".join(barrel["hit_collections"]),
                                     ", ".join(available) or "none"))
    plots = []
    for particle in particles:
        path = os.path.join(folder, "barrel_%s.png" % _file_name(particle))
        plot_shots(path, particle, [r for r in results if r["particle"] == particle], barrel)
        plots.append(path)
    if failures:
        raise StepFailed("\n".join(failures), plots)
    return details, plots


# ---------------------------------------------------------------------------
# A model's preflight, and the summaries


def run_model(model, preflight, folder, marker, planned, context=None):
    """A model's preflight, into folder (made anew): returns its record, as
    summary.json has it, writing marker (its passed_<digest>.json) if it
    passes. planned is what the plan read the model as (its
    description_sha256 and own code), which it should still be."""
    if os.path.isdir(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)
    log = Log(os.path.join(folder, LOG))
    context = context or {}
    record = {"model": model["name"], "config": model["config"], "status": None,
              "failed_step": None, "digest": os.path.basename(marker)[len("passed_"):-5],
              "started": time.strftime("%Y-%m-%d %H:%M:%S"), "host": socket.gethostname(),
              "key4hep": os.environ.get("DDML_KEY4HEP_RELEASE") or "unknown",
              "ddml": os.environ.get("DDML_COMMIT") or "unknown",
              "black": None, "steps": [], "plots": [], "context": context}
    log.write("# ddml-bench preflight of %s (%s), %s on %s\n# Key4hep %s, DDML %s" % (
        model["name"], model["config"], record["started"], record["host"], record["key4hep"],
        record["ddml"]))
    current = model
    failed = None
    for number, title in STEPS:
        step = {"number": number, "title": title, "status": NOT_RUN, "details": []}
        record["steps"].append(step)
        if failed:
            continue
        log.write("\n== %d. %s ==" % (number, title))
        try:
            if number == 1:
                current, details = check_description(model["name"])
                changed = [what for what in ("description_sha256", "python")
                           if current.get(what) != planned.get(what)]
                if changed:
                    raise StepFailed("%s changed since this run was planned (its %s), so run "
                                     "again" % (model["config"], " and ".join(
                                         "description" if c == "description_sha256"
                                         else "Python code" for c in changed)))
            elif number == 2:
                details = check_weights(current, log)
            elif number == 3:
                details = check_code(current, preflight, log, record)
            elif number == 4:
                details = check_import(current, preflight, log)
            else:
                details, plots = check_barrel(current, preflight, folder, log)
                record["plots"] = [os.path.basename(p) for p in plots]
            step.update(status=PASSED, details=details)
        except StepSkipped as skipped:
            step.update(status=SKIPPED, details=[str(skipped)])
        except StepFailed as error:
            step.update(status=FAILED, details=[str(error.args[0])])
            if len(error.args) > 1:
                record["plots"] = [os.path.basename(p) for p in error.args[1]]
            failed = number
        except Exception as error:  # anything else a step runs into: its failure too
            import traceback
            log.write(traceback.format_exc())
            step.update(status=FAILED, details=["%s: %s (the traceback is in %s)"
                                                % (type(error).__name__, error, LOG)])
            failed = number
        for line in step["details"]:
            log.write("%s: %s" % (step["status"], line))
        log.say("%s: %d. %s: %s%s" % (model["name"], number, title, step["status"],
                                      ": " + step["details"][0] if step["details"]
                                      and step["status"] != PASSED else ""))
    record.update(status=FAILED if failed else PASSED, failed_step=failed,
                  finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    write_summary(folder, record)
    if not failed:
        partial = marker + ".%d.tmp" % os.getpid()
        with open(partial, "w") as stream:
            json.dump(dict((k, v) for k, v in record.items() if k != "steps"), stream, indent=1)
        os.replace(partial, marker)
    write_overview(os.path.dirname(folder))
    return record


def _log_lines(record, folder):
    """Where the full logs are, in lines."""
    lines = ["  %s  every step's whole output: the check of the description, the md5 sum, "
             "Black's diff, the import's traceback, the check of DDML" % os.path.join(folder, LOG)]
    barrel = os.path.join(folder, BARREL)
    if os.path.isdir(barrel):
        lines.append("  %s  ddsim's output for the barrel shots (beside it: their input, "
                     "shots.slcio and shots.json, and simulation, shots.root)"
                     % os.path.join(barrel, "ddsim.log"))
    context = record.get("context") or {}
    if context.get("slurm_job"):
        lines.append("  slurm job %s: its log is in %s (pipeline_<time>/logs/ of the "
                     "submission)" % (context["slurm_job"], context.get("slurm_log_dir")))
    if context.get("github_run"):
        lines.append("  GitHub: the log of the preflight job of run %s, and its artifact "
                     "low-cost-preflight, which holds this folder" % context["github_run"])
    return lines


def summary_text(record, folder):
    """A model's summary.txt."""
    lines = ["Preflight of %s (%s): %s" % (record["model"], record["config"],
                                            record["status"] if record["status"] == PASSED
                                            else "FAILED at step %d, %s" % (
                                                record["failed_step"],
                                                dict(STEPS)[record["failed_step"]])),
             "  %s to %s on %s; Key4hep %s, DDML %s%s; checked as %s" % (
                 record["started"], record.get("finished", "?"), record["host"],
                 record["key4hep"], record["ddml"],
                 ", Black %s" % record["black"] if record.get("black") else "",
                 record["digest"]), ""]
    for step in record["steps"]:
        lines.append("%d. %-12s %s" % (step["number"], step["title"], step["status"]))
        for detail in step["details"]:
            lines += ["     " + part for part in detail.splitlines()]
        if step["status"] == NOT_RUN:
            lines.append("     (as an earlier step failed)")
    if record["plots"]:
        lines += ["", "Plots of the barrel shots, to look at:"]
        lines += ["  %s" % os.path.join(folder, plot) for plot in record["plots"]]
    lines += ["", "Full logs:"] + _log_lines(record, folder)
    if record["status"] == PASSED:
        lines += ["", "Its simulations can run until what it was checked as changes (its "
                  "description, code or weights, the preflight settings or the software)."]
    else:
        lines += ["", "Its simulations wait until it passes: fix what failed, then run "
                  "./preflight_models.sh %s (or the stage that simulates it) again."
                  % record["model"]]
    return "\n".join(lines) + "\n"


def write_summary(folder, record):
    with open(os.path.join(folder, SUMMARY_JSON), "w") as stream:
        json.dump(record, stream, indent=1)
    with open(os.path.join(folder, SUMMARY), "w") as stream:
        stream.write(summary_text(record, folder))


def write_overview(preflight_dir):
    """preflight_output/summary.txt: every model's last preflight there."""
    records = []
    for name in sorted(os.listdir(preflight_dir)) if os.path.isdir(preflight_dir) else []:
        path = os.path.join(preflight_dir, name, SUMMARY_JSON)
        if os.path.isfile(path):
            try:
                with open(path) as stream:
                    records.append(json.load(stream))
            except (OSError, ValueError):
                continue
    lines = ["Preflight of the models: the checks each passes before it is simulated",
             "(ddml_bench/preflight.py; ./preflight_models.sh runs them)", ""]
    if not records:
        lines.append("No model has been checked yet.")
    for record in records:
        status = record["status"] if record["status"] == PASSED else "FAILED at %d. %s" % (
            record["failed_step"], dict(STEPS).get(record["failed_step"], "?"))
        lines.append("  %-32s %-28s %s" % (record["model"], status, record.get("finished", "")))
        if record["status"] != PASSED:
            step = record["steps"][record["failed_step"] - 1]
            lines.append("      %s" % (step["details"][0].splitlines()[0] if step["details"]
                                       else ""))
    lines += ["", "Each model's steps, its plots and its full logs are in its folder here,",
              "<model>/: summary.txt says what each step found, preflight.log has every",
              "step's whole output (Black's diff, the import's traceback, the check of",
              "DDML), barrel/ddsim.log the simulation of the barrel shots, and",
              "barrel_<particle>.png the plots of their hits. Run as slurm jobs, each",
              "job's own log is in slurm.log_dir (pipeline_<time>/logs/); on GitHub, in",
              "the preflight job's log and its artifact low-cost-preflight."]
    partial = os.path.join(preflight_dir, ".%s.%d.tmp" % (SUMMARY, os.getpid()))
    os.makedirs(preflight_dir, exist_ok=True)
    with open(partial, "w") as stream:
        stream.write("\n".join(lines) + "\n")
    os.replace(partial, os.path.join(preflight_dir, SUMMARY))


def failed_description(name, folder, error, context=None):
    """A model whose description fails step 1 before any plan is made: its
    summary, the rest not run."""
    if os.path.isdir(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)
    log = Log(os.path.join(folder, LOG))
    log.write("== 1. description ==\nFAILED: %s" % error)
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    record = {"model": name, "config": "models/%s.yaml" % name, "status": FAILED,
              "failed_step": 1, "digest": "-", "started": now, "finished": now,
              "host": socket.gethostname(), "key4hep": "-", "ddml": "-", "black": None,
              "plots": [], "context": context or {},
              "steps": [{"number": n, "title": t, "status": FAILED if n == 1 else NOT_RUN,
                         "details": [str(error)] if n == 1 else []} for n, t in STEPS]}
    write_summary(folder, record)
    write_overview(os.path.dirname(folder))
    return record
