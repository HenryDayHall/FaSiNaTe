"""The simulators: Geant4, the reference, and the models described in models/.

Each model has a description, models/<name>.yaml (see models/caloclouds3.yaml
for a commented one); its file name names the model in commands and folders.
This is the one place that reads them.

How DDML runs a model is either DDML's presets (ddml.presets: names from
python/ddml/configs.py in DDML, which ddsim takes with --ml-model), or DDML
model configurations written out (ddml.model_configs, each a preset to start
from and the fields of DDML's ModelConfig to set; ddml_bench/ddml_configs.py),
for a model DDML has no preset for. A configuration can have one of DDML's
embedded-Python plugins run a module of models/ (its plugin_properties
PythonModule): the model's own inference code, e.g.
models/caloclouds3_rescaled.py, which ddsim imports from there, ahead of
DDML's modules (scripts/ddsim_steer_models.py). A model can be another's
closure test (closure): the same model, e.g. the same weights through another
of DDML's inference backends, which the plots then compare with it directly.

What a model is, beyond its presets or configurations and weights, is
recorded by two sha256 sums, which each simulation's log has (as ddsim
starts), and so the manifests of published measurements, which fetching them
checks: its description's (description_sha256(): as read, so not its
comments or layout, and without what only the plots use) and its own Python
code's (python_code(): each module of models/ its configurations run).
"""

import hashlib
import json
import os
import re

from .config import REPO_DIR
from .ddml_configs import ModelConfigError, check as check_model_config
from .layout import GEANT4, LayoutError, check_name

MODELS_DIR = os.path.join(REPO_DIR, "models")

# How the plots draw Geant4, and the models without plot settings in their
# description: the fixed order of the paper's colours, each with a line style
# of its own.
GEANT4_STYLE = {"label": "Geant4", "color": "#9a9a9a", "linestyle": "-"}
FALLBACK_STYLES = [("#e8710a", "-"), ("#1f6fb4", "--"), ("#2a9a3a", "-."),
                   ("#a8329e", ":"), ("#0095aa", (0, (5, 1, 1, 1, 1, 1)))]
# What of a description only the plots use (its label, its style, the model it
# is the closure test of), which its sha256 leaves out, so that drawing a model
# otherwise doesn't make it another model
PLOTS_ONLY = ("name", "plot", "closure")

_MD5 = re.compile(r"^[0-9a-f]{32}$")


class ModelError(Exception):
    pass


def model_names():
    """The models described in models/, by file name."""
    if not os.path.isdir(MODELS_DIR):
        return []
    return sorted(name[:-len(".yaml")] for name in os.listdir(MODELS_DIR)
                  if name.endswith(".yaml") and not name.startswith("."))


def load_model(name):
    """What the pipeline needs from models/<name>.yaml, after checking it:
    name, config (the description's path), label, compact_file, particle (the
    one it was trained on), particles (those DDML gives it), presets or
    model_configs (the other empty), weights, weights_md5, closure (None, or
    {of: the model it is a closure test of, identical: whether the events
    should be identical where both have the same seeds}), color and
    linestyle, and the sha256 sums of what it is: description_sha256 (of the
    description) and python ({module: {path, sha256}} of its own Python code,
    python_code())."""
    names = model_names()
    path = os.path.join(MODELS_DIR, name + ".yaml")
    source = os.path.relpath(path, REPO_DIR)
    if not os.path.isfile(path):
        raise ModelError("there is no model '%s'; models/ has: %s"
                         % (name, ", ".join(names) or "nothing"))
    try:
        check_name(name, "model")
    except LayoutError as error:
        raise ModelError("%s: %s" % (source, error))
    if name == GEANT4:
        raise ModelError("%s: '%s' names the Geant4 reference, so it can't be a model name"
                         % (source, GEANT4))
    try:
        import yaml
    except ImportError:
        raise ModelError("PyYAML is needed to read %s; install it with "
                         "'python3 -m pip install --user pyyaml'" % source)
    try:
        with open(path) as stream:
            data = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as error:
        raise ModelError("cannot read %s: %s" % (source, error))
    if not isinstance(data, dict):
        raise ModelError("%s: expected the model's description, a set of settings" % source)

    def get(keys, is_valid, what, required=True):
        node = data
        for key in keys.split("."):
            if not isinstance(node, dict) or key not in node:
                if required:
                    raise ModelError("%s: %s is missing; it should be %s" % (source, keys, what))
                return None
            node = node[key]
        if not is_valid(node):
            raise ModelError("%s: %s should be %s" % (source, keys, what))
        return node

    def text(value):
        return isinstance(value, str) and bool(value.strip())

    def section(value):
        return isinstance(value, dict) and bool(value)

    def names_list(value):
        return isinstance(value, list) and bool(value) and all(text(v) for v in value)

    get("representation", text, "how the model represents a shower, e.g. point_cloud")
    get("fixed_at_training", section, "the factors fixed when the model was trained")
    get("context", section, "the factors given to the model for each shower")
    get("ddml", section, "how DDML runs the model: its presets or model_configs, and weights")
    particle = get("fixed_at_training.particle.name", text,
                   "the Geant4 name of the particle the model was trained on")
    particles = get("ddml.particles", names_list,
                    "the particles DDML gives the model, by their Geant4 names (by default "
                    "the one it was trained on)", required=False)
    model = {
        "name": name,
        "config": source,
        "label": str(data.get("name") or name),
        "compact_file": get("fixed_at_training.detector.compact_file", text,
                            "the detector's compact file, relative to k4geo"),
        "particle": particle,
        "particles": sorted(set(particles)) if particles else [particle],
        "presets": get("ddml.presets", names_list,
                       "a list of DDML presets (from python/ddml/configs.py in DDML)",
                       required=False) or [],
        "model_configs": [],
        "weights": get("ddml.weights", text,
                       "the path of the weights in the DDML clone", required=False),
        "weights_md5": get("ddml.weights_md5",
                           lambda v: isinstance(v, str) and bool(_MD5.match(v)),
                           "an md5 sum: 32 hex digits, in quotes", required=False),
        "closure": None,
    }
    if model["weights_md5"] and not model["weights"]:
        raise ModelError("%s: ddml.weights_md5 is set, but not ddml.weights" % source)

    # How DDML runs it: its presets, or DDML model configurations written out
    configs = get("ddml.model_configs", lambda v: isinstance(v, list) and bool(v),
                  "a list of DDML model configurations: each a DDML preset's name, or a "
                  "preset to start from and the fields of DDML's ModelConfig to set (see "
                  "ddml_bench/ddml_configs.py)", required=False)
    if bool(model["presets"]) == bool(configs):
        raise ModelError("%s: ddml should have %s: presets, DDML's own (passed to ddsim with "
                         "--ml-model), or model_configs, DDML model configurations written out "
                         "(see ddml_bench/ddml_configs.py)"
                         % (source, "one of these, not both" if configs else "one of these"))
    for number, entry in enumerate(configs or [], 1):
        try:
            model["model_configs"].append(check_model_config(
                entry, "%s: ddml.model_configs, entry %d" % (source, number)))
        except ModelConfigError as error:
            raise ModelError(str(error))
    # What it is: its description, and its own Python code
    model["description_sha256"] = description_sha256(data)
    model["python"] = python_code(model["model_configs"])

    # The model it is the closure test of
    closure = data.get("closure")
    if closure is not None:
        if not isinstance(closure, dict) or set(closure) - {"of", "identical"} \
                or not isinstance(closure.get("of"), str):
            raise ModelError("%s: closure should be {of: the model this is the same as, e.g. "
                             "through another DDML backend, identical: true or false}" % source)
        twin = closure["of"]
        if twin == name or twin == GEANT4 or twin not in names:
            raise ModelError("%s: closure.of should be another model in models/ (%s), not %r"
                             % (source, ", ".join(n for n in names if n != name) or "none", twin))
        identical = closure.get("identical", False)
        if not isinstance(identical, bool):
            raise ModelError("%s: closure.identical should be true or false: whether the "
                             "events of the two should be identical where they are simulated "
                             "with the same seeds" % source)
        model["closure"] = {"of": twin, "identical": identical}

    # How the plots draw it; models without a style get the next fallback one
    plot = data.get("plot") or {}
    if not isinstance(plot, dict):
        raise ModelError("%s: plot should be a colour and a line style" % source)
    color, linestyle = FALLBACK_STYLES[names.index(name) % len(FALLBACK_STYLES)]
    model["color"] = str(plot.get("color") or color)
    model["linestyle"] = plot.get("linestyle") or linestyle
    return model


def load_models(names=None):
    """The descriptions of the models named, or of every model in models/."""
    found = []
    for name in model_names() if names is None else names:
        name = os.path.basename(name)
        if name.endswith(".yaml"):
            name = name[:-len(".yaml")]
        if name not in [model["name"] for model in found]:
            found.append(load_model(name))
    return found


def reference(config):
    """Geant4, as the models are described: its detector and how it is drawn."""
    return dict(GEANT4_STYLE, name=GEANT4, compact_file=config["geant4"]["compact"])


# ---------------------------------------------------------------------------
# What a model is: the sha256 sums of its description and its own Python code

_MODULE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")


def _same_every_time(value):
    """A description's settings with every key a string, for JSON."""
    if isinstance(value, dict):
        return dict((str(key), _same_every_time(item)) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return [_same_every_time(item) for item in value]
    return value


def description_sha256(data):
    """The sha256 of a model's description as read (data, from
    models/<name>.yaml), without what only the plots use (PLOTS_ONLY): so its
    comments, layout, label and plot style don't change it, and anything else
    does, e.g. a setting the model's own Python code reads, as
    models/caloclouds3_rescaled.py reads energy_rescale."""
    kept = dict((key, value) for key, value in data.items() if key not in PLOTS_ONLY)
    text = json.dumps(_same_every_time(kept), sort_keys=True, separators=(",", ":"),
                      default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def description_sha256_now(model):
    """description_sha256() of a model's description as it is now (as a
    simulation starts), or None if it can't be read here, e.g. without
    PyYAML."""
    try:
        import yaml
        with open(os.path.join(REPO_DIR, model["config"])) as stream:
            data = yaml.safe_load(stream)
    except Exception:  # no PyYAML, the file gone, or not YAML any more
        return None
    return description_sha256(data) if isinstance(data, dict) else None


def python_module_file(module):
    """The file of a module of models/: the models' own inference code, which
    a DDML model configuration's PythonModule can name, and ddsim imports from
    models/, ahead of DDML's modules (scripts/ddsim_steer_models.py). None if
    models/ has no module of that name."""
    if not isinstance(module, str) or not _MODULE.match(module):
        return None
    base = os.path.join(MODELS_DIR, *module.split("."))
    for path in (base + ".py", os.path.join(base, "__init__.py")):
        if os.path.isfile(path):
            return path
    return None


def python_code(model_configs):
    """{module: {"path", "sha256"}} of the models' own Python code that these
    DDML model configurations (ddml.model_configs, as checked) run: each
    module of models/ that one names as its PythonModule (plugin_properties),
    with its path in the repository and its sha256 as it is now."""
    code = {}
    for entry in model_configs:
        properties = entry.get("plugin_properties") if isinstance(entry, dict) else None
        module = (properties or {}).get("PythonModule")
        path = python_module_file(module)
        if path:
            digest = hashlib.sha256()
            with open(path, "rb") as stream:
                digest.update(stream.read())
            code[module] = {"path": os.path.relpath(path, REPO_DIR).replace(os.sep, "/"),
                            "sha256": digest.hexdigest()}
    return code


def python_summary(code):
    """A model's own Python code (python_code()) in one line, as the log of a
    simulation records it: <module>:<sha256> of each module, by name, or
    none."""
    return " ".join("%s:%s" % (module, code[module]["sha256"]) for module in sorted(code)) \
        or "none"
