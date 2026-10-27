"""DDML model configurations written out in a model's description: the entries
of ddml.model_configs in models/<name>.yaml, for a model that DDML has no
preset for, or one that is a preset with something changed (other weights, a
lower trigger, ...).

An entry is the name of one of DDML's presets (python/ddml/configs.py in the
DDML build), or a mapping: the preset it starts from (preset), if any, and the
fields of DDML's ModelConfig (python/ddml/model.py) that it sets:

  plugin                the DDML plugin, <factory>/<instance name>, e.g.
                        RegularGridGANPolyhedraBarrelONNXModel/BarrelModelONNX
                        (the factories are declared in src/MLModelActions.cc)
  geometry              where DDML places the showers: a geometry of
                        python/ddml/configs.py by name, e.g. ILD_BARREL, or
                        {ecal: {detector, region, symmetry}, hcal: {...}},
                        the fields of DDML's DetectorGeometry
  plugin_properties     the plugin's properties, e.g. ModelPath, OptimizeFlag,
                        IntraOpNumThreads. ModelPath is relative to DDML's
                        scripts/ folder, where ddsim runs, as in DDML's presets
  applicable_particles  the particles DDML gives the model (Geant4's names)
  triggers              {particle: energy in GeV}, above which DDML gives the
                        particle to the model (DDML's presets give energies in
                        Geant4's units; these are GeV)
  correct_angles        DDML's CorrectForAngles: place each layer of the shower
                        where the particle's line of flight crosses it, for a
                        model of showers at normal incidence
  is_hadron             a model of hadron showers (DDML places them in the
                        geometry's hcal too)

Without a preset, plugin, geometry, applicable_particles and triggers are
needed. With one, the fields given replace the preset's, except
plugin_properties, which are set over the preset's one by one.

check() checks an entry where the plan is made, which needs no DDML. build()
makes DDML's ModelConfig of each entry with DDML's own Python package, in the
environment from setup.sh: in ddsim, through scripts/ddsim_steer_models.py,
which the models stage runs for a model with model_configs, and in the stage's
check of the environment, so that both make the same. On ddsim's command line
each entry is an option --ml-config <JSON> (option()), which the steering file
reads (from_command_line()). This module needs only the standard library, as
ddsim imports it.
"""

import json
import re

# ddsim's option for an entry, as DDML's own steering file takes --ml-model
OPTION = "--ml-config"
# DDML's ModelConfig's fields, which an entry may set
FIELDS = ("plugin", "geometry", "plugin_properties", "applicable_particles", "triggers",
          "correct_angles", "is_hadron")
# What an entry without a preset needs (DDML's ModelConfig has no default for
# these; plugin_properties defaults to none here)
REQUIRED = ("plugin", "geometry", "applicable_particles", "triggers")
# The parts of a geometry, and of each part (DDML's PluginGeometry and
# DetectorGeometry)
GEOMETRY_PARTS = ("ecal", "hcal")
DETECTOR_FIELDS = ("detector", "region", "symmetry")

_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_PLUGIN = re.compile(r"^[A-Za-z_][A-Za-z0-9_:]*(/[A-Za-z0-9_.:-]+)?$")


class ModelConfigError(Exception):
    pass


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _scalar(value):
    return isinstance(value, (str, int, float, bool))


def _geometry_problem(value):
    """What is wrong with a geometry, or None."""
    if isinstance(value, str):
        if not _IDENTIFIER.match(value):
            return "should name a geometry of DDML's python/ddml/configs.py, e.g. ILD_BARREL"
        return None
    if not isinstance(value, dict) or "ecal" not in value:
        return ("should name a geometry of DDML's python/ddml/configs.py, e.g. ILD_BARREL, "
                "or be {ecal: {detector, region, symmetry}}, with hcal too if needed")
    for part, detector in value.items():
        if part not in GEOMETRY_PARTS:
            return "has %s, but its parts are %s" % (part, " and ".join(GEOMETRY_PARTS))
        if part == "hcal" and detector is None:
            continue
        if not isinstance(detector, dict):
            return "%s should be {detector, region, symmetry}" % part
        for key in detector:
            if key not in DETECTOR_FIELDS:
                return "%s has %s, but its fields are %s" % (part, key, ", ".join(DETECTOR_FIELDS))
        for key in ("detector", "region"):
            if not _text(detector.get(key)):
                return "%s.%s should be the name of a %s" % (
                    part, key, "subdetector" if key == "detector" else "detector region")
        symmetry = detector.get("symmetry")
        if symmetry is not None and (not isinstance(symmetry, int) or isinstance(symmetry, bool)
                                     or symmetry < 1):
            return "%s.symmetry should be a whole number, at least 1, or null" % part
    return None


def _field_problem(key, value):
    """What is wrong with the value of one of an entry's fields, or None."""
    if key == "plugin":
        if not _text(value) or not _PLUGIN.match(value):
            return ("should be a DDML plugin, <factory>/<instance name>, e.g. "
                    "CaloCloudsTwoAngleModelPolyhedraBarrelTorchModel/BarrelModelTorch")
    elif key == "geometry":
        return _geometry_problem(value)
    elif key == "plugin_properties":
        if not isinstance(value, dict):
            return "should be the plugin's properties, e.g. {ModelPath: ../models/model.pt}"
        for name, setting in value.items():
            if not _text(name):
                return "has a property without a name"
            ok = _scalar(setting) or (isinstance(setting, list) and all(_scalar(v) for v in setting))
            if not ok:
                return "%s should be a number, a string, true or false, or a list of them" % name
    elif key == "applicable_particles":
        if not isinstance(value, list) or not value or not all(_text(v) for v in value):
            return "should be a list of particles, by their Geant4 names, e.g. [gamma]"
        if len(set(value)) != len(value):
            return "names a particle twice"
    elif key == "triggers":
        if not isinstance(value, dict) or not value:
            return "should be {particle: energy in GeV}, e.g. {gamma: 10}"
        for particle, energy in value.items():
            if not _text(particle) or not _number(energy) or energy < 0:
                return "should be {particle: energy in GeV, at least 0}, e.g. {gamma: 10}"
    elif key in ("correct_angles", "is_hadron"):
        if not isinstance(value, bool):
            return "should be true or false"
    return None


def check(entry, where):
    """An entry of ddml.model_configs, checked, as a mapping: a preset's name
    becomes {preset: name}. where says where it is, for the messages."""
    if isinstance(entry, str):
        entry = {"preset": entry}
    if not isinstance(entry, dict) or not entry:
        raise ModelConfigError("%s should be the name of a DDML preset, or a DDML model "
                               "configuration: a preset to start from and the fields of DDML's "
                               "ModelConfig to set (%s)" % (where, ", ".join(FIELDS)))
    for key in entry:
        if key != "preset" and key not in FIELDS:
            raise ModelConfigError("%s has %s, which DDML's ModelConfig doesn't have; it can set "
                                   "preset, %s" % (where, key, ", ".join(FIELDS)))
    preset = entry.get("preset")
    if "preset" in entry and (not isinstance(preset, str) or not _IDENTIFIER.match(preset)):
        raise ModelConfigError("%s: preset should name one of DDML's presets (in "
                               "python/ddml/configs.py), e.g. CC3_BARREL" % where)
    if preset is None:
        missing = [key for key in REQUIRED if key not in entry]
        if missing:
            raise ModelConfigError("%s has no preset, so it needs %s too" % (
                where, ", ".join(missing)))
    for key in FIELDS:
        if key in entry:
            problem = _field_problem(key, entry[key])
            if problem:
                raise ModelConfigError("%s: %s %s" % (where, key, problem))
    return dict(entry)


def option(entry):
    """An entry as ddsim's --ml-config takes it: JSON, the same every time."""
    return json.dumps(entry, sort_keys=True, separators=(",", ":"))


def canonical(entries):
    """A list of entries as one string, the same every time, for comparing
    what models ran with (ddml_bench/manifest.py, ddml_bench/fetch.py)."""
    return json.dumps(list(entries), sort_keys=True, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Making DDML's ModelConfigs, with DDML's Python package (in the environment)


def _build_geometry(value, presets, geometry_module):
    if isinstance(value, str):
        found = getattr(presets, value, None)
        if value.startswith("_") or not isinstance(found, geometry_module.PluginGeometry):
            raise ModelConfigError("DDML has no geometry %s in %s" % (value, presets.__file__))
        return found

    def part(fields):
        if fields is None:
            return None
        return geometry_module.DetectorGeometry(
            detector=fields["detector"], region=fields["region"], symmetry=fields.get("symmetry"))

    return geometry_module.PluginGeometry(ecal=part(value["ecal"]), hcal=part(value.get("hcal")))


def build(entries):
    """DDML's ModelConfig of each entry (as check() returns them), made with
    DDML's Python package, ddml, as DDML's own presets are; the triggers in
    Geant4's units (g4units, from DD4hep)."""
    import dataclasses
    import importlib
    from g4units import GeV
    ddml = importlib.import_module("ddml")
    presets = importlib.import_module("ddml.configs")
    geometry_module = importlib.import_module("ddml.geometry")
    model_config = getattr(ddml, "ModelConfig", None)
    if model_config is None:
        raise ModelConfigError("the ddml Python package in %s has no ModelConfig; is it DDML's "
                               "python-inference branch?" % ddml.__file__)
    configs = []
    for number, entry in enumerate(entries, 1):
        fields = {}
        for key, value in entry.items():
            if key == "preset":
                continue
            if key == "geometry":
                value = _build_geometry(value, presets, geometry_module)
            elif key == "applicable_particles":
                value = frozenset(value)
            elif key == "triggers":
                value = dict((particle, float(energy) * GeV) for particle, energy in value.items())
            elif key == "plugin_properties":
                value = dict(value)
            fields[key] = value
        preset = entry.get("preset")
        try:
            if preset is None:
                fields.setdefault("plugin_properties", {})
                configs.append(model_config(**fields))
                continue
            base = getattr(presets, preset, None)
            if preset.startswith("_") or not isinstance(base, model_config):
                raise ModelConfigError("DDML has no preset %s (see %s)" % (preset, presets.__file__))
            if "plugin_properties" in fields:
                merged = dict(base.plugin_properties)
                merged.update(fields["plugin_properties"])
                fields["plugin_properties"] = merged
            configs.append(dataclasses.replace(base, **fields))
        except (TypeError, ValueError) as error:
            raise ModelConfigError("model configuration %d (%s): DDML's ModelConfig doesn't take "
                                   "it: %s" % (number, option(entry), error))
    return configs


def describe(config):
    """One line about a built ModelConfig, for logs."""
    from g4units import GeV
    ecal = config.geometry.ecal
    where = "%s in %s" % (ecal.detector, ecal.region)
    if ecal.symmetry is not None:
        where += " (symmetry %d)" % ecal.symmetry
    triggers = ", ".join("%s above %g GeV" % (particle, config.triggers[particle] / GeV)
                         for particle in sorted(config.triggers))
    properties = ", ".join("%s=%s" % (name, config.plugin_properties[name])
                           for name in sorted(config.plugin_properties))
    return "%s, %s; %s; CorrectForAngles %s; %s%s" % (
        config.plugin, where, triggers or "no triggers", config.correct_angles,
        properties or "no properties", "; hadron showers" if config.is_hadron else "")


def from_command_line(argv):
    """(entries, record_calo_entry) of the options DDML's steering file takes
    for its fast simulation, taken out of argv (sys.argv, in place), as DDML's
    get_fastsim_configuration() takes its own: --ml-config JSON, each an entry,
    and, as DDML's, --ml-model PRESET (an entry of just that preset) and
    --record-calo-entry. The entries are in the order given."""
    import argparse

    def entry(text):
        try:
            value = json.loads(text)
        except ValueError as error:
            raise argparse.ArgumentTypeError("not JSON: %s" % error)
        try:
            return check(value, "--ml-config")
        except ModelConfigError as error:
            raise argparse.ArgumentTypeError(str(error))

    # No abbreviations, so that none of ddsim's own options is taken for one
    parser = argparse.ArgumentParser(prog="ddsim_steer_models.py", add_help=False,
                                     allow_abbrev=False)
    parser.add_argument(OPTION, dest="entries", action="append", type=entry, default=[],
                        help="a DDML model configuration, as JSON (repeatable)")
    parser.add_argument("--ml-model", dest="entries", action="append", default=[],
                        type=lambda name: check(name, "--ml-model"),
                        help="one of DDML's presets (repeatable)")
    parser.add_argument("--record-calo-entry", action="store_true", default=False,
                        help="record the particles as they enter the calorimeter")
    args, rest = parser.parse_known_args(argv[1:])
    argv[:] = argv[:1] + rest
    return args.entries, args.record_calo_entry
