#!/usr/bin/env python3
"""Measure the benchmark observables on the reconstructed events.

For each simulator (each model in models/, and geant4) whose reconstruction of
the scenarios the observables use is finished, this gathers the numbers every
plot of the observables in the configuration needs, into

  <paths.data_dir>/observables/<simulator>.npz

(<simulator>_n<N>.npz with --events N, from the reconstructions named
<input>_n<N>_REC.edm4hep.root). A reconstruction is finished when every input
of the scenario has its <input>_REC.edm4hep.root in
rec_output/<scenario>/<simulator>/. A simulator whose .npz file already has
every key the configuration asks for, made with the same settings from the same
reconstructed files, is skipped unless forced.

The observables known (the keys of `observables` in the configuration):

  radial_profile        energy of the reconstructed hits against the distance
                        from the incident particle's line of flight (through its
                        vertex, along its momentum) to the hit's cell centre,
                        one profile per max_radius
  longitudinal_profile  energy of the reconstructed hits per calorimeter layer,
                        from the layer field of each hit's cellID

The incident particle is the first MC particle with generator status 1 (the
particle gun's), from the MC truth collection: MCParticles, as ddsim names it in
EDM4hep and the reconstruction passes it on, or MCParticle, its LCIO name. Each profile is stored per input file of the scenario, as the sum over
events of each bin's energy and of its square, with the number of events; the
mean energy per event in a bin, and its statistical uncertainty, follow from
those. Keys, with "__" between the parts:

  <observable>__<scenario>__<profile>__edges        bin edges
  <observable>__<scenario>__<profile>__sum          [file, bin] sum of energies, GeV
  <observable>__<scenario>__<profile>__sum_sq       [file, bin] sum of squares, GeV^2
  <observable>__<scenario>__<profile>__settings     the settings, as JSON
  <scenario>__files                                 the reconstructed files
  <scenario>__n_events                              [file] events used
  <scenario>__n_no_primary                          [file] events left out
  <scenario>__true_energy                           [file] mean incident energy, GeV
  meta__simulator, meta__label, meta__color, meta__linestyle

This runs in two steps, like the simulation:

  observables.py plan [--events N] [--force] [SIMULATOR ...]
      Read the configuration and the model descriptions; print the plan as JSON.

  observables.py compute [--dry-run] < PLAN
      In the environment from setup.sh (it needs pyLCIO to name the scenarios'
      inputs, and uproot and awkward to read the reconstructed files).
"""

import argparse
import json
import os
import sys

from fasinate_config import REPO_DIR, ConfigError, load_config
from scenario_inputs import INPUT_FILES, scenario_commands

GEANT4 = "geant4"
MODELS_DIR = os.path.join(REPO_DIR, "models")
REC_SUFFIX = "_REC.edm4hep.root"
SEP = "__"
# The colour and line style of the Geant4 reference, and of models without
# plot settings in models/*.yaml: the fixed order of the paper's colours,
# each with a line style of its own.
GEANT4_STYLE = {"label": "Geant4", "color": "#9a9a9a", "linestyle": "-"}
FALLBACK_STYLES = [("#e8710a", "-"), ("#1f6fb4", "--"), ("#2a9a3a", "-."),
                   ("#a8329e", ":"), ("#0095aa", (0, (5, 1, 1, 1, 1, 1)))]
# Events read from a file at a time
CHUNK = 2000


class ObservableError(Exception):
    pass


def say(message):
    print("observables: " + message, file=sys.stderr)


def profile_key(observable, scenario, profile, what):
    return SEP.join((observable, scenario, profile, what))


# ---------------------------------------------------------------------------
# The observables: their profiles (binning) and what each hit contributes


def _numbers(settings, key, observable, kind=float):
    value = settings.get(key)
    values = value if isinstance(value, list) else [value]
    try:
        return [kind(v) for v in values]
    except (TypeError, ValueError):
        raise ObservableError("observables.%s.%s should be a number or a list of numbers"
                              % (observable, key))


# Each returns [(profile name, (first edge, last edge, number of bins))]; the
# binning is plain numbers, so the plan can be checked without numpy.


def radial_profiles(observable, settings):
    radii = _numbers(settings, "max_radius", observable)
    n_bins = _numbers(settings, "n_bins", observable, int)
    if len(n_bins) == 1:
        n_bins = n_bins * len(radii)
    if len(n_bins) != len(radii) or min(radii) <= 0 or min(n_bins) < 1:
        raise ObservableError("observables.%s: n_bins needs a positive number of bins for "
                              "each max_radius" % observable)
    return [("r%g" % radius, (0.0, radius, bins)) for radius, bins in zip(radii, n_bins)]


def longitudinal_profiles(observable, settings):
    max_layer = _numbers(settings, "max_layer", observable, int)[0]
    if max_layer < 0:
        raise ObservableError("observables.%s.max_layer should be at least 0" % observable)
    # One bin per layer number, centred on it
    return [("layers", (-0.5, max_layer + 0.5, max_layer + 1))]


def bin_edges(binning):
    import numpy as np
    first, last, n_bins = binning
    return np.linspace(first, last, n_bins + 1)


# observable: (its profiles, what it needs from each hit)
KINDS = {
    "radial_profile": (radial_profiles, "radius"),
    "longitudinal_profile": (longitudinal_profiles, "layer"),
}


def observable_settings(name, settings):
    """The settings that decide an observable's numbers (not how it is compared)."""
    return {k: v for k, v in sorted(settings.items()) if k != "metric"}


# ---------------------------------------------------------------------------
# The plan, made outside the environment


def _model_entries():
    try:
        import yaml
    except ImportError:
        raise ObservableError("PyYAML is needed to read models/*.yaml")
    entries = []
    names = sorted(f[:-len(".yaml")] for f in os.listdir(MODELS_DIR)
                   if f.endswith(".yaml")) if os.path.isdir(MODELS_DIR) else []
    for i, name in enumerate(names):
        with open(os.path.join(MODELS_DIR, name + ".yaml")) as stream:
            data = yaml.safe_load(stream) or {}
        plot = data.get("plot") or {}
        color, linestyle = FALLBACK_STYLES[i % len(FALLBACK_STYLES)]
        entries.append({
            "name": name,
            "label": str(data.get("name") or name),
            "color": str(plot.get("color") or color),
            "linestyle": plot.get("linestyle") or linestyle,
            "compact": ((data.get("fixed_at_training") or {}).get("detector") or {})
            .get("compact_file"),
        })
    return entries


def make_plan(simulators=(), events=None, force=False):
    config = load_config()
    observables = config.get("observables") or {}
    if not observables:
        raise ObservableError("the configuration has no observables")
    scenarios = []
    for name, settings in observables.items():
        if name not in KINDS:
            raise ObservableError("there's no observable '%s'; the known ones are: %s"
                                  % (name, ", ".join(KINDS)))
        if not isinstance(settings, dict) or not settings.get("scenarios"):
            raise ObservableError("observables.%s needs a list of scenarios" % name)
        KINDS[name][0](name, settings)  # check the binning early
        if not settings.get("hit_collections"):
            raise ObservableError("observables.%s needs hit_collections" % name)
        for scenario in settings["scenarios"]:
            if scenario not in scenarios:
                scenarios.append(scenario)
    if events is not None and events < 1:
        raise ObservableError("the number of events must be at least 1, not %d" % events)

    entries = _model_entries()
    entries.append(dict(GEANT4_STYLE, name=GEANT4, compact=config["geant4"]["compact"]))
    known = [e["name"] for e in entries]
    for name in simulators:
        if name not in known:
            raise ObservableError("there's no simulator '%s'; the simulators are: %s"
                                  % (name, ", ".join(known)))
    if simulators:
        entries = [e for e in entries if e["name"] in simulators]

    inputs = {}
    for name, kind, commands in scenario_commands(config, scenarios):
        if kind not in INPUT_FILES:
            raise ObservableError("scenario %s: there's no way to tell which files a %s "
                                  "input makes" % (name, kind))
        inputs[name] = {"type": kind, "commands": commands}

    data_dir = config["paths"]["data_dir"]
    return {
        "rec_dir": os.path.join(data_dir, "rec_output"),
        "out_dir": os.path.join(data_dir, "observables"),
        "events": events,
        "force": force,
        "observables": observables,
        "inputs": inputs,
        "simulators": entries,
    }


# ---------------------------------------------------------------------------
# Reading the reconstructed files


def parse_encoding(description):
    """{field: (offset, width, signed)} from a DD4hep cellID description such as
    "system:5,module:3,...,cellX:32:-16,cellY:-16"."""
    fields = {}
    offset = 0
    for part in description.split(","):
        bits = [b.strip() for b in part.split(":")]
        try:
            if len(bits) == 2:
                name, width = bits[0], int(bits[1])
            elif len(bits) == 3:
                name, offset, width = bits[0], int(bits[1]), int(bits[2])
            else:
                raise ValueError
        except ValueError:
            raise ObservableError("can't read the cellID encoding %r" % description)
        fields[name] = (offset, abs(width), width < 0)
        offset += abs(width)
    return fields


def decode_field(cell_ids, field):
    import numpy as np
    offset, width, signed = field
    values = ((cell_ids.astype(np.uint64) >> np.uint64(offset))
              & np.uint64((1 << width) - 1)).astype(np.int64)
    if signed:
        values = np.where(values >= (1 << (width - 1)), values - (1 << width), values)
    return values


def encoding_from_file(path, collection):
    """The cellID encoding podio stored for a collection, or None."""
    try:
        from podio.root_io import Reader
    except ImportError:
        return None
    key = collection + "__CellIDEncoding"
    try:
        reader = Reader(path)
        categories = list(getattr(reader, "categories", ["metadata", "events"]))
    except Exception:
        return None
    for category in ["metadata"] + [c for c in categories if c != "metadata"]:
        try:
            frames = reader.get(category)
            for frame in frames:
                try:
                    value = frame.get_parameter(key)
                except Exception:
                    value = None
                if value:
                    return str(value)
                break  # parameters are in the first frame of a category, if anywhere
        except Exception:
            continue
    return None


def encoding_from_geometry(compact, collection):
    """The cellID encoding of the readout of a collection, from the detector."""
    if not compact:
        return None
    if not os.path.isabs(compact):
        k4geo = os.environ.get("k4geo_DIR") or os.environ.get("K4GEO")
        if not k4geo:
            return None
        compact = os.path.join(k4geo, compact)
    try:
        import dd4hep
        detector = dd4hep.Detector.getInstance()
        if not getattr(encoding_from_geometry, "loaded", None) == compact:
            detector.fromXML(compact)
            encoding_from_geometry.loaded = compact
        names = [collection[:-len("Rec")]] if collection.endswith("Rec") else []
        for name in names + [collection]:
            readout = detector.readout(name)
            if readout.isValid():
                return str(readout.idSpec().fieldDescription())
    except Exception as error:  # DD4hep missing, or the geometry can't load
        say("could not read the cellID encoding from %s: %s" % (compact, error))
    return None


# The MC truth collection, by the names it can have: ddsim names it MCParticles
# when it writes EDM4hep (and the reconstruction keeps the input's collections),
# and MCParticle is its LCIO name, which a conversion from LCIO keeps
MC_COLLECTIONS = ("MCParticles", "MCParticle")


class Branches:
    """Columns of a podio events tree, found by name (e.g. "MCParticles.vertex.x")."""

    def __init__(self, path):
        import uproot
        self.path = path
        self.file = uproot.open(path)
        if "events" not in self.file:
            raise ObservableError("%s has no 'events' tree; is it an EDM4hep ROOT file?" % path)
        self.tree = self.file["events"]
        self.keys = {}
        for key in self.tree.keys(recursive=True):
            self.keys.setdefault(key.split("/")[-1], key)
        self.entries = self.tree.num_entries

    def jagged(self, name, start, stop):
        if name not in self.keys:
            collection = name.split(".")[0]
            raise ObservableError("%s has no %s (does it have the collection %s?)"
                                  % (self.path, name, collection))
        return self.tree[self.keys[name]].array(entry_start=start, entry_stop=stop,
                                                library="ak")

    def collection_with(self, candidates, member):
        """The first of the candidate collections that has the member."""
        for name in candidates:
            if "%s.%s" % (name, member) in self.keys:
                return name
        others = sorted({key.split(".")[0] for key in self.keys
                         if key.endswith("." + member)})
        raise ObservableError("%s has no MC truth collection (%s with a %s); %s" % (
            self.path, " or ".join(candidates), member,
            "collections with a %s: %s" % (member, ", ".join(others)) if others
            else "no collection in it has a %s" % member))

    def close(self):
        self.file.close()


def read_primary(branches, start, stop):
    """Vertex, unit direction and energy of each event's incident particle (the
    first MC particle of generator status 1), and which events have one."""
    import awkward as ak
    import numpy as np
    mc = branches.collection_with(MC_COLLECTIONS, "generatorStatus")
    status = branches.jagged(mc + ".generatorStatus", start, stop)
    mask = status == 1

    def first(name):
        values = ak.firsts(branches.jagged(name, start, stop)[mask])
        return ak.to_numpy(ak.fill_none(values, np.nan)).astype(np.float64)

    vertex = np.stack([first(mc + ".vertex." + c) for c in "xyz"], axis=1)
    momentum = np.stack([first(mc + ".momentum." + c) for c in "xyz"], axis=1)
    mass = first(mc + ".mass")
    p = np.linalg.norm(momentum, axis=1)
    has = np.isfinite(p) & (p > 0)
    direction = np.zeros_like(momentum)
    direction[has] = momentum[has] / p[has, None]
    return vertex, direction, np.sqrt(p ** 2 + np.nan_to_num(mass) ** 2), has


def read_hits(branches, collections, start, stop, need_cell_id):
    """The hits of the collections, flattened: event index, energy, position
    and cellID (or None)."""
    import awkward as ak
    import numpy as np
    parts = []
    for collection in collections:
        energy = branches.jagged(collection + ".energy", start, stop)
        counts = ak.to_numpy(ak.num(energy))
        event = np.repeat(np.arange(len(counts)), counts)
        position = np.stack([ak.to_numpy(ak.flatten(
            branches.jagged("%s.position.%s" % (collection, c), start, stop)))
            for c in "xyz"], axis=1).astype(np.float64)
        cell_id = (ak.to_numpy(ak.flatten(branches.jagged(collection + ".cellID", start, stop)))
                   if need_cell_id else None)
        parts.append((collection, event, ak.to_numpy(ak.flatten(energy)).astype(np.float64),
                      position, cell_id))
    return parts


def accumulate(edges, values, energy, event, n_events, keep_events):
    """Per bin, the sum over the kept events of the bin's energy, and of its square."""
    import numpy as np
    n_bins = len(edges) - 1
    index = np.searchsorted(edges, values, side="right") - 1
    valid = (index >= 0) & (index < n_bins) & np.isfinite(values)
    per_event = np.bincount(event[valid] * n_bins + index[valid], weights=energy[valid],
                            minlength=n_events * n_bins).reshape(n_events, n_bins)
    per_event = per_event[keep_events]
    return per_event.sum(axis=0), (per_event ** 2).sum(axis=0)


# ---------------------------------------------------------------------------
# Computing


def expected_files(plan, scenario):
    group = plan["inputs"][scenario]
    suffix = "" if plan["events"] is None else "_n%d" % plan["events"]
    stems = [os.path.splitext(os.path.basename(f))[0]
             for f in INPUT_FILES[group["type"]](group["commands"])]
    return [stem + suffix + REC_SUFFIX for stem in stems]


def wanted(plan):
    """[(observable, scenario, profile, edges, settings JSON)] the configuration asks for."""
    result = []
    for name, settings in plan["observables"].items():
        settings_json = json.dumps(observable_settings(name, settings), sort_keys=True)
        for profile, binning in KINDS[name][0](name, settings):
            for scenario in settings["scenarios"]:
                result.append((name, scenario, profile, bin_edges(binning), settings_json))
    return result


def is_up_to_date(path, profiles, files):
    """The .npz file has every key asked for, with the same binning and settings,
    from the same reconstructed files."""
    import numpy as np
    if not os.path.isfile(path):
        return False, "no %s yet" % os.path.basename(path)
    try:
        with np.load(path, allow_pickle=False) as data:
            keys = set(data.files)
            for name, scenario, profile, edges, settings_json in profiles:
                for what in ("edges", "sum", "sum_sq", "settings"):
                    if profile_key(name, scenario, profile, what) not in keys:
                        return False, "%s has no %s" % (
                            os.path.basename(path), profile_key(name, scenario, profile, what))
                stored = data[profile_key(name, scenario, profile, "edges")]
                if stored.shape != edges.shape or not np.allclose(stored, edges):
                    return False, "the binning of %s/%s changed" % (name, profile)
                if str(data[profile_key(name, scenario, profile, "settings")]) != settings_json:
                    return False, "the settings of %s changed" % name
            for scenario, paths in files.items():
                for what in ("files", "n_events", "n_no_primary", "true_energy"):
                    if SEP.join((scenario, what)) not in keys:
                        return False, "%s has no %s" % (os.path.basename(path),
                                                       SEP.join((scenario, what)))
                if list(data[SEP.join((scenario, "files"))]) != paths:
                    return False, "the reconstructed files of %s changed" % scenario
                newest = max(os.path.getmtime(p) for p in paths)
                if newest > os.path.getmtime(path):
                    return False, "files of %s were reconstructed again since" % scenario
    except (OSError, ValueError) as error:
        return False, "can't read %s: %s" % (path, error)
    return True, ""


def cell_id_fields(settings, observable, path, collection, compact, cache):
    key = (collection, compact)
    if key not in cache:
        description = (settings.get("cell_id_encoding") or encoding_from_file(path, collection)
                       or encoding_from_geometry(compact, collection))
        if not description:
            raise ObservableError(
                "can't tell how to decode the cellIDs of %s: it isn't in %s, and the detector "
                "geometry couldn't give it; set observables.%s.cell_id_encoding"
                % (collection, path, observable))
        cache[key] = parse_encoding(description)
    fields = cache[key]
    field = settings.get("layer_field") or "layer"
    if field not in fields:
        raise ObservableError("the cellID encoding of %s has no field '%s' (it has: %s); "
                              "set observables.%s.layer_field"
                              % (collection, field, ", ".join(fields), observable))
    return fields[field]


def measure_file(path, observables, profiles, simulator, cache):
    """Sums for every profile of one reconstructed file, and its event counts."""
    import numpy as np
    by_collection = {}
    for name, settings in observables.items():
        for collection in settings["hit_collections"]:
            by_collection.setdefault(collection, set()).add(KINDS[name][1])
    sums = {key: [np.zeros(len(edges) - 1), np.zeros(len(edges) - 1)]
            for key, edges in profiles.items()}
    n_events = n_no_primary = 0
    energy_total = 0.0
    branches = Branches(path)
    try:
        for start in range(0, branches.entries, CHUNK):
            stop = min(start + CHUNK, branches.entries)
            n = stop - start
            vertex, direction, true_energy, has = read_primary(branches, start, stop)
            n_events += int(has.sum())
            n_no_primary += int(n - has.sum())
            energy_total += float(true_energy[has].sum())
            hits = read_hits(branches, list(by_collection), start, stop,
                             any("layer" in v for v in by_collection.values()))
            for collection, event, energy, position, cell_id in hits:
                values = {}
                if "radius" in by_collection[collection]:
                    offset = position - vertex[event]
                    along = np.einsum("ij,ij->i", offset, direction[event])
                    values["radius"] = np.linalg.norm(
                        offset - along[:, None] * direction[event], axis=1)
                for name, settings in observables.items():
                    if collection not in settings["hit_collections"]:
                        continue
                    need = KINDS[name][1]
                    if need == "layer" and "layer" not in values:
                        field = cell_id_fields(settings, name, path, collection,
                                               simulator.get("compact"), cache)
                        values["layer"] = decode_field(cell_id, field).astype(np.float64)
                        # A wrong cellID encoding gives layers outside the
                        # binning, which would otherwise drop out silently
                        max_layer = _numbers(settings, "max_layer", name, int)[0]
                        outside = (values["layer"] < 0) | (values["layer"] > max_layer)
                        if outside.mean() > 0.01 and (path, collection) not in cache:
                            cache[(path, collection)] = True
                            say("    warning: %.0f%% of the hits of %s have a layer outside "
                                "0 to %d; check observables.%s.cell_id_encoding and "
                                "max_layer" % (100 * outside.mean(), collection, max_layer,
                                               name))
                    for (obs, profile), edges in profiles.items():
                        if obs != name:
                            continue
                        total, total_sq = accumulate(edges, values[need], energy, event, n, has)
                        sums[(obs, profile)][0] += total
                        sums[(obs, profile)][1] += total_sq
    finally:
        branches.close()
    mean_energy = energy_total / n_events if n_events else float("nan")
    return sums, n_events, n_no_primary, mean_energy


def compute_simulator(plan, simulator, files, profiles, out_path):
    import numpy as np
    arrays = {
        "meta__simulator": np.array(simulator["name"]),
        "meta__label": np.array(simulator["label"]),
        "meta__color": np.array(simulator["color"]),
        "meta__linestyle": np.array(json.dumps(simulator["linestyle"])),
    }
    cache = {}
    for scenario, paths in files.items():
        observables = {name: settings for name, settings in plan["observables"].items()
                       if scenario in settings["scenarios"]}
        scenario_profiles = {(name, profile): edges
                             for name, scen, profile, edges, _ in profiles if scen == scenario}
        per_file = {key: ([], []) for key in scenario_profiles}
        counts, no_primary, energies = [], [], []
        for i, path in enumerate(paths, 1):
            say("  %s [%d/%d] %s" % (scenario, i, len(paths), os.path.basename(path)))
            sums, n, missing, energy = measure_file(path, observables, scenario_profiles,
                                                    simulator, cache)
            if missing:
                say("    %d event(s) without an incident particle (generator status 1) "
                    "were left out" % missing)
            for key in scenario_profiles:
                per_file[key][0].append(sums[key][0])
                per_file[key][1].append(sums[key][1])
            counts.append(n)
            no_primary.append(missing)
            energies.append(energy)
        arrays[SEP.join((scenario, "files"))] = np.array(paths)
        arrays[SEP.join((scenario, "n_events"))] = np.array(counts, dtype=np.int64)
        arrays[SEP.join((scenario, "n_no_primary"))] = np.array(no_primary, dtype=np.int64)
        arrays[SEP.join((scenario, "true_energy"))] = np.array(energies)
        for name, scen, profile, edges, settings_json in profiles:
            if scen != scenario:
                continue
            arrays[profile_key(name, scenario, profile, "edges")] = edges
            arrays[profile_key(name, scenario, profile, "sum")] = np.array(
                per_file[(name, profile)][0])
            arrays[profile_key(name, scenario, profile, "sum_sq")] = np.array(
                per_file[(name, profile)][1])
            arrays[profile_key(name, scenario, profile, "settings")] = np.array(settings_json)

    # Written to a hidden file first, so the .npz file is always complete
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    partial = os.path.join(os.path.dirname(out_path),
                           ".%s.%d.npz" % (os.path.basename(out_path), os.getpid()))
    try:
        with open(partial, "wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(partial, out_path)
    finally:
        if os.path.exists(partial):
            os.remove(partial)


def compute(plan, dry_run=False):
    profiles = wanted(plan)
    scenarios = sorted({p[1] for p in profiles})
    expected = {s: expected_files(plan, s) for s in scenarios}
    suffix = "" if plan["events"] is None else "_n%d" % plan["events"]
    done, skipped, unfinished, failed = [], [], [], []
    for simulator in plan["simulators"]:
        name = simulator["name"]
        files, missing = {}, []
        for scenario in scenarios:
            folder = os.path.join(plan["rec_dir"], scenario, name)
            paths = [os.path.join(folder, f) for f in expected[scenario]]
            absent = [p for p in paths if not os.path.isfile(p)]
            if absent:
                missing.append("%s: %d of %d reconstructions" % (scenario, len(absent),
                                                                 len(paths)))
            files[scenario] = paths
        if missing:
            say("%s: skipped, its reconstruction isn't finished (missing %s)"
                % (name, "; ".join(missing)))
            unfinished.append(name)
            continue
        out_path = os.path.join(plan["out_dir"], name + suffix + ".npz")
        current, why = is_up_to_date(out_path, profiles, files)
        if current and not plan["force"]:
            say("%s: %s is up to date, so skipped" % (name, out_path))
            skipped.append(name)
            continue
        if dry_run:
            say("%s: would measure %s (%s)" % (name, out_path,
                                               "--force" if current else why))
            continue
        say("%s: measuring, as %s" % (name, "--force was given" if current else why))
        try:
            compute_simulator(plan, simulator, files, profiles, out_path)
        except (ObservableError, OSError) as error:
            say("%s: FAILED: %s" % (name, error))
            failed.append(name)
            continue
        except Exception as error:  # e.g. a file uproot can't read; go on with the others
            say("%s: FAILED reading its reconstructed files: %s: %s"
                % (name, type(error).__name__, error))
            failed.append(name)
            continue
        say("%s: wrote %s" % (name, out_path))
        done.append(name)
    if dry_run:
        say("dry run, so nothing was measured")
    else:
        say("measured %d, up to date %d, not reconstructed yet %d, failed %d"
            % (len(done), len(skipped), len(unfinished), len(failed)))
    return 1 if failed else 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Measure the benchmark observables on the reconstructed events.",
        epilog="See the module docstring for the keys of the .npz files.")
    steps = parser.add_subparsers(dest="step")
    plan_parser = steps.add_parser("plan", help="print the plan as JSON")
    plan_parser.add_argument("simulators", nargs="*", metavar="SIMULATOR",
                             help="models in models/, or geant4 (default: all of them)")
    plan_parser.add_argument("--events", type=int, metavar="N",
                             help="use the reconstructions of the first N events "
                                  "(<input>_n<N>_REC.edm4hep.root)")
    plan_parser.add_argument("--force", action="store_true",
                             help="measure again even where the .npz file is up to date")
    compute_parser = steps.add_parser("compute", help="read the plan on stdin and measure")
    compute_parser.add_argument("--dry-run", action="store_true",
                                help="show what would be measured")
    args = parser.parse_args(argv)
    try:
        if args.step == "plan":
            print(json.dumps(make_plan(args.simulators, args.events, args.force)))
            return 0
        if args.step == "compute":
            return compute(json.load(sys.stdin), args.dry_run)
        parser.error("choose a step: plan or compute")
    except (ObservableError, ConfigError) as error:
        say(str(error))
        return 1


if __name__ == "__main__":
    sys.exit(main())
