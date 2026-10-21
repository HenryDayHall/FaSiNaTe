"""Measure the benchmark observables on reconstructed events.

The observables stage (ddml_bench/stages/observables.py) gathers, for each
simulator (each model in models/, and geant4) whose reconstructions of the
scenarios the observables use are finished, the numbers every plot of the
observables in the configuration needs, into the .npz file the layout names:

  <paths.data_dir>/observables/<simulator><sample>.npz

The observables known (the keys of `observables` in the configuration):

  radial_profile        energy of the reconstructed hits against the distance
                        from the incident particle's line of flight (through its
                        vertex, along its momentum) to the hit's cell centre,
                        one profile per max_radius; stored as the energy in
                        each ring around the line of flight, which the figures
                        divide by the ring's area (ddml_bench/plot.py)
  longitudinal_profile  energy of the reconstructed hits per calorimeter layer,
                        from the layer field of each hit's cellID
  energy                the reconstructed energy of each event, the sum of the
                        energies of its reconstructed particles (PandoraPFA's,
                        in particle_collections; only those of the PDG codes in
                        pdg, if given), with the incident particle's energy;
                        the figures take the energy linearity and resolution
                        from the central 90% of the events at each incident
                        energy (ddml_bench/plot.py)

The incident particle is the first MC particle with generator status 1 (the
particle gun's), from the MC truth collection: MCParticles, as ddsim names it in
EDM4hep and the reconstruction passes it on, or MCParticle, its LCIO name. Each
profile is stored per input file of the scenario, as the sum over events of each
bin's energy and of its square, with the number of events; the mean energy per
event in a bin, and its statistical uncertainty, follow from those. The energy
is stored per event (the central 90% needs every event), in the order of the
files, each file's events in turn. Keys, with "__" between the parts:

  <observable>__<scenario>__<profile>__edges        bin edges
  <observable>__<scenario>__<profile>__sum          [file, bin] sum of energies, GeV
  <observable>__<scenario>__<profile>__sum_sq       [file, bin] sum of squares, GeV^2
  <observable>__<scenario>__<profile>__settings     the settings, as JSON
  energy__<scenario>__reconstructed                 [event] reconstructed energy, GeV
  energy__<scenario>__true                          [event] incident energy, GeV
  energy__<scenario>__settings                      the settings, as JSON
  <scenario>__files                                 the reconstructed files, relative
                                                    to paths.data_dir
  <scenario>__n_events                              [file] events used
  <scenario>__n_no_primary                          [file] events left out
  <scenario>__true_energy                           [file] mean incident energy, GeV
  meta__simulator, meta__label, meta__color, meta__linestyle

A .npz file is up to date when it has every key the configuration asks for,
made with the same settings from the same reconstructed files, none of them
reconstructed again since. (Files measured before the paths were recorded
relative to paths.data_dir, with full paths, still count.)
"""

import json
import os
import sys

from .layout import SEP

# Events read from a file at a time
CHUNK = 2000


class ObservableError(Exception):
    pass


def say(message):
    print("observables: " + message, file=sys.stderr)


def profile_key(observable, scenario, profile, what):
    return SEP.join((observable, scenario, profile, what))


def event_key(observable, scenario, what):
    """The key of an observable stored per event."""
    return SEP.join((observable, scenario, what))


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

# The observables of each event as a whole, stored per event rather than in
# bins, as the figures need every event: observable: what is stored of each
PER_EVENT = {
    "energy": ("reconstructed", "true"),
}


def check_particles(observable, settings):
    """The settings of an observable of the reconstructed particles."""
    collections = settings.get("particle_collections")
    if not isinstance(collections, list) or not collections \
            or not all(isinstance(c, str) and c for c in collections):
        raise ObservableError("observables.%s needs particle_collections, a list of the "
                              "collections of reconstructed particles, e.g. [PandoraPFOs]"
                              % observable)
    codes = settings.get("pdg")
    if codes is not None and (not isinstance(codes, list) or not all(
            isinstance(c, int) and not isinstance(c, bool) for c in codes)):
        raise ObservableError("observables.%s.pdg should be a list of PDG codes, e.g. [22], "
                              "or null for every reconstructed particle" % observable)


def observable_settings(name, settings):
    """The settings that decide an observable's numbers (not how it is compared)."""
    return {k: v for k, v in sorted(settings.items()) if k != "metric"}


def check_observables(config):
    """The observables in the configuration, checked, and the scenarios they are
    measured on, in order."""
    observables = config.get("observables") or {}
    if not observables:
        raise ObservableError("the configuration has no observables")
    scenarios = []
    for name, settings in observables.items():
        if name not in KINDS and name not in PER_EVENT:
            raise ObservableError("there's no observable '%s'; the known ones are: %s"
                                  % (name, ", ".join(list(KINDS) + list(PER_EVENT))))
        if not isinstance(settings, dict) or not settings.get("scenarios"):
            raise ObservableError("observables.%s needs a list of scenarios" % name)
        if name in PER_EVENT:
            check_particles(name, settings)
        else:
            KINDS[name][0](name, settings)  # check the binning early
            if not settings.get("hit_collections"):
                raise ObservableError("observables.%s needs hit_collections" % name)
        for scenario in settings["scenarios"]:
            if scenario not in scenarios:
                scenarios.append(scenario)
    return observables, scenarios


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


# The member of a reconstructed particle holding its PDG code: PDG since
# EDM4hep 0.99, type before
PDG_MEMBERS = ("PDG", "type")


def read_reconstructed_energy(branches, settings, observable, start, stop):
    """The reconstructed energy of each event: the sum of the energies of its
    reconstructed particles in the particle collections, or of those whose PDG
    code (either sign) is in pdg, if given. 0 for an event with none."""
    import awkward as ak
    import numpy as np
    codes = np.abs(np.array(settings.get("pdg") or [], dtype=np.int64))
    total = np.zeros(stop - start)
    for collection in settings["particle_collections"]:
        name = collection + ".energy"
        if name not in branches.keys:
            raise ObservableError("%s has no %s (does it have the collection %s? set "
                                  "observables.%s.particle_collections)"
                                  % (branches.path, name, collection, observable))
        energy = ak.values_astype(branches.jagged(name, start, stop), np.float64)
        if codes.size:
            member = next((m for m in PDG_MEMBERS if "%s.%s" % (collection, m) in branches.keys),
                          None)
            if member is None:
                raise ObservableError("the reconstructed particles of %s in %s have no PDG "
                                      "code (%s), which observables.%s.pdg needs"
                                      % (collection, branches.path, " or ".join(PDG_MEMBERS),
                                         observable))
            pdg = branches.jagged("%s.%s" % (collection, member), start, stop)
            keep = np.isin(np.abs(ak.to_numpy(ak.flatten(pdg)).astype(np.int64)), codes)
            energy = energy[ak.unflatten(keep, ak.num(pdg))]
        total += ak.to_numpy(ak.sum(energy, axis=1))
    return total


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


def wanted(observables):
    """[(observable, scenario, profile, edges, settings JSON)] that the observables
    in the configuration ask for; profile and edges are None for those stored
    per event."""
    result = []
    for name, settings in observables.items():
        settings_json = json.dumps(observable_settings(name, settings), sort_keys=True)
        if name in PER_EVENT:
            for scenario in settings["scenarios"]:
                result.append((name, scenario, None, None, settings_json))
            continue
        for profile, binning in KINDS[name][0](name, settings):
            for scenario in settings["scenarios"]:
                result.append((name, scenario, profile, bin_edges(binning), settings_json))
    return result


def is_up_to_date(path, profiles, files, relative=None):
    """The .npz file has every key asked for, with the same binning and settings,
    from the same reconstructed files ({scenario: [path]}), none of them
    reconstructed again since. relative(path) is how the files are recorded."""
    import numpy as np
    if not os.path.isfile(path):
        return False, "no %s yet" % os.path.basename(path)
    try:
        with np.load(path, allow_pickle=False) as data:
            keys = set(data.files)
            for name, scenario, profile, edges, settings_json in profiles:
                if profile is None:
                    for what in PER_EVENT[name] + ("settings",):
                        if event_key(name, scenario, what) not in keys:
                            return False, "%s has no %s" % (
                                os.path.basename(path), event_key(name, scenario, what))
                    if str(data[event_key(name, scenario, "settings")]) != settings_json:
                        return False, "the settings of %s changed" % name
                    continue
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
                stored = [str(f) for f in data[SEP.join((scenario, "files"))]]
                recorded = [relative(p) for p in paths] if relative else list(paths)
                # Files measured before the records were relative have full paths
                if stored != recorded and stored != list(paths):
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
    """Sums for every profile of one reconstructed file, what is stored of each
    event for the observables stored per event ({observable: {what: [event]}}),
    and its event counts."""
    import numpy as np
    profiled = dict((name, settings) for name, settings in observables.items()
                    if name in KINDS)
    by_event = dict((name, settings) for name, settings in observables.items()
                    if name in PER_EVENT)
    by_collection = {}
    for name, settings in profiled.items():
        for collection in settings["hit_collections"]:
            by_collection.setdefault(collection, set()).add(KINDS[name][1])
    sums = {key: [np.zeros(len(edges) - 1), np.zeros(len(edges) - 1)]
            for key, edges in profiles.items()}
    events = dict((name, dict((what, []) for what in PER_EVENT[name])) for name in by_event)
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
            for name, settings in by_event.items():
                reconstructed = read_reconstructed_energy(branches, settings, name, start, stop)
                events[name]["reconstructed"].append(reconstructed[has])
                events[name]["true"].append(true_energy[has])
            hits = read_hits(branches, list(by_collection), start, stop,
                             any("layer" in v for v in by_collection.values()))
            for collection, event, energy, position, cell_id in hits:
                values = {}
                if "radius" in by_collection[collection]:
                    offset = position - vertex[event]
                    along = np.einsum("ij,ij->i", offset, direction[event])
                    values["radius"] = np.linalg.norm(
                        offset - along[:, None] * direction[event], axis=1)
                for name, settings in profiled.items():
                    if collection not in settings["hit_collections"]:
                        continue
                    need = KINDS[name][1]
                    if need == "layer" and "layer" not in values:
                        field = cell_id_fields(settings, name, path, collection,
                                               simulator.get("compact_file"), cache)
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
    events = dict((name, dict((what, np.concatenate(parts) if parts else np.zeros(0))
                              for what, parts in stored.items()))
                  for name, stored in events.items())
    return sums, events, n_events, n_no_primary, mean_energy


def compute_simulator(observables, simulator, files, profiles, out_path, relative=None):
    """Measure one simulator: every profile, and observable stored per event
    (from wanted()), on its reconstructed files ({scenario: [path]}), into
    out_path. The files are recorded as relative(path)."""
    import numpy as np
    arrays = {
        "meta__simulator": np.array(simulator["name"]),
        "meta__label": np.array(simulator["label"]),
        "meta__color": np.array(simulator["color"]),
        "meta__linestyle": np.array(json.dumps(simulator["linestyle"])),
    }
    cache = {}
    for scenario, paths in files.items():
        measured = {name: settings for name, settings in observables.items()
                    if scenario in settings["scenarios"]}
        scenario_profiles = {(name, profile): edges
                             for name, scen, profile, edges, _ in profiles
                             if scen == scenario and profile is not None}
        per_file = {key: ([], []) for key in scenario_profiles}
        per_event = dict((name, dict((what, []) for what in PER_EVENT[name]))
                         for name in measured if name in PER_EVENT)
        counts, no_primary, energies = [], [], []
        for i, path in enumerate(paths, 1):
            say("  %s [%d/%d] %s" % (scenario, i, len(paths), os.path.basename(path)))
            sums, events, n, missing, energy = measure_file(path, measured, scenario_profiles,
                                                            simulator, cache)
            if missing:
                say("    %d event(s) without an incident particle (generator status 1) "
                    "were left out" % missing)
            for key in scenario_profiles:
                per_file[key][0].append(sums[key][0])
                per_file[key][1].append(sums[key][1])
            for name, stored in per_event.items():
                for what in stored:
                    stored[what].append(events[name][what])
            counts.append(n)
            no_primary.append(missing)
            energies.append(energy)
        arrays[SEP.join((scenario, "files"))] = np.array(
            [relative(p) for p in paths] if relative else paths)
        arrays[SEP.join((scenario, "n_events"))] = np.array(counts, dtype=np.int64)
        arrays[SEP.join((scenario, "n_no_primary"))] = np.array(no_primary, dtype=np.int64)
        arrays[SEP.join((scenario, "true_energy"))] = np.array(energies)
        for name, scen, profile, edges, settings_json in profiles:
            if scen != scenario:
                continue
            if profile is None:
                for what, parts in per_event[name].items():
                    arrays[event_key(name, scenario, what)] = (
                        np.concatenate(parts) if parts else np.zeros(0))
                arrays[event_key(name, scenario, "settings")] = np.array(settings_json)
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
