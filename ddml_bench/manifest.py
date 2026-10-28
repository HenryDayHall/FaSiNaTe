"""Manifests of a data folder, for publishing it (e.g. on Zenodo) as records
that can be fetched back into a data folder of the same layout.

  ddml-bench manifest [--data-dir DIR] --out OUT [--stage] [--records LIST]

The files of the data folder (paths.data_dir) go into records:

  essential                       what the pipeline needs to add a model without
                                  running anything of the others: the scenarios'
                                  inputs (<folder>/*.slcio of each scenario's
                                  input.folder: input_single_particle_gun,
                                  input_physics_samples) and the measurements of
                                  the full samples (observables/<simulator>.npz),
                                  all required
  input_truth__<scenario>         the guns' truth beside each scenario's inputs
                                  (.h5, optional: no stage reads it)
  sim_output__<scenario>__<simulator>
  rec_output__<scenario>__<simulator>
                                  the simulations and reconstructions of each
                                  scenario by each simulator, with their logs,
                                  file by file (archive: never fetched
                                  automatically); in parts, __part1, __part2,
                                  ..., where they are more than a record takes
                                  (--max-record-gb, and 99 files), each
                                  simulation or reconstruction's files together
  other                           the rest, small, as a tar archive per top
                                  folder (observables.tar: the plots and the test
                                  samples' .npz files; slurm.tar: the
                                  submissions and their slurm logs), members
                                  named by their paths in the data folder

Each record gets OUT/<record>/manifest.json; with --stage, OUT/<record>/ also
gets the files to upload under the flat names the manifest gives them: links
to the data folder's files (copies with --copy), so nothing big is copied,
and the other record's tar archives.

A manifest says, for each file: its name in the record, its path in the data
folder, its sha256 and size, and its role: required (fetched by the
pipeline), optional (fetched on request) or archive (never fetched
automatically). An input says which scenarios have it and the gun options
that made it, or that it was made elsewhere (external: true; e.g. a physics
sample of ILD's MC production). A measurement says which simulator and scenarios it holds, the
settings of each observable in it, the reconstructions it was measured from,
with their ddsim seeds and events, the inputs those were simulated from, with
their sha256 (derived_from), the software they were made with, from the logs
of those simulations and reconstructions (stack), a model's description from
models/ (model: its presets or DDML model configurations, detector and
weights with their md5 sum), and whether it is up to date
with this configuration, as the observables stage decides; the essential
record's stack is what all of them should share (Key4hep, DDML, ILDConfig,
the patches, the reconstruction's arguments and the detector), with a
warning where they don't. A tar archive lists its members, each with its
sha256 and size.

Every manifest also has a suggested title, the configuration's scenarios and
observables, the settings of the simulation and reconstruction and the pins
of the software stack (configuration), and the commit of this repository
(fasinate): the commit writing the manifest, not necessarily the one that made
the data, which the .npz files don't record. The essential record's manifest
also lists every record (catalogue): its title, files and size, so that the
essential record says where everything else is.

It needs numpy, to read the .npz files, and the configuration (PyYAML) for
the scenarios, the observables and the check of the measurements, without
which it says so and goes on.
"""

import argparse
import datetime
import fnmatch
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile

from .config import REPO_DIR
from .ddml_configs import OPTION as ML_CONFIG, canonical
from .layout import (INPUT_DIR, INPUT_FOLDERS, OBSERVABLES_DIR, REC_DIR, REC_SUFFIX, SEED_TAG,
                     SEP, SIM_DIR, input_stem)

FORMAT = "fasinate-manifest/2"
ESSENTIAL, TRUTH, OTHER = "essential", "input_truth", "other"
REQUIRED, OPTIONAL, ARCHIVE = "required", "optional", "archive"
# What Zenodo takes in one record, by default (it can raise a record's quota
# on request); the manifest is one of the files
MAX_FILES = 100
MAX_GB = 50
BLOCK = 1 << 22

_TEST_SAMPLE = re.compile(r"_n\d+$")
_SEED = re.compile(re.escape(SEED_TAG) + r"\d+$")
# The endings of a simulation's or reconstruction's files, after its name
_SAMPLE_FILE = re.compile(r"(_REC\.edm4hep\.root|_REC\.slcio|_AIDA\.root|_PfoAnalysis\.root"
                          r"|\.edm4hep\.root|\.root|\.slcio|\.log)$")
# A line of the header that the simulation and reconstruction stages write at
# the top of their logs (and the scripts before them did), e.g.
# "# Key4hep   2026-04-08"
_HEADER = re.compile(r"^# (\w+)\s+(.*?)\s*$")
_COMPACT = re.compile(r"([^/]+/compact/.+)$")
# The arguments of ILDReconstruction.py that the reconstruct stage sets itself
_MANAGED = ("--inputFiles", "--outputFileBase", "--compactFile", "--num-events")
# and those it sets by the scenario (for a particle gun's events, not a physics
# sample's), which aren't the software's, so not its stack
_BY_SCENARIO = ("--usingParticleGun",)
# What the simulations and reconstructions of every simulator should share
SHARED = ("key4hep", "ddml", "ildconfig", "patches", "reconstruction_args", "compact")
STAGE_WORDS = {SIM_DIR: "simulations", REC_DIR: "reconstructions"}
TITLE = "FaSiNaTe benchmark data"


class ManifestError(Exception):
    pass


def say(message):
    print("manifest: " + message, file=sys.stderr)


# ---------------------------------------------------------------------------
# The files of the data folder, and the record each goes into


def classify(relative, truth_scenarios=None, input_folders=INPUT_FOLDERS):
    """(record, role, name in the record, packed) of a file, by its path
    relative to the data folder: packed is the name of the tar archive it goes
    into, or None for a file of its own. truth_scenarios: {input (its path
    relative to the data folder): scenario}, which names the record of its
    truth; input_folders: the folders of the scenarios' inputs."""
    parts = relative.split("/")
    name = parts[-1]
    if len(parts) == 2 and parts[0] in input_folders and name.endswith(".slcio"):
        return ESSENTIAL, REQUIRED, name, None
    if len(parts) == 2 and parts[0] == OBSERVABLES_DIR and name.endswith(".npz") \
            and not _TEST_SAMPLE.search(name[:-len(".npz")]):
        return ESSENTIAL, REQUIRED, name, None
    if len(parts) == 2 and parts[0] in input_folders and name.endswith(".h5"):
        scenario = (truth_scenarios or {}).get("%s/%s.slcio" % (parts[0], input_stem(name)))
        return (TRUTH + SEP + scenario if scenario else TRUTH), OPTIONAL, name, None
    if parts[0] in (SIM_DIR, REC_DIR) and len(parts) >= 4:
        return SEP.join(parts[:3]), ARCHIVE, SEP.join(parts[3:]), None
    group = parts[0] if len(parts) > 1 else "loose"  # loose: files at the top
    return OTHER, ARCHIVE, group + ".tar", group + ".tar"


def sample_of(name):
    """The simulation or reconstruction a file is of (its files go together)."""
    return _SAMPLE_FILE.sub("", name)


def title(record):
    """A title for a record, e.g. for Zenodo."""
    base, _, part = record.partition(SEP + "part")
    words = base.split(SEP)
    if base == ESSENTIAL:
        text = "inputs and measurements, all the pipeline needs"
    elif words[0] == TRUTH:
        text = "the inputs' generated truth" + (", %s" % words[1] if len(words) > 1 else "")
    elif words[0] in STAGE_WORDS and len(words) == 3:
        text = "%s %s, %s" % ("Geant4" if words[2] == "geant4" else words[2],
                              STAGE_WORDS[words[0]], words[1])
    else:
        text = "plots, test samples and job logs"
    return "%s: %s%s" % (TITLE, text, " (part %s)" % part if part else "")


def split(record, files, max_bytes, warnings):
    """[(record name, files)]: the files of a record in parts no bigger than a
    record takes, each simulation's or reconstruction's files together."""
    if record in (ESSENTIAL, OTHER) or record.startswith(TRUTH):
        return [(record, files)]
    groups = {}
    for entry in files:
        groups.setdefault(sample_of(entry["name"]), []).append(entry)
    parts, size = [[]], 0
    for sample in sorted(groups):
        group = groups[sample]
        bytes_ = sum(e["size"] for e in group)
        if parts[-1] and (size + bytes_ > max_bytes
                          or len(parts[-1]) + len(group) > MAX_FILES - 1):
            parts.append([])
            size = 0
        if bytes_ > max_bytes:
            warnings.append("%s: the files of %s are %.1f GB, more than a record takes "
                            "(--max-record-gb), so they make a record of their own that is too "
                            "big" % (record, sample, bytes_ / 1e9))
        parts[-1].extend(group)
        size += bytes_
    if len(parts) == 1:
        return [(record, parts[0])]
    return [("%s%spart%d" % (record, SEP, i), part) for i, part in enumerate(parts, 1)]


def scan(data_dir, leave_out=()):
    """The paths, relative to the data folder, of its files, sorted; without
    hidden ones (unfinished, or the harness's own) or the folders left out."""
    leave_out = [os.path.realpath(p) for p in leave_out]
    found = []
    for root, folders, names in os.walk(data_dir):
        folders[:] = sorted(f for f in folders if not f.startswith(".")
                            and os.path.realpath(os.path.join(root, f)) not in leave_out)
        for name in sorted(names):
            path = os.path.join(root, name)
            if not name.startswith(".") and os.path.isfile(path):
                found.append(os.path.relpath(path, data_dir).replace(os.sep, "/"))
    return found


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# The software, from the headers of the logs of the simulations and
# reconstructions


def read_log(path):
    """{field: value} of a log's header, and what its command says: the
    Key4hep release, the DDML commit (simulations), the ILDConfig commit and
    the patches (reconstructions), the seed, the detector, the models'
    presets or DDML model configurations (--ml-config, as one string,
    ddml_configs.canonical()) 
    and the reconstruction's arguments (but --usingParticleGun, which
    the stage gives the particle guns' scenarios and not the physics
    samples').
    {} without a log."""
    try:
        with open(path, errors="replace") as stream:
            lines = [stream.readline() for _ in range(40)]
    except OSError:
        return {}
    header = {}
    for line in lines:
        match = _HEADER.match(line.rstrip("\n"))
        if match and match.group(1) not in header:
            header[match.group(1)] = match.group(2)
    found = {}
    if "Key4hep" in header:
        found["key4hep"] = header["Key4hep"]
    if "DDML" in header:
        found["ddml"] = header["DDML"].split(",")[0].strip()
    if "ILDConfig" in header:
        found["ildconfig"] = header["ILDConfig"].split(",")[0].strip()
    if "patches" in header:
        found["patches"] = header["patches"].split(" (")[0].strip()
    if header.get("seed", "").isdigit():
        found["seed"] = int(header["seed"])
    command = header.get("command", "")
    if ") " in command:
        try:
            args = shlex.split(command.split(") ", 1)[1])
        except ValueError:
            args = []
        found.update(_from_command(args))
    return found


def _from_command(args):
    found, presets, configs, extra = {}, [], [], []
    rec = any(os.path.basename(a) == "k4run" for a in args[:1])
    i = 0
    while i < len(args):
        arg = args[i]
        key, _, value = arg.partition("=")
        if key in ("--compactFile", "--steeringFile", "--ml-model", ML_CONFIG) and not value \
                and i + 1 < len(args):
            value = args[i + 1]
            i += 1
        if key == "--compactFile":
            match = _COMPACT.search(value)
            found["compact"] = match.group(1) if match else value
        elif key == "--steeringFile":
            found["steering"] = os.path.basename(value)
        elif key == "--ml-model":
            presets.append(value)
        elif key == ML_CONFIG:
            try:
                configs.append(json.loads(value))
            except ValueError:
                configs.append(value)
        elif rec and i > 1 and key not in _MANAGED and key not in _BY_SCENARIO:
            extra.append(arg)
        i += 1
    if presets:
        found["presets"] = " ".join(presets)
    if configs:
        found["model_configs"] = canonical(configs)
    if rec:
        found["reconstruction_args"] = " ".join(extra)
    return found


def summarise(readings):
    """{field: value, or {value: how many logs} where they differ} of the logs
    read."""
    values = {}
    for reading in readings:
        for field, value in reading.items():
            if field != "seed":
                values.setdefault(field, {}).setdefault(value, 0)
                values[field][value] += 1
    return dict((field, next(iter(counts)) if len(counts) == 1 else counts)
                for field, counts in sorted(values.items()))


# ---------------------------------------------------------------------------
# What a measurement holds, from its .npz file


def recorded_path(path):
    """A reconstructed file as a .npz file records it, relative to the data
    folder: as recorded, or, for full paths, from rec_output on."""
    path = str(path)
    if not os.path.isabs(path):
        return path
    marker = "/" + REC_DIR + "/"
    return path[path.rindex(marker) + 1:] if marker in path else path


def reconstruction_sources(relative, input_folders=None):
    """(input, simulation log, reconstruction log), relative to the data
    folder, of a reconstructed file; input_folders: {scenario: the folder of
    its inputs}, input_single_particle_gun for those it hasn't."""
    parts = relative.split("/")
    name = parts[-1]
    base = name[:-len(REC_SUFFIX)] if name.endswith(REC_SUFFIX) else os.path.splitext(name)[0]
    stem = _TEST_SAMPLE.sub("", _SEED.sub("", base))
    folder = "/".join(parts[1:-1])  # <scenario>/<simulator>
    inputs = (input_folders or {}).get(parts[1] if len(parts) > 2 else None, INPUT_DIR)
    return ("%s/%s.slcio" % (inputs, stem), "%s/%s/%s.log" % (SIM_DIR, folder, base),
            "%s/%s/%s.log" % (REC_DIR, folder, base))


def read_measurements(path):
    """What a .npz file holds: its simulator, scenarios, observables and their
    settings, and the reconstructed files of each scenario with their seeds
    and events."""
    try:
        import numpy as np
    except ImportError:
        raise ManifestError("reading the .npz files needs numpy (python3 -m pip install --user "
                            "numpy, or run this in the environment: source setup.sh)")
    with np.load(path, allow_pickle=False) as data:
        keys = list(data.files)
        found = {"simulator": str(data["meta__simulator"]) if "meta__simulator" in keys
                 else os.path.basename(path)[:-len(".npz")],
                 "label": str(data["meta__label"]) if "meta__label" in keys else None,
                 "scenarios": {}, "observables": {}}
        for key in keys:
            parts = key.split(SEP)
            if len(parts) == 2 and parts[1] == "files":
                scenario = parts[0]
                files = [recorded_path(f) for f in data[key]]
                seeds = (data[SEP.join((scenario, "seeds"))].tolist()
                         if SEP.join((scenario, "seeds")) in keys else [-1] * len(files))
                events = (data[SEP.join((scenario, "n_events"))].tolist()
                          if SEP.join((scenario, "n_events")) in keys else [None] * len(files))
                found["scenarios"][scenario] = [
                    {"path": f, "seed": None if s == -1 else int(s), "events": n}
                    for f, s, n in zip(files, seeds, events)]
            elif parts[-1] == "settings" and len(parts) in (3, 4):
                entry = found["observables"].setdefault(parts[0], {
                    "settings": json.loads(str(data[key])), "scenarios": [], "profiles": []})
                if parts[1] not in entry["scenarios"]:
                    entry["scenarios"].append(parts[1])
                if len(parts) == 4 and parts[2] not in entry["profiles"]:
                    entry["profiles"].append(parts[2])
    for entry in found["observables"].values():
        if not entry["profiles"]:
            del entry["profiles"]
    return found


# ---------------------------------------------------------------------------
# The configuration, and the check of the measurements against it


def configuration():
    """The configuration, or None (said why) if it can't be read."""
    try:
        from .config import load_config
        return load_config()
    except Exception as error:  # no PyYAML, or a bad configuration
        say("warning: can't read the configuration (%s), so the manifests won't have it, the "
            "inputs' scenarios or the check of the measurements" % error)
        return None


def configuration_snapshot(config):
    """What of the configuration the data depends on (not where things are)."""
    ildconfig = config["detector_configs"]["ildconfig"]
    return {
        "scenarios": config.get("scenarios"),
        "observables": config.get("observables"),
        "geant4": config.get("geant4"),
        "reconstruction": config.get("reconstruction"),
        "pins": {
            "key4hep.release": config["key4hep"].get("release"),
            "ddml.repository": config["ddml"].get("repository"),
            "ddml.branch": config["ddml"].get("branch"),
            "ddml.commit": config["ddml"].get("commit"),
            "ildconfig.repository": ildconfig.get("repository"),
            "ildconfig.branch": ildconfig.get("branch"),
            "ildconfig.commit": ildconfig.get("commit"),
        },
    }


def input_commands(config, data_dir):
    """{input (its path relative to the data folder): {"scenarios",
    "generator", "options"}} of the configuration's scenarios, disabled ones
    too (with "enabled": false); the options are the gun's, without where it
    writes. An input made elsewhere has "external": true instead of the
    generator and options."""
    from .layout import Layout
    from .scenarios import scenario_inputs
    layout = Layout(data_dir)
    found = {}
    for group in scenario_inputs(config, layout):
        for entry in group["inputs"]:
            command = entry["command"]
            if command is None:
                described = {"external": True}
            else:
                options, skip = [], False
                for arg in command[2:]:  # after python3 and the script
                    if skip:
                        skip = False
                    elif arg == "--save_dir":
                        skip = True
                    else:
                        options.append(arg)
                described = {"generator": os.path.basename(command[1]), "options": options}
            item = found.setdefault(layout.relative(entry["file"]).replace(os.sep, "/"),
                                    dict(described, scenarios=[], enabled=False))
            item["scenarios"].append(group["scenario"])
            item["enabled"] = item["enabled"] or group["enabled"]
    return found


def input_folders(config, data_dir):
    """{scenario: the folder of its inputs} of the configuration's scenarios
    ({} without one)."""
    from .layout import Layout
    from .scenarios import scenario_inputs
    if not config:
        return {}
    return dict((group["scenario"], group["folder"])
                for group in scenario_inputs(config, Layout(data_dir)))


def measurement_status(data_dir):
    """{.npz path: (up to date, why not)} of the simulators' measurements, as
    the observables stage decides them with this configuration (in this data
    folder); {} if the plan can't be made."""
    from .graph import Graph
    from .stages import BY_NAME
    try:
        graph = Graph(_plan_in(data_dir))
    except Exception as error:  # PlanError, a bad model, ...
        say("warning: can't check the measurements against the configuration: %s" % error)
        return {}
    stage = BY_NAME["observables"]
    return dict((os.path.realpath(task["output"]), stage.current(task))
                for task in graph.list if task["stage"] == "observables")


def _plan_in(data_dir):
    """The plan of the pipeline, with data_dir as paths.data_dir."""
    from .graph import make_plan
    old = os.environ.get("FASINATE_CONFIG")
    handle, override = tempfile.mkstemp(suffix=".yaml", prefix=".manifest.")
    try:
        with os.fdopen(handle, "w") as stream:
            stream.write("paths:\n  data_dir: %s\n" % json.dumps(os.path.abspath(data_dir)))
        os.environ["FASINATE_CONFIG"] = ":".join([f for f in (old or "").split(":") if f]
                                                 + [override])
        return make_plan()
    finally:
        os.remove(override)
        if old is None:
            os.environ.pop("FASINATE_CONFIG", None)
        else:
            os.environ["FASINATE_CONFIG"] = old


def described_models():
    """{model: what of models/<model>.yaml decides its showers}, for the
    measurements of the models: the presets or DDML model configurations, the
    detector and the weights (with their md5 sum), which fetching published
    measurements checks."""
    try:
        from .models import load_models
        models = load_models()
    except Exception as error:  # no PyYAML, or a model's description is wrong
        say("warning: can't read the models' descriptions (%s), so the manifest won't have "
            "them" % error)
        return {}
    return dict((m["name"], {"config": m["config"], "presets": list(m["presets"]),
                             "model_configs": list(m["model_configs"]),
                             "compact_file": m["compact_file"], "weights": m["weights"],
                             "weights_md5": m["weights_md5"]}) for m in models)


def repository_commit():
    """The commit of this repository, and whether it has changes not committed."""
    def git(*args):
        return subprocess.check_output(["git", "-C", REPO_DIR] + list(args),
                                       stderr=subprocess.DEVNULL, universal_newlines=True)
    try:
        return {"commit": git("rev-parse", "HEAD").strip(),
                "dirty": bool(git("status", "--porcelain", "--untracked-files=no").strip())}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


# ---------------------------------------------------------------------------
# Building the manifests


class Hashing(object):
    """A file object that hashes what goes through it: read from source, or
    written to sink."""

    def __init__(self, source=None, sink=None):
        self.source, self.sink = source, sink
        self.digest = hashlib.sha256()
        self.size = 0

    def read(self, size=-1):
        data = self.source.read(size)
        self.digest.update(data)
        self.size += len(data)
        return data

    def write(self, data):
        self.digest.update(data)
        self.size += len(data)
        return self.sink.write(data)

    def hexdigest(self):
        return self.digest.hexdigest()


def pack(archive, members, data_dir):
    """Write the members (paths relative to the data folder) to a tar archive
    under those names, keeping their times; returns the archive's sha256 and
    size, and each member's."""
    hashes = {}
    with open(archive + ".partial", "wb") as stream:
        out = Hashing(sink=stream)
        with tarfile.open(fileobj=out, mode="w|", format=tarfile.PAX_FORMAT) as tar:
            for relative in members:
                path = os.path.join(data_dir, relative)
                info = tar.gettarinfo(path, arcname=relative)
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                with open(path, "rb") as source:
                    reader = Hashing(source=source)
                    tar.addfile(info, reader)
                hashes[relative] = (reader.hexdigest(), reader.size)
    os.replace(archive + ".partial", archive)
    return out.hexdigest(), os.path.getsize(archive), hashes


def plan_records(data_dir, leave_out=(), truth_scenarios=None, max_bytes=MAX_GB * 10 ** 9,
                 warnings=None, folders=INPUT_FOLDERS):
    """{record: [file entries]} of the data folder, without hashing: each
    entry its name, path (or members, for a tar archive), size and role;
    folders are the folders of the inputs."""
    warnings = [] if warnings is None else warnings
    by_record = {}
    for relative in scan(data_dir, leave_out):
        record, role, name, packed = classify(relative, truth_scenarios, folders)
        size = os.path.getsize(os.path.join(data_dir, relative))
        files = by_record.setdefault(record, [])
        if packed:
            entry = next((e for e in files if e["name"] == packed), None)
            if entry is None:
                entry = {"name": packed, "role": role, "kind": "archive", "size": 0,
                         "members": []}
                files.append(entry)
            entry["members"].append({"path": relative, "size": size})
            entry["size"] += size
        else:
            files.append({"name": name, "path": relative, "size": size, "role": role})
    records = {}
    for record in sorted(by_record):
        for name, files in split(record, by_record[record], max_bytes, warnings):
            records[name] = files
    return records


def catalogue(records):
    """What every record holds, for the essential record's manifest."""
    return dict((name, {"title": title(name), "files": len(files),
                        "size": sum(e["size"] for e in files),
                        "paths": [e["path"] for e in files if "path" in e]
                        + [m["path"] for e in files for m in e.get("members", [])]})
                for name, files in records.items())


def build(data_dir, out, only=(), stage=False, copy=False, config=None, check=True,
          max_gb=MAX_GB):
    """Write OUT/<record>/manifest.json for each record, or those matching the
    patterns in only (and, with stage, the files to upload beside it);
    returns {record: manifest}."""
    data_dir = os.path.abspath(data_dir)
    if not os.path.isdir(data_dir):
        raise ManifestError("no data folder %s" % data_dir)
    warnings = []
    commands = input_commands(config, data_dir) if config else {}
    truth = dict((path, item["scenarios"][0]) for path, item in commands.items())
    by_scenario = input_folders(config, data_dir)
    folders = tuple(sorted(set(INPUT_FOLDERS) | set(by_scenario.values())))
    records = plan_records(data_dir, [out], truth, max_gb * 10 ** 9, warnings, folders)
    chosen = [r for r in records if not only or any(fnmatch.fnmatch(r, p) for p in only)]
    if only and not chosen:
        raise ManifestError("no record matches %s; the records are: %s"
                            % (", ".join(only), ", ".join(records)))
    say("%d files in %s, in %d records" % (sum(len(f) for f in records.values()), data_dir,
                                          len(records)))
    common = {"format": FORMAT,
              "created": datetime.datetime.now(datetime.timezone.utc).strftime(
                  "%Y-%m-%dT%H:%M:%SZ"),
              "fasinate": repository_commit(),
              "configuration": configuration_snapshot(config) if config else None}

    # The inputs are hashed whichever records are made, for the measurements'
    # derived_from
    inputs = {}
    for entry in records[ESSENTIAL] if ESSENTIAL in records else []:
        if entry["path"].split("/")[0] in folders and entry["path"].endswith(".slcio"):
            inputs[entry["path"]] = entry
            entry.update(sha256=sha256(os.path.join(data_dir, entry["path"])), kind="input")
            if entry["path"] in commands:
                entry.update((key, value) for key, value in commands[entry["path"]].items()
                             if key != "enabled")
            elif config:
                warnings.append("%s isn't an input of any scenario in the configuration"
                                % entry["path"])
    # (A disabled scenario's inputs are described if they are there, and not
    # missed if they aren't)
    for path, item in commands.items():
        if path not in inputs and item["enabled"]:
            warnings.append("the configuration's input %s isn't in the data folder%s" % (
                path, ", so runs here leave out %s, and its measurements with it"
                % " and ".join(item["scenarios"]) if item.get("external") else ""))
    status = measurement_status(data_dir) if config and check and ESSENTIAL in chosen else {}
    models = described_models() if ESSENTIAL in chosen else {}

    manifests = {}
    # The essential record last, as its catalogue says what the others hold
    for record in sorted(chosen, key=lambda r: r == ESSENTIAL):
        files, logs = records[record], []
        for i, entry in enumerate(files):
            if record == ESSENTIAL and entry["path"].startswith(OBSERVABLES_DIR + "/"):
                files[i] = _measurement(data_dir, entry["path"], entry["role"], entry["name"],
                                        inputs, status, warnings, logs, by_scenario)
                if files[i]["simulator"] in models:
                    files[i]["model"] = models[files[i]["simulator"]]
            elif "members" in entry:
                _archive(entry, data_dir, os.path.join(out, record), stage)
            elif "sha256" not in entry:
                entry["sha256"] = sha256(os.path.join(data_dir, entry["path"]))
                entry.setdefault("kind", "truth" if entry["role"] == OPTIONAL else "file")
        folder = os.path.join(out, record)
        os.makedirs(folder, exist_ok=True)
        names = [entry["name"] for entry in files]
        doubled = sorted({n for n in names if names.count(n) > 1})
        if doubled:
            raise ManifestError("the %s record would have two files named %s; Zenodo's records "
                                "are flat, so rename one" % (record, ", ".join(doubled)))
        if stage:
            for entry in files:
                if "path" in entry:
                    _place(os.path.join(data_dir, entry["path"]),
                           os.path.join(folder, entry["name"]), copy)
        manifest = dict(common, record=record, title=title(record), files=files)
        if record == ESSENTIAL:
            manifest["catalogue"] = catalogue(records)
            # What every simulator should share (not the steering or presets)
            manifest["stack"] = dict((field, value) for field, value in summarise(logs).items()
                                     if field in SHARED)
            for field, value in manifest["stack"].items():
                if isinstance(value, dict):
                    warnings.append("the measurements were made with different %s: %s" % (
                        field, ", ".join("%s (%d logs)" % item for item in value.items())))
        total = sum(entry["size"] or 0 for entry in files)
        if len(files) + 1 > MAX_FILES or total > max_gb * 10 ** 9:
            warnings.append("the %s record has %d files (with the manifest) and %.1f GB, more "
                            "than %d files or %g GB" % (record, len(files) + 1, total / 1e9,
                                                       MAX_FILES, max_gb))
        with open(os.path.join(folder, "manifest.json"), "w") as stream:
            json.dump(manifest, stream, indent=1, sort_keys=True)
            stream.write("\n")
        manifests[record] = manifest
        say("%s: %d files, %.3g GB -> %s" % (record, len(files), total / 1e9,
                                             os.path.join(folder, "manifest.json")))
    for warning in warnings:
        say("warning: " + warning)
    return manifests


def _measurement(data_dir, relative, role, name, inputs, status, warnings, logs,
                 input_folders=None):
    path = os.path.join(data_dir, relative)
    held = read_measurements(path)
    entry = {"name": name, "path": relative, "sha256": sha256(path),
             "size": os.path.getsize(path), "role": role, "kind": "measurements",
             "simulator": held["simulator"], "label": held["label"],
             "scenarios": sorted(held["scenarios"]), "observables": held["observables"],
             "reconstructions": {}, "derived_from": []}
    readings, missing_logs = [], 0
    for scenario in sorted(held["scenarios"]):
        entry["reconstructions"][scenario] = held["scenarios"][scenario]
        for reconstruction in held["scenarios"][scenario]:
            source, sim_log, rec_log = reconstruction_sources(reconstruction["path"],
                                                              input_folders)
            if source in inputs:
                if source not in [d["path"] for d in entry["derived_from"]]:
                    entry["derived_from"].append({"path": source,
                                                  "sha256": inputs[source]["sha256"]})
            else:
                warnings.append("%s was measured from %s, simulated from %s, which isn't in "
                                "the data folder" % (relative, reconstruction["path"], source))
            for log in (sim_log, rec_log):
                reading = read_log(os.path.join(data_dir, log))
                missing_logs += not reading
                if reading:
                    readings.append(reading)
                if reading.get("seed") is not None and reconstruction["seed"] is None:
                    reconstruction["seed"] = reading["seed"]
    entry["stack"] = summarise(readings)
    logs.extend(readings)
    if missing_logs:
        warnings.append("%s: %d of the logs of its simulations and reconstructions aren't "
                        "there, so its stack is from the others" % (relative, missing_logs))
    for field, value in entry["stack"].items():
        if isinstance(value, dict):
            warnings.append("%s: its simulations and reconstructions were made with different "
                            "%s: %s" % (relative, field, ", ".join(
                                "%s (%d logs)" % item for item in value.items())))
    if os.path.realpath(path) in status:
        done, why = status[os.path.realpath(path)]
        entry["up_to_date"] = done
        if not done:
            entry["why_not"] = why
            warnings.append("%s isn't up to date with this configuration (%s), so the pipeline "
                            "would measure it again" % (relative, why))
    elif status:
        entry["up_to_date"] = None
        warnings.append("%s isn't the measurements of a simulator of this configuration"
                        % relative)
    return entry


def _place(source, target, copy):
    if os.path.lexists(target):
        os.remove(target)
    if copy:
        shutil.copy2(source, target)
    else:
        os.symlink(os.path.realpath(source), target)


def _archive(entry, data_dir, folder, stage):
    members = [member["path"] for member in entry["members"]]
    if stage:
        os.makedirs(folder, exist_ok=True)
        digest, size, hashes = pack(os.path.join(folder, entry["name"]), members, data_dir)
        entry.update(sha256=digest, size=size)
    else:
        hashes = dict((p, (sha256(os.path.join(data_dir, p)),
                           os.path.getsize(os.path.join(data_dir, p)))) for p in members)
        entry.update(sha256=None, size=None)
    for member in entry["members"]:
        member["sha256"], member["size"] = hashes[member["path"]]


def dry_run(data_dir, leave_out=(), truth_scenarios=None, max_gb=MAX_GB, folders=INPUT_FOLDERS):
    """What would go where, without hashing or writing anything."""
    warnings = []
    records = plan_records(data_dir, leave_out, truth_scenarios, max_gb * 10 ** 9, warnings,
                           folders)
    for record, files in records.items():
        print("%s: %d files, %.3g GB  (%s)" % (record, len(files),
                                              sum(e["size"] for e in files) / 1e9,
                                              title(record)))
        for entry in files:
            print("  %-9s %-64s %10.4g MB%s" % (
                entry["role"], entry["name"], entry["size"] / 1e6,
                "  (%d files)" % len(entry["members"]) if "members" in entry else ""))
    for warning in warnings:
        say("warning: " + warning)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="ddml-bench manifest",
        description="Write the manifests of a data folder's records for publishing it: the "
                    "essential record (inputs and measurements), and the rest by stage, "
                    "scenario and simulator; see ddml_bench/manifest.py.")
    parser.add_argument("--data-dir", help="the data folder (default: paths.data_dir)")
    parser.add_argument("--out", help="where the records go, a folder each with its "
                                      "manifest.json (required unless --dry-run)")
    parser.add_argument("--records", default="",
                        help="only the records matching these patterns, separated by commas, "
                             "e.g. 'essential,*__di_photon__*' (default: all)")
    parser.add_argument("--max-record-gb", type=float, default=MAX_GB, metavar="GB",
                        help="split the simulations and reconstructions into records of at most "
                             "this much (default: %g, Zenodo's quota without asking)" % MAX_GB)
    parser.add_argument("--stage", action="store_true",
                        help="also put the files to upload in each record's folder: links to "
                             "the data folder's files, and the other record's tar archives")
    parser.add_argument("--copy", action="store_true",
                        help="with --stage, copy the files instead of linking to them")
    parser.add_argument("--no-check", action="store_true",
                        help="don't check the measurements against the configuration")
    parser.add_argument("-c", "--config", action="append", default=[], metavar="FILE",
                        help="merge FILE over the configuration (repeatable)")
    parser.add_argument("--dry-run", action="store_true",
                        help="show what goes into which record, then stop")
    args = parser.parse_args(argv)
    from .cli import use_config_files
    use_config_files(args.config)
    if not args.max_record_gb > 0:
        parser.error("--max-record-gb needs a size above 0")
    only = [r.strip() for r in args.records.split(",") if r.strip()]
    config = configuration()
    data_dir = args.data_dir or (config or {}).get("paths", {}).get("data_dir")
    if not data_dir:
        parser.error("give --data-dir, as the configuration can't be read")
    try:
        if args.dry_run:
            truth = dict((path, item["scenarios"][0]) for path, item in
                         (input_commands(config, data_dir) if config else {}).items())
            folders = tuple(sorted(set(INPUT_FOLDERS)
                                   | set(input_folders(config, data_dir).values())))
            dry_run(data_dir, [args.out] if args.out else [], truth, args.max_record_gb, folders)
            return 0
        if not args.out:
            parser.error("give --out, the folder the records go to")
        build(data_dir, os.path.abspath(args.out), only, args.stage, args.copy, config,
              check=not args.no_check, max_gb=args.max_record_gb)
    except ManifestError as error:
        say(str(error))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
