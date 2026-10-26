"""Manifests of a data folder, for publishing it (e.g. on Zenodo) as records
that can be fetched back into a data folder of the same layout.

  ddml-bench manifest [--data-dir DIR] --out OUT [--stage] [--records LIST]

The files of the data folder (paths.data_dir) go into three records, by how
often they change and whether the pipeline fetches them:

  inputs        input_single_particle_gun/: the scenarios' inputs (.slcio,
                required: every simulator simulates these events) and the
                guns' truth beside them (.h5, optional: no stage reads it)
  measurements  observables/<simulator>.npz of the full samples (required:
                everything the figures and metrics are made from)
  bulk          everything else, never fetched by the pipeline, packed into
                one tar archive per folder: sim_output/<scenario>/<simulator>
                and rec_output/<scenario>/<simulator> each (the simulations,
                reconstructions and their logs), and the rest of each top
                folder (observables: the plots and the test samples' .npz
                files; slurm: the submissions and their slurm logs)

Each record gets OUT/<record>/manifest.json; with --stage, OUT/<record>/ also
gets the files to upload under the flat names the manifest gives them (links
to the inputs and measurements, or copies with --copy, and the bulk record's
tar archives, whose members are named by their paths in the data folder, so
that unpacking one in a data folder puts them in place).

A manifest says, for each file: its name in the record, its path in the data
folder, its sha256 and size, and its role: required (fetched by the
pipeline), optional (fetched on request) or archive (never fetched
automatically). An input says which scenarios have it and the gun options
that made it. A measurement says which simulator and scenarios it holds, the
settings of each observable in it, the reconstructions it was measured from,
with their ddsim seeds and events, the inputs those were simulated from, with
their sha256 (derived_from), the software they were made with, from the logs
of those simulations and reconstructions (stack), and whether it is up to date
with this configuration, as the observables stage decides; the record's stack
is what all of them should share (Key4hep, DDML, ILDConfig, the patches, the
reconstruction's arguments and the detector), with a warning where they
don't. A bulk archive lists its members, each with its sha256 and size.

Every manifest also has the configuration's scenarios and observables, the
settings of the simulation and reconstruction and the pins of the software
stack (configuration), and the commit of this repository (fasinate): the
commit writing the manifest, not necessarily the one that made the data,
which the .npz files don't record.

It needs numpy, to read the .npz files, and the configuration (PyYAML) for
the scenarios, the observables and the check of the measurements, without
which it says so and goes on.
"""

import argparse
import datetime
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
from .layout import (INPUT_DIR, OBSERVABLES_DIR, REC_DIR, REC_SUFFIX, SEED_TAG, SEP, SIM_DIR,
                     input_stem)

FORMAT = "fasinate-manifest/1"
RECORDS = ("inputs", "measurements", "bulk")
REQUIRED, OPTIONAL, ARCHIVE = "required", "optional", "archive"
# What Zenodo takes in one record, by default
MAX_FILES = 100
MAX_BYTES = 50 * 10 ** 9
BLOCK = 1 << 22

_TEST_SAMPLE = re.compile(r"_n\d+$")
_SEED = re.compile(re.escape(SEED_TAG) + r"\d+$")
# A line of the header that the simulation and reconstruction stages write at
# the top of their logs (and the scripts before them did), e.g.
# "# Key4hep   2026-04-08"
_HEADER = re.compile(r"^# (\w+)\s+(.*?)\s*$")
_COMPACT = re.compile(r"([^/]+/compact/.+)$")
# The arguments of ILDReconstruction.py that the reconstruct stage sets itself
_MANAGED = ("--inputFiles", "--outputFileBase", "--compactFile", "--num-events")
# What the simulations and reconstructions of every simulator should share
SHARED = ("key4hep", "ddml", "ildconfig", "patches", "reconstruction_args", "compact")


class ManifestError(Exception):
    pass


def say(message):
    print("manifest: " + message, file=sys.stderr)


# ---------------------------------------------------------------------------
# The files of the data folder, and the record each goes into


def classify(relative):
    """(record, role, name in the record) of a file, by its path relative to
    the data folder; the name of a bulk file is its archive's."""
    parts = relative.split("/")
    name = parts[-1]
    if len(parts) == 2 and parts[0] == INPUT_DIR and name.endswith(".slcio"):
        return "inputs", REQUIRED, name
    if len(parts) == 2 and parts[0] == INPUT_DIR and name.endswith(".h5"):
        return "inputs", OPTIONAL, name
    if len(parts) == 2 and parts[0] == OBSERVABLES_DIR and name.endswith(".npz") \
            and not _TEST_SAMPLE.search(name[:-len(".npz")]):
        return "measurements", REQUIRED, name
    if parts[0] in (SIM_DIR, REC_DIR) and len(parts) >= 4:
        group = SEP.join(parts[:3])
    elif len(parts) > 1:
        group = parts[0]
    else:
        group = "loose"  # files at the top of the data folder
    return "bulk", ARCHIVE, group + ".tar"


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
    presets and the reconstruction's arguments. {} without a log."""
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
    found, presets, extra = {}, [], []
    rec = any(os.path.basename(a) == "k4run" for a in args[:1])
    i = 0
    while i < len(args):
        arg = args[i]
        key, _, value = arg.partition("=")
        if key in ("--compactFile", "--steeringFile", "--ml-model") and not value \
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
        elif rec and i > 1 and key not in _MANAGED:
            extra.append(arg)
        i += 1
    if presets:
        found["presets"] = " ".join(presets)
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


def reconstruction_sources(relative):
    """(input, simulation log, reconstruction log), relative to the data
    folder, of a reconstructed file."""
    parts = relative.split("/")
    name = parts[-1]
    base = name[:-len(REC_SUFFIX)] if name.endswith(REC_SUFFIX) else os.path.splitext(name)[0]
    stem = _TEST_SAMPLE.sub("", _SEED.sub("", base))
    folder = "/".join(parts[1:-1])  # <scenario>/<simulator>
    return ("%s/%s.slcio" % (INPUT_DIR, stem), "%s/%s/%s.log" % (SIM_DIR, folder, base),
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
    """{input file name: {"scenarios", "options"}} of the configuration's
    scenarios; the options are the gun's, without where it writes."""
    from .layout import Layout
    from .scenarios import scenario_inputs
    found = {}
    for group in scenario_inputs(config, Layout(data_dir)):
        for entry in group["inputs"]:
            options, skip = [], False
            for arg in entry["command"][2:]:  # after python3 and the script
                if skip:
                    skip = False
                elif arg == "--save_dir":
                    skip = True
                else:
                    options.append(arg)
            item = found.setdefault(os.path.basename(entry["file"]), {
                "scenarios": [], "generator": os.path.basename(entry["command"][1]),
                "options": options})
            item["scenarios"].append(group["scenario"])
    return found


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


def build(data_dir, out, records=RECORDS, stage=False, copy=False, config=None, check=True):
    """Write OUT/<record>/manifest.json for each record (and, with stage, the
    files to upload beside it); returns {record: manifest}."""
    data_dir = os.path.abspath(data_dir)
    if not os.path.isdir(data_dir):
        raise ManifestError("no data folder %s" % data_dir)
    paths = scan(data_dir, leave_out=[out])
    by_record = dict((record, []) for record in RECORDS)
    for relative in paths:
        record, role, name = classify(relative)
        by_record[record].append((relative, role, name))

    common = {"format": FORMAT,
              "created": datetime.datetime.now(datetime.timezone.utc).strftime(
                  "%Y-%m-%dT%H:%M:%SZ"),
              "fasinate": repository_commit(),
              "configuration": configuration_snapshot(config) if config else None}
    warnings = []
    say("%d files in %s" % (len(paths), data_dir))

    # The inputs are hashed whichever records are made, for the measurements'
    # derived_from
    inputs = {}
    commands = input_commands(config, data_dir) if config else {}
    for relative, role, name in by_record["inputs"]:
        path = os.path.join(data_dir, relative)
        entry = {"name": name, "path": relative, "sha256": sha256(path),
                 "size": os.path.getsize(path), "role": role,
                 "kind": "input" if role == REQUIRED else "truth"}
        source = commands.get(name if role == REQUIRED else input_stem(name) + ".slcio")
        if source:
            entry.update(source)
        elif config and role == REQUIRED:
            warnings.append("%s isn't an input of any scenario in the configuration" % relative)
        inputs[relative] = entry
    for name in commands:
        if "%s/%s" % (INPUT_DIR, name) not in inputs:
            warnings.append("the configuration's input %s isn't in the data folder" % name)

    manifests = {}
    status = measurement_status(data_dir) if config and check \
        and "measurements" in records else {}
    for record in records:
        files, logs = [], []
        if record == "inputs":
            files = list(inputs.values())
        elif record == "measurements":
            for relative, role, name in by_record["measurements"]:
                files.append(_measurement(data_dir, relative, role, name, inputs, status,
                                          warnings, logs))
        else:
            groups = {}
            for relative, role, name in by_record["bulk"]:
                groups.setdefault(name, []).append(relative)
            for name in sorted(groups):
                files.append({"name": name, "role": ARCHIVE, "kind": "archive",
                              "sha256": None, "size": None,
                              "members": [{"path": p} for p in groups[name]]})

        folder = os.path.join(out, record)
        os.makedirs(folder, exist_ok=True)
        names = [entry["name"] for entry in files]
        doubled = sorted({n for n in names if names.count(n) > 1})
        if doubled:
            raise ManifestError("the %s record would have two files named %s; Zenodo's records "
                                "are flat, so rename one" % (record, ", ".join(doubled)))
        for entry in files:
            if record == "bulk":
                _archive(entry, data_dir, folder, stage)
            elif stage:
                _place(os.path.join(data_dir, entry["path"]), os.path.join(folder, entry["name"]),
                       copy)
        manifest = dict(common, record=record, files=files)
        if record == "measurements":
            # What every simulator should share (not the steering or presets)
            manifest["stack"] = dict((field, value) for field, value in summarise(logs).items()
                                     if field in SHARED)
            for field, value in manifest["stack"].items():
                if isinstance(value, dict):
                    warnings.append("the measurements were made with different %s: %s" % (
                        field, ", ".join("%s (%d logs)" % item for item in value.items())))
        total = sum(entry["size"] or 0 for entry in files)
        if len(files) + 1 > MAX_FILES or total > MAX_BYTES:
            warnings.append("the %s record has %d files (with the manifest) and %.1f GB; a "
                            "Zenodo record takes %d files and %d GB unless you ask for more"
                            % (record, len(files) + 1, total / 1e9, MAX_FILES, MAX_BYTES // 10**9))
        with open(os.path.join(folder, "manifest.json"), "w") as stream:
            json.dump(manifest, stream, indent=1, sort_keys=True)
            stream.write("\n")
        manifests[record] = manifest
        say("%s: %d files, %.3g GB%s -> %s" % (
            record, len(files), total / 1e9,
            "" if stage or record != "bulk" else " (members; archives not made without --stage)",
            os.path.join(folder, "manifest.json")))
    for warning in warnings:
        say("warning: " + warning)
    return manifests


def _measurement(data_dir, relative, role, name, inputs, status, warnings, logs):
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
            source, sim_log, rec_log = reconstruction_sources(reconstruction["path"])
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
        digest, size, hashes = pack(os.path.join(folder, entry["name"]), members, data_dir)
        entry.update(sha256=digest, size=size)
    else:
        hashes = dict((p, (sha256(os.path.join(data_dir, p)),
                           os.path.getsize(os.path.join(data_dir, p)))) for p in members)
    for member in entry["members"]:
        member["sha256"], member["size"] = hashes[member["path"]]


def dry_run(data_dir, leave_out=()):
    """What would go where, without hashing or writing anything."""
    totals = {}
    for relative in scan(data_dir, leave_out):
        record, role, name = classify(relative)
        files = totals.setdefault(record, {}).setdefault((name, role), [0, 0])
        files[0] += 1
        files[1] += os.path.getsize(os.path.join(data_dir, relative))
    for record in RECORDS:
        files = totals.get(record, {})
        print("%s: %d files, %.3g GB" % (record, len(files),
                                         sum(v[1] for v in files.values()) / 1e9))
        for (name, role), (count, size) in sorted(files.items()):
            print("  %-9s %-60s %10.3g MB%s" % (role, name, size / 1e6,
                                                "  (%d files)" % count if count > 1 else ""))


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="ddml-bench manifest",
        description="Write the manifests of a data folder's records for publishing it "
                    "(inputs, measurements, bulk); see ddml_bench/manifest.py.")
    parser.add_argument("--data-dir", help="the data folder (default: paths.data_dir)")
    parser.add_argument("--out", help="where the records go, a folder each with its "
                                      "manifest.json (required unless --dry-run)")
    parser.add_argument("--records", default=",".join(RECORDS),
                        help="only these records, separated by commas (default: all)")
    parser.add_argument("--stage", action="store_true",
                        help="also put the files to upload in each record's folder: links to "
                             "the inputs and measurements, and the bulk record's tar archives")
    parser.add_argument("--copy", action="store_true",
                        help="with --stage, copy the inputs and measurements instead of linking")
    parser.add_argument("--no-check", action="store_true",
                        help="don't check the measurements against the configuration")
    parser.add_argument("-c", "--config", action="append", default=[], metavar="FILE",
                        help="merge FILE over the configuration (repeatable)")
    parser.add_argument("--dry-run", action="store_true",
                        help="show what goes into which record, then stop")
    args = parser.parse_args(argv)
    from .cli import use_config_files
    use_config_files(args.config)
    records = [r.strip() for r in args.records.split(",") if r.strip()]
    for record in records:
        if record not in RECORDS:
            parser.error("there's no record '%s'; the records are %s" % (record,
                                                                        ", ".join(RECORDS)))
    config = configuration()
    data_dir = args.data_dir or (config or {}).get("paths", {}).get("data_dir")
    if not data_dir:
        parser.error("give --data-dir, as the configuration can't be read")
    if args.dry_run:
        dry_run(data_dir, [args.out] if args.out else [])
        return 0
    if not args.out:
        parser.error("give --out, the folder the records go to")
    try:
        build(data_dir, os.path.abspath(args.out), records, args.stage, args.copy, config,
              check=not args.no_check)
    except ManifestError as error:
        say(str(error))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
