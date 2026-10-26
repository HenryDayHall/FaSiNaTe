"""Fetching published data (Zenodo records with a manifest.json, as
ddml_bench/manifest.py writes them) instead of making it.

Before a run works out what is done, prepare() reads the manifests of the
records in the configuration's `published` section and fetches into the data
folder each published file that a task of the run would make, where that file
isn't there yet and the published one is what the task would make:

  an input (the generate stage)  when its gun options are the configuration's,
                                 as the inputs' names don't say everything
                                 (e.g. the number of events)
  a simulator's measurements     when they are what the observables stage
  (the observables stage)        would make: the same observables with the same
                                 settings, measured on the same scenarios'
                                 reconstructions, of a simulator of this
                                 configuration (a model with the same presets,
                                 detector and, where the manifest has them,
                                 weights), and every input they were measured
                                 from is the published one: fetched now, or
                                 here already and the same by sha256

Nothing that is there is replaced, and nothing of a stage that is forced
(--force) is fetched. Each file is downloaded to a hidden file beside where
it goes, checked against the sha256 and size in the manifest, and then moved
into place; the manifests themselves are checked against the md5 sums
configured for them (Zenodo shows each file's md5). So, as published
measurements are up to date by their own record (ddml_bench/measure.py),
nothing is simulated or reconstructed for them, and adding a model to a fresh
data folder runs only that model's tasks and the plots.

It runs where the plan is made, before the environment is set up (on a slurm
cluster, the login node, so the jobs need no access to the internet), and
needs only the standard library. With --dry-run nothing is fetched, but the
plan says what would be (to_fetch), so that what each stage would do is
reported as it would be.
"""

import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

from .layout import Layout

USER_AGENT = "FaSiNaTe-ddml-bench (+https://github.com/HenryDayHall/FaSiNaTe)"
FORMATS = ("fasinate-manifest/1", "fasinate-manifest/2")
BLOCK = 1 << 20
TRIES = 4


class FetchError(Exception):
    pass


def say(message):
    print("ddml-bench: " + message)
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Downloading


def _open(url):
    """The response for url, trying again after a while when the server asks
    for one (HTTP 429, 5xx) or the connection fails."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(1, TRIES + 1):
        try:
            return urllib.request.urlopen(request, timeout=120)
        except urllib.error.HTTPError as error:
            if error.code != 429 and error.code < 500 or attempt == TRIES:
                raise FetchError("%s: HTTP %d %s" % (url, error.code, error.reason))
            wait = error.headers.get("Retry-After") if error.headers else None
            delay = int(wait) if wait and wait.isdigit() else 5 * attempt
        except (urllib.error.URLError, OSError) as error:
            if attempt == TRIES or url.startswith("file:"):
                raise FetchError("%s: %s" % (url, getattr(error, "reason", error)))
            delay = 5 * attempt
        say("  %s didn't answer, trying again in %d s" % (url, delay))
        time.sleep(delay)


def download(url, target, sha256=None, size=None, md5=None):
    """Download url to target, through a hidden file beside it, checking its
    sha256, md5 and size where given; returns its size."""
    folder, name = os.path.split(target)
    os.makedirs(folder, exist_ok=True)
    partial = os.path.join(folder, ".%s.%d.fetching" % (name, os.getpid()))
    digests = {"sha256": hashlib.sha256(), "md5": _hash("md5")}
    got = 0
    try:
        response = _open(url)
        with response, open(partial, "wb") as stream:
            for block in iter(lambda: response.read(BLOCK), b""):
                stream.write(block)
                got += len(block)
                for digest in digests.values():
                    digest.update(block)
        for kind, wanted in (("sha256", sha256), ("md5", md5)):
            if wanted and digests[kind].hexdigest() != wanted:
                raise FetchError("%s isn't the published file: its %s is %s, not %s"
                                 % (url, kind, digests[kind].hexdigest(), wanted))
        if size is not None and got != size:
            raise FetchError("%s isn't the published file: it has %d bytes, not %d"
                             % (url, got, size))
        os.replace(partial, target)
    finally:
        if os.path.exists(partial):
            os.remove(partial)
    return got


def file_url(settings, record, name):
    return settings["url"].format(record=record, name=urllib.parse.quote(name))


def read_manifest(settings, entry):
    """The manifest of a record (from the cache, once fetched and checked)."""
    record = str(entry["zenodo"])
    cached = os.path.join(settings["cache_dir"], "zenodo-" + record, "manifest.json")
    pinned = [(kind, entry.get("manifest_" + kind)) for kind in ("md5", "sha256")]
    if os.path.isfile(cached) and any(value and _digest(cached, kind) != value
                                      for kind, value in pinned):
        os.remove(cached)  # not the manifest configured: fetched again, and checked
    if not os.path.isfile(cached):
        download(file_url(settings, record, "manifest.json"), cached,
                 md5=entry.get("manifest_md5"), sha256=entry.get("manifest_sha256"))
    try:
        with open(cached) as stream:
            manifest = json.load(stream)
    except ValueError as error:
        os.remove(cached)
        raise FetchError("the manifest of Zenodo record %s isn't JSON: %s" % (record, error))
    if manifest.get("format") not in FORMATS:
        raise FetchError("the manifest of Zenodo record %s is in the format %r, which this "
                         "FaSiNaTe doesn't know (%s); update FaSiNaTe"
                         % (record, manifest.get("format"), ", ".join(FORMATS)))
    return manifest


def _hash(kind):
    """A hash, which md5 is too as a checksum (not for security), as FIPS
    systems only allow that way."""
    try:
        return hashlib.new(kind, usedforsecurity=False)
    except TypeError:  # Python before 3.9
        return hashlib.new(kind)


def _digest(path, kind):
    digest = _hash(kind)
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_of(path):
    return _digest(path, "sha256")


# ---------------------------------------------------------------------------
# Is the published file what the task would make?


def gun_options(args):
    """{option: [values]} of a gun's arguments, without where it writes, so
    that their order doesn't matter."""
    options, key = {}, None
    for arg in args:
        if arg.startswith("--"):
            key = arg
            options.setdefault(key, [])
        elif key is not None:
            options[key].append(arg)
    options.pop("--save_dir", None)
    return options


def input_matches(task, published):
    """(the same, why not) for a generate task and a published input."""
    command = task["params"]["command"]
    if not published.get("options"):
        return False, "the manifest doesn't say how it was made"
    if published.get("generator") != os.path.basename(command[1]):
        return False, "it was made by %s, not %s" % (published.get("generator"),
                                                     os.path.basename(command[1]))
    mine, theirs = gun_options(command[2:]), gun_options(published["options"])
    if mine != theirs:
        differ = sorted(k for k in set(mine) | set(theirs) if mine.get(k) != theirs.get(k))
        return False, "it was made with other options (%s: %s here, %s there)" % (
            differ[0], " ".join(mine.get(differ[0], [])) or "none",
            " ".join(theirs.get(differ[0], [])) or "none")
    return True, ""


def wanted_settings(observables):
    """[(observable, scenario, profile or None, settings JSON)] that the
    configuration asks for, as measure.wanted() has them, without numpy."""
    from . import measure
    result = []
    for name, settings in observables.items():
        settings_json = json.dumps(measure.observable_settings(name, settings), sort_keys=True)
        profiles = ([None] if name in measure.PER_EVENT
                    else [profile for profile, _ in measure.KINDS[name][0](name, settings)])
        for profile in profiles:
            for scenario in settings["scenarios"]:
                result.append((name, scenario, profile, settings_json))
    return result


def measurements_match(task, published, plan):
    """(the same, why not) for an observables task and published measurements,
    by what the manifest says they hold."""
    params = task["params"]
    if published.get("simulator") != task["simulator"]:
        return False, "they are the measurements of %s" % published.get("simulator")
    held = published.get("observables") or {}
    for name, scenario, profile, settings_json in wanted_settings(params["observables"]):
        entry = held.get(name)
        if entry is None or scenario not in entry.get("scenarios", []):
            return False, "they don't have %s on %s" % (name, scenario)
        if json.dumps(entry.get("settings"), sort_keys=True) != settings_json:
            return False, "the settings of %s differ from this configuration's" % name
        if profile is not None and profile not in entry.get("profiles", []):
            return False, "they don't have the %s profile of %s" % (profile, name)
    layout = Layout(params["data_dir"])
    recorded = published.get("reconstructions") or {}
    for scenario, paths in params["files"].items():
        theirs = [r.get("path") for r in recorded.get(scenario, [])]
        if theirs != [layout.relative(p) for p in paths]:
            return False, "they were measured on other reconstructions of %s" % scenario
    stack = published.get("stack") or {}
    model = next((m for m in plan["models"] if m["name"] == task["simulator"]), None)
    simulator = model or plan["reference"]
    compact = stack.get("compact")
    if isinstance(compact, str) and compact != simulator["compact_file"]:
        return False, "they were simulated in %s, not %s" % (compact, simulator["compact_file"])
    if model is not None:
        presets = stack.get("presets")
        if isinstance(presets, str) and presets.split() != list(model["presets"]):
            return False, "the model ran with the presets %s, not %s" % (
                presets, " ".join(model["presets"]))
        described = published.get("model") or {}
        if described.get("weights_md5") and model.get("weights_md5") \
                and described["weights_md5"] != model["weights_md5"]:
            return False, "the model's weights differ (md5 %s, not %s)" % (
                described["weights_md5"], model["weights_md5"])
    return True, ""


# ---------------------------------------------------------------------------
# Choosing what to fetch, and fetching it


def choose(plan, manifests):
    """([(task, record, entry)] to fetch, [why each published file at a task's
    output isn't fetched]) for the plan's tasks, by the manifests ({record:
    manifest})."""
    from .graph import Graph
    graph = Graph(plan)
    layout = graph.layout
    published = {}  # path in the data folder: (record, entry)
    for record, manifest in manifests.items():
        for entry in manifest.get("files", []):
            if "path" in entry and entry.get("role") == "required":
                published.setdefault(entry["path"], (record, entry))

    chosen, notes = [], []
    fetched_inputs = {}  # path: sha256 of the inputs to fetch
    forced = set(plan.get("force") or [])
    for stage in ("generate", "observables"):
        for task in graph.list:
            if task["stage"] != stage or stage in forced:
                continue
            relative = layout.relative(task["output"])
            if relative not in published or os.path.exists(task["output"]):
                continue
            record, entry = published[relative]
            if stage == "generate":
                same, why = input_matches(task, entry)
                if same:
                    fetched_inputs[relative] = entry["sha256"]
            else:
                same, why = measurements_match(task, entry, plan)
                if same:
                    same, why = _inputs_are_published(layout, entry, fetched_inputs)
            if same:
                chosen.append((task, record, entry))
            else:
                notes.append("the published %s (Zenodo record %s) isn't used, as %s"
                             % (entry["name"], record, why))
    return chosen, notes


def _inputs_are_published(layout, entry, fetched_inputs):
    """(all the published, why not): every input the measurements were made
    from is fetched now, or is here and the same."""
    for source in entry.get("derived_from") or []:
        path = source.get("path")
        if path in fetched_inputs:
            continue
        local = os.path.join(layout.data_dir, path)
        if not os.path.isfile(local):
            return False, "its input %s isn't here and won't be fetched" % os.path.basename(path)
        if sha256_of(local) != source.get("sha256"):
            return False, ("your %s differs from the one it was measured from (move yours "
                           "away to fetch the published one)" % os.path.basename(path))
    return True, ""


def stack_note(manifests, pins):
    """A note when the configuration doesn't pin the software the published
    measurements were made with, or pins another."""
    stack = {}
    for manifest in manifests.values():
        for field, value in (manifest.get("stack") or {}).items():
            stack.setdefault(field, value)
    differ = []
    for field, setting in (("key4hep", "key4hep.release"), ("ddml", "ddml.commit"),
                           ("ildconfig", "detector_configs.ildconfig.commit")):
        theirs, mine = stack.get(field), pins.get(field)
        if isinstance(theirs, str) and theirs != "unknown" and str(mine) != theirs:
            differ.append('%s: "%s" (here %s)' % (setting, theirs, mine or "not pinned"))
    if differ:
        return ("so that a model simulated here is compared like with like, pin the software "
                "the published measurements were made with: " + "; ".join(differ))
    return None


def prepare(plan, dry_run=False):
    """Fetch what the plan's published records have of what its tasks would
    make (see the module docstring); with dry_run, note it in the plan
    instead (to_fetch). Raises FetchError if a record can't be read or a
    file fetched."""
    settings = plan.get("published") or {}
    if not settings.get("enabled") or not settings.get("records"):
        return []
    manifests = {}
    for entry in settings["records"]:
        record = str(entry["zenodo"])
        try:
            manifests[record] = read_manifest(settings, entry)
        except FetchError as error:
            raise FetchError("can't read the published data of Zenodo record %s: %s\n"
                             "(to make everything here instead, run with --no-fetch, or set "
                             "published.enabled: false)" % (record, error))
    chosen, notes = choose(plan, manifests)
    for note in notes:
        say(note)
    if chosen:
        note = stack_note(manifests, settings.get("pins") or {})
        if note and any(task["stage"] == "observables" for task, _, _ in chosen):
            say("note: " + note)
    if not chosen:
        return []
    total = sum(entry["size"] for _, _, entry in chosen)
    records = sorted({record for _, record, _ in chosen})
    if dry_run:
        plan["to_fetch"] = dict((task["output"], "Zenodo record %s" % record)
                                for task, record, _ in chosen)
        say("would fetch %d published file(s), %.3g MB, from Zenodo record %s"
            % (len(chosen), total / 1e6, ", ".join(records)))
        return chosen
    say("fetching %d published file(s), %.3g MB, from Zenodo record %s, instead of making them"
        % (len(chosen), total / 1e6, ", ".join(records)))
    for number, (task, record, entry) in enumerate(chosen, 1):
        say("  [%d/%d] %s" % (number, len(chosen), entry["name"]))
        download(file_url(settings, record, entry["name"]), task["output"],
                 sha256=entry["sha256"], size=entry.get("size"))
    return chosen
