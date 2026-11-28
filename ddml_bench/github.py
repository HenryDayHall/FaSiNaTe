"""The low-cost mode on GitHub's runners (.github/workflows/low-cost.yml): models
tested on the first events of each input against the published full samples
of Geant4, its second sample and the other models, as ./low_cost.sh tests
them (ddml_bench/low_cost.py), but split into the jobs of a workflow, so that
a fork of FaSiNaTe with a model added to models/ has it simulated,
reconstructed, measured and drawn on GitHub's free runners. docs/github.md
says how to use it; configs/github.yaml is the configuration it runs with, and
the github section of configs/default.yaml has its settings.

  ddml-bench github keys                the keys of the workflow's caches
  ddml-bench github plan [options]      choose the models, split their sample
                                        into chains, pack them into shards
  ddml-bench github check               build the software, if needed, and check
                                        that it can run the chains
  ddml-bench github preflight           check the models with chains to run
                                        before any of them is simulated
  ddml-bench github shard INDEX         run a shard's chains
  ddml-bench github report [--out DIR]  merge, draw and summarise

each with --dir DIR, the folder of the run (default: low_cost_run in the
repository). check, preflight and shard carry on in the environment from
setup.sh (scripts/run_in_environment.sh) with `ddml-bench github tasks-check
FILE`, `preflight-tasks FILE`, `chains FILE` and, for each chain, `chain FILE
ID`.

A run, job by job:

plan    On a runner, with Python, PyYAML and numpy (plan_run()). Checks that
        the software is pinned to what the published measurements were made
        with, fetches what is published of the full benchmark, as a low-cost
        run does, and chooses the models to test: those named (--models), or
        else every model in models/ whose full measurements aren't published
        (nor here). A model's sample is the same wherever it is made from the
        same things, which its fingerprint sums up (fingerprint()): the model
        (its description, without what only the plots use, and its own Python
        code), the software, the events of the sample, the observables and
        the inputs. The report keeps each sample it makes as the artifact
        low-cost-sample-<model>-<fingerprint> (for 90 days), and a later run
        of a model with the same fingerprint downloads it with GitHub's gh
        (find_sample()) rather than make it again, unless --fresh. The rest of
        the sample is split into chains, each a simulation, its reconstruction
        and the measurement of that one reconstruction, whose time is
        estimated as low_cost.estimate() estimates it, divided by the runner's
        speed (github.speed). The chains are packed into as few shards as keep
        each under github.shard_hours, with github.parallel chains at once on
        each (pack()); a run whose shards would take more than
        github.budget_hours of runners all together, or one of them more than
        github.deadline_hours, is refused. It writes the plan of the run,
        DIR/plan.json, and the job's outputs and summary.

check   In Key4hep's container with CVMFS (check()). scripts/run_in_environment.sh
        builds DDML, ILDConfig and the patches where the workflow's cache of
        the software hasn't them, then, in the environment from setup.sh, the
        stages check that it can run every chain.

preflight
        In the container, after check (preflight()). Each model with chains
        to run has its preflight (ddml_bench/preflight.py), as the pipeline
        runs it before simulating it: its description, its weights, its own
        Python code (Black, which the workflow installs as the plan job's
        output black says, and importing it) and shots into the ECal barrel.
        What it found goes to <paths.data_dir>/preflight_output, which the
        workflow keeps as the artifact low-cost-preflight, and to the job's
        summary. The shards start only once every model passes.

shard   In the container, a job each (run_shard()). Works out the tasks of its
        chains here (from the plan job's commit and configuration they are the
        same, with this runner's paths), then runs them in the environment
        (run_chains()), github.parallel at once, the longest first: each
        simulates, reconstructs, and measures that reconstruction alone into
        DIR/parts/<chain>.npz (measure.compute_simulator()), then deletes the
        simulation and the reconstruction, keeping their logs in DIR/logs
        (run_chain()). It starts no chain after github.deadline_hours from the
        start of its job ($FASINATE_GITHUB_JOB_START, or its own start), nor
        while another runs and less than github.min_free_gb of disk is free,
        and stops those still running at github.stop_hours, so that what it
        has done is uploaded before GitHub stops the job. A chain whose part
        is there already, from an earlier attempt of the job, is done.

report  On a runner, with numpy and matplotlib (report()). Merges the parts
        of each model into observables/<model><sample>.npz
        (measure.merge_parts()), array for array what measuring them all
        together makes, so it is up to date by its record; takes the samples
        found by the plan; and draws them against the full samples as the
        low-cost mode does, without the Pareto fronts (ddml_bench/plot.py
        --sample --no-pareto), into DIR/results with the measurements and the
        run's record, run.json. The job's summary (summary()) has each metric
        of each model with its statistical floor at the sample's size, its
        value projected to the full sample's, and Geant4's second sample's
        and the best published model's values; and what the chains took
        against their estimates, which measures the runners' speed. A model
        with chains missing isn't drawn, and the report then fails, saying
        which, after drawing the rest. Each sample made is put in DIR/keep/
        <artifact>, for the workflow to keep.

Everything but the chains runs where the plan is made, with the standard
library and PyYAML (and numpy for plan and report).
"""

import csv
import hashlib
import json
import math
import os
import shutil
import signal
import socket
import subprocess
import sys
import time

from . import process
from .config import REPO_DIR, ConfigError, ddml_build_stamp, load_config, patches_stamp
from .graph import make_plan
from .layout import GEANT4, REC_SUFFIX_LCIO_ONLY, SECOND_GEANT4, Layout, sample_suffix

FORMAT = "fasinate-github-plan/1"
SAMPLE_FORMAT = "fasinate-github-sample/1"
DEFAULT_DIR = os.path.join(REPO_DIR, "low_cost_run")
PLAN = "plan.json"
PARTS = "parts"
LOGS = "logs"
SAMPLES = "samples"
KEEP = "keep"
RESULTS = "results"
SAMPLE_ARTIFACT = "low-cost-sample-%s-%s"
SAMPLE_RECORD = "sample.json"
JOB_START = "FASINATE_GITHUB_JOB_START"
IMAGE = "FASINATE_GITHUB_IMAGE"
ENTRY = os.path.join(REPO_DIR, "scripts", "ddml-bench")
# The scripts that build the software, whose contents are in its cache's key
BUILD_SCRIPTS = ("scripts/build_ddml.sh", "scripts/build_patches.sh",
                 "scripts/fetch_detector_configs.sh", "scripts/fetch_ddml_models.py",
                 "scripts/environment.sh")
# The settings of the github section: (name, whole number, 0 allowed)
SETTINGS = (("parallel", True, False), ("speed", False, False), ("shard_hours", False, False),
            ("max_shards", True, False), ("deadline_hours", False, False),
            ("stop_hours", False, False), ("setup_hours", False, True),
            ("budget_hours", False, False), ("min_free_gb", False, True))
# How often a shard looks at its chains, seconds
POLL = 2.0


class GitHubError(Exception):
    pass


def say(message):
    print("ddml-bench: " + message)
    sys.stdout.flush()


def _read_json(path):
    try:
        with open(path) as stream:
            return json.load(stream)
    except (OSError, ValueError) as error:
        raise GitHubError("can't read %s: %s" % (path, error))


def _write_json(path, data):
    """Write data to path as JSON, whole or not at all."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    partial = "%s.%d.partial" % (path, os.getpid())
    with open(partial, "w") as stream:
        json.dump(data, stream, indent=1, sort_keys=True)
        stream.write("\n")
    os.replace(partial, path)


def _digest(value, length=16):
    """A short sha256 of a JSON-able value, the same for the same value."""
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def set_outputs(**values):
    """The step's outputs, for the workflow ($GITHUB_OUTPUT), each a line
    name=value (lists and the like as JSON); printed too."""
    lines = []
    for name, value in values.items():
        text = value if isinstance(value, str) else json.dumps(value)
        lines.append("%s=%s" % (name.replace("_", "-"), text))
    for line in lines:
        print(line)
    path = os.environ.get("GITHUB_OUTPUT")
    if path:
        with open(path, "a") as stream:
            stream.write("".join(line + "\n" for line in lines))


def add_summary(text):
    """Markdown for the job's summary on the run's page ($GITHUB_STEP_SUMMARY)."""
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a") as stream:
            stream.write(text.rstrip("\n") + "\n\n")


def black_requirement(config):
    """What pip installs for the preflight's check against Black: the major
    version of preflight.black, or any."""
    from .preflight import PreflightError, settings
    try:
        black = settings(config)["black"]
    except PreflightError as error:
        raise GitHubError(str(error))
    return "black==%s.*" % black if black else "black"


def github_settings(config):
    """The github section, checked."""
    section = config.get("github") or {}
    found = {}
    for key, whole, zero in SETTINGS:
        value = section.get(key)
        if isinstance(value, bool) or not isinstance(value, int if whole else (int, float)) \
                or not math.isfinite(value) or value < 0 or (value == 0 and not zero):
            raise ConfigError("github.%s should be %s %s, not %r"
                              % (key, "a whole number" if whole else "a number",
                                 "of 0 or more" if zero else "above 0", value))
        found[key] = value
    if found["stop_hours"] < found["deadline_hours"]:
        raise ConfigError("github.stop_hours (%g) is when a shard stops its chains, so it should "
                          "be no earlier than github.deadline_hours (%g), when it starts the "
                          "last" % (found["stop_hours"], found["deadline_hours"]))
    return found


# ---------------------------------------------------------------------------
# The caches


def cache_keys(config, image=None):
    """{output: key} of the workflow's caches: published-key, of what the plan
    job fetches (the published records, and the settings that decide which
    of their files are what a run would make: the scenarios, the observables
    and geant4), and software-key, of what the check job builds (DDML and the
    patches by their build stamps, ILDConfig by its pin, the scripts that
    build them, and the image of the container, image or $FASINATE_GITHUB_IMAGE)."""
    from .fetch import sha256_of
    published = config.get("published") or {}
    ildconfig = config["detector_configs"]["ildconfig"]
    scripts = dict((name, sha256_of(os.path.join(REPO_DIR, name))) for name in BUILD_SCRIPTS)
    return {
        "published-key": "fasinate-published-" + _digest(
            [published.get("records"), published.get("url"), config.get("scenarios"),
             config.get("observables"), config.get("geant4")]),
        "software-key": "fasinate-software-" + _digest(
            [ddml_build_stamp(config), patches_stamp(config), ildconfig.get("repository"),
             ildconfig.get("branch"), ildconfig.get("commit"), scripts,
             os.environ.get(IMAGE, "") if image is None else image]),
    }


# ---------------------------------------------------------------------------
# The plan


def untested_models(full):
    """[(model, why)] of the models of the full benchmark's plan whose full
    measurements aren't here and up to date (fetched, or measured here)."""
    from .graph import Graph
    from .stages import BY_NAME
    stage = BY_NAME["observables"]
    names = [model["name"] for model in full["models"]]
    found = []
    for task in Graph(full).list:
        if task["stage"] == "observables" and task["simulator"] in names:
            done, why = stage.current(task)
            if not done:
                found.append((task["simulator"], why))
    return found


def chain_hours(plan, config, speed):
    """{scenario: hours} that a chain of each scenario of the plan is
    estimated to take on a runner: low_cost.estimate()'s, for one simulation,
    divided by the runner's speed."""
    from .low_cost import estimate
    hours = {}
    for scenario, simulations, _, _, simulation, _, reconstruction, total in estimate(plan, config):
        if total is None or not simulations:
            raise GitHubError(
                "the time of a chain of %s can't be estimated (%s): set low_cost.estimate's "
                "simulation_s and reconstruction_s for it" % (
                    scenario, "no time per event of its simulation" if simulation is None
                    else "no time per event of its reconstruction"))
        hours[scenario] = total / simulations / speed
    return hours


def fingerprint(model, task, plan, config, layout, digests):
    """What a model's sample is made from, summed up in 16 hex digits: the model
    (its description, without what only the plots use, its own Python code and
    its detector), the software (DDML's and the patches' build stamps,
    ILDConfig's commit, the reconstruction's arguments), the sample's events, the
    observables and their settings, the reconstructions measured (named after
    their inputs, segments, events and seeds), their seeds, and the inputs'
    sha256 sums (digests: {input: sha256}). task is the model's observables
    task."""
    from .fetch import wanted_settings
    from .models import python_summary
    params = task["params"]
    ildconfig = config["detector_configs"]["ildconfig"]
    return _digest({
        "format": SAMPLE_FORMAT,
        "model": [model["name"], model.get("description_sha256"),
                  python_summary(model.get("python") or {}), model["compact_file"]],
        # (ILDConfig by its commit, from whichever clone; unpinned, by its branch)
        "software": [ddml_build_stamp(config), patches_stamp(config),
                     ildconfig.get("commit") or [ildconfig.get("repository"),
                                                 ildconfig.get("branch")],
                     plan["reconstruction"]["extra_args"]],
        "events": plan["events"],
        "observables": wanted_settings(params["observables"]),
        "reconstructions": dict((scenario, [layout.relative(path) for path in paths])
                                for scenario, paths in params["files"].items()),
        "seeds": params["seeds"],
        "inputs": digests,
    })


def find_sample(name, repository, folder):
    """Download the newest artifact of that name of the repository's workflow
    runs that hasn't expired into folder, with GitHub's gh (with the token in
    $GH_TOKEN); returns the id of its run, or None."""
    def gh(*args):
        try:
            done = subprocess.run(["gh"] + list(args), stdin=subprocess.DEVNULL,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  universal_newlines=True, timeout=600)
        except (OSError, subprocess.TimeoutExpired) as error:
            return None, str(error)
        if done.returncode != 0:
            last = done.stderr.strip().splitlines()
            return None, last[-1] if last else "gh exited with status %d" % done.returncode
        return done.stdout, None

    listed, error = gh("api", "-X", "GET", "repos/%s/actions/artifacts" % repository,
                       "-f", "name=" + name, "-f", "per_page=30")
    if listed is None:
        say("can't look for the artifact %s: %s" % (name, error))
        return None
    try:
        artifacts = json.loads(listed).get("artifacts") or []
    except (ValueError, AttributeError):
        artifacts = []
    artifacts = sorted((a for a in artifacts if isinstance(a, dict) and a.get("name") == name
                        and not a.get("expired") and (a.get("workflow_run") or {}).get("id")),
                       key=lambda a: a.get("created_at") or "", reverse=True)
    for artifact in artifacts:
        run = artifact["workflow_run"]["id"]
        shutil.rmtree(folder, ignore_errors=True)
        _, error = gh("run", "download", str(run), "--repo", repository, "--name", name,
                      "--dir", folder)
        if error is None:
            return run
        say("can't download the artifact %s of run %s: %s" % (name, run, error))
    return None


def reuse_sample(model, task, layout, print_, artifact, repository, work):
    """The sample of the model kept by an earlier run as artifact, if there is
    one and it is what this run would make (its record and fingerprint):
    moved to DIR/samples; returns {file, run} or None."""
    from . import measure
    folder = os.path.join(work, "incoming", artifact)
    run = find_sample(artifact, repository, folder)
    if run is None:
        return None
    name = os.path.basename(layout.observables(model))
    found = os.path.join(folder, name)
    record = os.path.join(folder, SAMPLE_RECORD)
    params = task["params"]
    try:
        kept = _read_json(record)
    except GitHubError:
        kept = {}
    why = None
    if kept.get("fingerprint") != print_ or not os.path.isfile(found):
        why = "it isn't the sample of the model's fingerprint %s" % print_
    else:
        current, reason = measure.is_up_to_date(found, measure.wanted(params["observables"]),
                                                params["files"], layout.relative)
        if not current:
            why = reason
    if why:
        say("the artifact %s of run %s isn't used, as %s" % (artifact, run, why))
        shutil.rmtree(folder, ignore_errors=True)
        return None
    target = os.path.join(work, SAMPLES, name)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    shutil.move(found, target)
    shutil.rmtree(folder, ignore_errors=True)
    return {"file": os.path.relpath(target, work), "run": run}


def pack(chains, settings):
    """Pack the chains ([{id, hours}]) into shards: [{chains: [id], hours}],
    each shard's hours its estimate, with settings' parallel chains at once.
    The longest chains go first, each into the place for a chain at a time (a
    slot) with the fewest hours so far, of as few shards as keep each under
    shard_hours, or under the longest chain's hours if that is longer (which
    no number of shards makes shorter), up to max_shards."""
    if not chains:
        return []
    parallel = settings["parallel"]
    order = sorted(chains, key=lambda chain: (-chain["hours"], chain["id"]))
    limit = max(settings["shard_hours"], order[0]["hours"])
    total = sum(chain["hours"] for chain in chains)
    count = max(1, min(settings["max_shards"], int(math.ceil(total / (parallel * limit)))))
    while True:
        slots = [[0.0, []] for _ in range(count * parallel)]
        for chain in order:
            slot = min(slots, key=lambda s: s[0])
            slot[0] += chain["hours"]
            slot[1].append(chain["id"])
        if max(load for load, _ in slots) <= limit + 1e-9 or count >= settings["max_shards"]:
            break
        count += 1
    shards = []
    for number in range(count):
        mine = slots[number * parallel:(number + 1) * parallel]
        ids = [chain_id for _, held in mine for chain_id in held]
        if ids:
            shards.append({"chains": sorted(ids),
                           "hours": round(max(load for load, _ in mine), 3)})
    return shards


def plan_run(work, models=(), events=None, fresh=False, any_stack=False):
    """Plan a run (see the module docstring): returns the plan written to
    DIR/plan.json."""
    from . import fetch, low_cost
    from .graph import Graph
    from .models import model_names
    config = load_config()
    settings = github_settings(config)
    names = model_names()
    for model in models:
        if model in (GEANT4, SECOND_GEANT4):
            raise GitHubError("the low-cost mode tests models against the full samples of "
                              "Geant4 and the rest, so %s can't be one of them" % model)
        if model not in names:
            raise GitHubError("there is no model '%s'; models/ has: %s"
                              % (model, ", ".join(names) or "nothing"))
    numbers = low_cost.configured_events(config, events)
    suffix = sample_suffix(numbers)

    # What is published of the full benchmark: its inputs, the full samples'
    # measurements and the timing table (for the estimate)
    full = make_plan()
    low_cost.check_stack(full["published"], any_stack)
    fetch.prepare(full, note_stack=False)
    found = low_cost.full_measurements(full)
    reference = next(item for item in found if item[0] == GEANT4)
    if reference[2] is None:
        raise GitHubError("there are no full measurements of Geant4 (%s) to compare with, here "
                          "or published as this configuration would make them (see above)"
                          % reference[1])
    compared = [name for name, _, state in found if state]

    # The models to test
    if models:
        chosen = [(model, "named") for model in dict.fromkeys(models)]
    else:
        chosen = untested_models(full)
    tested = [model for model, _ in chosen]
    record = {"format": FORMAT, "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "models": tested, "why": dict(chosen), "events": numbers, "suffix": suffix,
              "compared": [name for name in compared if name not in tested],
              "records": [str(r["zenodo"]) for r in full["published"].get("records") or []
                          if full["published"].get("enabled")],
              "stack": full["published"].get("pins"), "fingerprints": {}, "artifacts": {},
              "reused": {}, "chains": [], "shards": [], "settings": settings,
              "runner_hours": 0.0}
    os.makedirs(work, exist_ok=True)
    if not tested:
        say("every model in models/ has its full measurements here (published), so there is "
            "nothing to test; name the models to test anyway (--models)")
        _write_json(os.path.join(work, PLAN), record)
        return record

    plan = make_plan(simulators=tested, events=numbers, low_cost=True)
    plan["published"]["enabled"] = False
    graph = Graph(plan)
    if graph.left_out:
        raise GitHubError("; ".join(graph.left_out_reasons()))
    layout = graph.layout
    hours = chain_hours(plan, config, settings["speed"])
    repository = os.environ.get("GITHUB_REPOSITORY")
    if not fresh and not repository:
        say("not in a GitHub workflow ($GITHUB_REPOSITORY isn't set), so no sample kept by an "
            "earlier run is looked for")
    from .fetch import sha256_of
    digests = {}
    for model in tested:
        task = graph.by_key["observables:" + layout.observables(model)]
        inputs = sorted({t["inputs"][0] for t in graph.list
                         if t["stage"] == "models" and t["simulator"] == model})
        for path in inputs:
            if path not in digests:
                digests[path] = sha256_of(path) if os.path.isfile(path) else None
        description = next(m for m in plan["models"] if m["name"] == model)
        print_ = fingerprint(description, task, plan, config, layout,
                             dict((layout.relative(path), digests[path]) for path in inputs))
        artifact = SAMPLE_ARTIFACT % (model, print_)
        record["fingerprints"][model] = print_
        record["artifacts"][model] = artifact
        if not fresh and repository:
            reused = reuse_sample(model, task, layout, print_, artifact, repository, work)
            if reused:
                say("%s: its sample was made by run %s (the artifact %s), so it isn't made again"
                    % (model, reused["run"], artifact))
                record["reused"][model] = reused
                continue
        # Its chains: a simulation, its reconstruction and the measurement of it
        # alone, of every reconstruction its measurements need, in their order
        for key in task["needs"]:
            reconstruction = graph.by_key[key]
            simulation = graph.by_key[reconstruction["needs"][0]]
            record["chains"].append({
                "id": "%03d" % len(record["chains"]), "model": model,
                "scenario": reconstruction["scenario"], "name": reconstruction["name"],
                "seed": reconstruction["seed"],
                "simulation": layout.relative(simulation["output"]),
                "reconstruction": layout.relative(reconstruction["output"]),
                "hours": round(hours[reconstruction["scenario"]], 4)})

    record["shards"] = shards = pack(record["chains"], settings)
    record["runner_hours"] = round(sum(shard["hours"] + settings["setup_hours"]
                                       for shard in shards), 2)
    longest = max([shard["hours"] for shard in shards] or [0.0])
    _write_json(os.path.join(work, PLAN), record)
    for line in plan_lines(record):
        say(line)
    if longest > settings["deadline_hours"]:
        raise GitHubError(
            "a shard of this run would take about %.1f h, more than a shard has to start its "
            "chains (github.deadline_hours, %g h): test fewer events (low_cost.events, or "
            "-n), or allow more shards (github.max_shards)" % (longest,
                                                              settings["deadline_hours"]))
    if record["runner_hours"] > settings["budget_hours"]:
        raise GitHubError(
            "this run would take about %.1f runner-hours, more than github.budget_hours (%g): "
            "test fewer events (low_cost.events, or -n), or fewer models at once (--models)"
            % (record["runner_hours"], settings["budget_hours"]))
    return record


def plan_lines(record):
    """What a plan is, in lines of text."""
    lines = []
    for model in record["models"]:
        why = record["why"].get(model) or ""
        if model in record["reused"]:
            what = "kept from run %s" % record["reused"][model]["run"]
        else:
            what = "%d chain(s)" % sum(1 for c in record["chains"] if c["model"] == model)
        lines.append("%s (%s): %s" % (model, "named" if why == "named" else "not published: "
                                      + why, what))
    if record["chains"]:
        by_scenario = {}
        for chain in record["chains"]:
            count, hours = by_scenario.get(chain["scenario"], (0, 0.0))
            by_scenario[chain["scenario"]] = (count + 1, hours + chain["hours"])
        for scenario, (count, hours) in by_scenario.items():
            lines.append("  %-16s %3d chain(s) of %d events, %.2f h each, %.1f h in all"
                         % (scenario, count, record["events"][scenario], hours / count, hours))
        settings = record["settings"]
        lines.append("%d shard(s), %d chain(s) at once on each: %s h; about %.1f runner-hours "
                     "with their set-up (github.budget_hours: %g)" % (
                         len(record["shards"]), settings["parallel"],
                         ", ".join("%.1f" % shard["hours"] for shard in record["shards"]),
                         record["runner_hours"], settings["budget_hours"]))
    return lines


# ---------------------------------------------------------------------------
# The tasks of the chains, on a runner


def read_plan(work):
    record = _read_json(os.path.join(work, PLAN))
    if record.get("format") != FORMAT:
        raise GitHubError("%s isn't a plan of this FaSiNaTe (%s), so make it again"
                          % (os.path.join(work, PLAN), record.get("format")))
    return record


def chain_tasks(work, record, ids):
    """The chains of the plan's ids, each with its tasks here: tasks (its
    simulation and reconstruction), observables and simulator (of its model's
    measurements), data_dir, and part (where its measurement goes). Raises
    GitHubError if the plan made here hasn't them: another commit or
    configuration than the plan job's, or inputs missing."""
    from .graph import Graph
    by_id = dict((chain["id"], chain) for chain in record["chains"])
    unknown = [chain_id for chain_id in ids if chain_id not in by_id]
    if unknown:
        raise GitHubError("the plan has no chain %s" % unknown[0])
    models = list(dict.fromkeys(by_id[chain_id]["model"] for chain_id in ids))
    plan = make_plan(simulators=models, events=record["events"], low_cost=True)
    plan["published"]["enabled"] = False
    graph = Graph(plan)
    layout = graph.layout
    chains = []
    for chain_id in ids:
        chain = by_id[chain_id]
        reconstruction = graph.by_key.get("reconstruct:" + layout.absolute(chain["reconstruction"]))
        simulation = reconstruction and graph.by_key[reconstruction["needs"][0]]
        if not simulation or layout.relative(simulation["output"]) != chain["simulation"]:
            raise GitHubError(
                "the chain %s (%s) isn't a task of the plan made here, so this isn't the commit "
                "and configuration the run was planned with%s" % (
                    chain_id, chain["reconstruction"],
                    "" if not graph.left_out else ", or the inputs aren't all here: "
                    + "; ".join(graph.left_out_reasons())))
        measuring = graph.by_key["observables:" + layout.observables(chain["model"])]
        chains.append(dict(chain, tasks=[simulation, reconstruction],
                           observables=measuring["params"]["observables"],
                           simulator=measuring["params"]["simulator"],
                           data_dir=plan["data_dir"],
                           part=os.path.join(work, PARTS, chain_id + ".npz")))
    return chains


def part_files(work, chain_id, parts=None):
    """(the .npz, the .json) of a chain's measurement."""
    folder = parts or os.path.join(work, PARTS)
    return (os.path.join(folder, chain_id + ".npz"), os.path.join(folder, chain_id + ".json"))


def part_is_done(work, chain_id, parts=None):
    measured, record = part_files(work, chain_id, parts)
    if not (os.path.isfile(measured) and os.path.isfile(record)):
        return False
    try:
        return _read_json(record).get("status") == "done"
    except GitHubError:
        return False


def in_the_environment(args, work):
    """Run `ddml-bench github ARGS` in the environment from setup.sh (here, if
    this is it); returns its exit status."""
    from .environment import in_environment, run_in_environment
    if in_environment():
        return main(args)
    return run_in_environment(["github"] + list(args), binds=[work])


def check(work):
    """The check job: the environment can run every chain of the plan (built
    first, where it isn't, by scripts/run_in_environment.sh)."""
    record = read_plan(work)
    ids = [chain["id"] for chain in record["chains"]]
    if not ids:
        say("the plan has no chains to run, so there is nothing to check")
        return 0
    path = os.path.join(work, "check.json")
    _write_json(path, {"chains": chain_tasks(work, record, ids), "work": work})
    return in_the_environment(["tasks-check", path], work)


def check_tasks(path):
    """In the environment: what would stop the chains of path from running."""
    from .stages import STAGES
    from .stages.base import missing_modules
    chains = _read_json(path)["chains"]
    tasks = [task for chain in chains for task in chain["tasks"]]
    problems = []
    for stage in STAGES:
        mine = [task for task in tasks if task["stage"] == stage.name]
        if mine:
            problems += stage.check_environment(mine)
    problems += missing_modules(("numpy", "uproot", "awkward"), "measuring the observables")
    for problem in dict.fromkeys(problems):
        say(problem)
    if problems:
        return 1
    say("this environment can run the %d chain(s) of the plan (Key4hep %s, DDML %s)" % (
        len(chains), os.environ.get("DDML_KEY4HEP_RELEASE") or "?",
        os.environ.get("DDML_COMMIT") or "?"))
    return 0


def preflight(work):
    """The preflight job: each model with chains in the plan checked as the
    pipeline checks it before simulating it, in the environment; returns 0 if
    every one passes. What it found goes to the job's summary."""
    from . import preflight as checks
    from .graph import Graph
    record = read_plan(work)
    models = sorted({chain["model"] for chain in record["chains"]})
    if not models:
        say("the plan has no chains to run, so no model needs its preflight")
        return 0
    plan = make_plan(simulators=models, events=record["events"], low_cost=True)
    plan["published"]["enabled"] = False
    # (with every scenario: the preflight needs none of their inputs)
    graph = Graph(plan, keep_unavailable=True)
    context = {"github_run": os.environ.get("GITHUB_RUN_ID") or "?"}
    tasks = []
    for model in models:
        task = graph.by_key[graph.preflight[model]]
        task["params"]["context"] = context
        tasks.append(task)
    path = os.path.join(work, "preflight.json")
    _write_json(path, {"tasks": tasks})
    try:
        return in_the_environment(["preflight-tasks", path], work)
    finally:
        lines = ["## Preflight of the models"]
        for task in tasks:
            summary = os.path.join(task["params"]["folder"], checks.SUMMARY)
            if os.path.isfile(summary):
                with open(summary) as stream:
                    lines += ["", "```", stream.read().rstrip("\n"), "```"]
            else:
                lines += ["", "%s: no summary (see the job's log)" % task["simulator"]]
        lines += ["", "The plots and the full logs are in the artifact low-cost-preflight."]
        add_summary("\n".join(lines))


def preflight_tasks(path):
    """In the environment: run the preflight tasks of path; 0 if all pass."""
    from .executors import run_task
    from .stages import BY_NAME, StageError
    tasks = _read_json(path)["tasks"]
    problems = BY_NAME["preflight"].check_environment(tasks) if tasks else []
    for problem in dict.fromkeys(problems):
        say(problem)
    if problems:
        say("this environment can't run the preflight (see above)")
        return 1
    failed = []
    for task in tasks:
        task["redo"] = True
        try:
            run_task(task)
        except StageError as error:
            say("FAILED: %s" % error)
            failed.append(task["simulator"])
    if failed:
        say("the preflight of %s failed, so none of the chains is run"
            % " and ".join(failed))
        return 1
    say("every model with chains passed its preflight: %s"
        % ", ".join(task["simulator"] for task in tasks))
    return 0


def job_start():
    """When the job started: $FASINATE_GITHUB_JOB_START (seconds since 1970),
    which the workflow sets first, or now."""
    try:
        return float(os.environ[JOB_START])
    except (KeyError, ValueError):
        return time.time()


def run_shard(work, index):
    """A shard job: its chains that aren't done, in the environment."""
    record = read_plan(work)
    shards = record["shards"]
    if not 0 <= index < len(shards):
        raise GitHubError("the plan has %d shard(s), so there is no shard %d"
                          % (len(shards), index))
    ids = shards[index]["chains"]
    done = [chain_id for chain_id in ids if part_is_done(work, chain_id)]
    if done:
        say("%d of the shard's %d chain(s) are done already, by an earlier attempt: %s"
            % (len(done), len(ids), " ".join(done)))
    todo = [chain_id for chain_id in ids if chain_id not in done]
    if not todo:
        add_summary(shard_summary(record, index, work))
        return 0
    settings = github_settings(load_config())
    start = job_start()
    path = os.path.join(work, "shard%d.json" % index)
    _write_json(path, {"index": index, "chains": chain_tasks(work, record, todo),
                       "parallel": settings["parallel"],
                       "deadline": start + 3600.0 * settings["deadline_hours"],
                       "stop": start + 3600.0 * settings["stop_hours"],
                       "min_free_gb": settings["min_free_gb"], "work": work})
    try:
        return in_the_environment(["chains", path], work)
    finally:
        add_summary(shard_summary(record, index, work))


def free_gb(path):
    try:
        return shutil.disk_usage(path).free / 1e9
    except OSError:
        return float("inf")


def _hours(seconds):
    return "%dh%02dm" % (seconds // 3600, seconds % 3600 // 60)


def _start(path, chain, work):
    """Start a chain, in a process group of its own, its output to its log."""
    with open(os.path.join(work, LOGS, chain["id"] + ".log"), "w") as log:
        return subprocess.Popen([sys.executable, ENTRY, "github", "chain", path, chain["id"]],
                                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)


def _stop(child, wait=60):
    """Stop a chain and what it runs (SIGTERM, then SIGKILL)."""
    for signum in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(child.pid, signum)
        except OSError:
            pass
        try:
            child.wait(timeout=wait)
            return
        except subprocess.TimeoutExpired:
            continue


def run_chains(path, poll=POLL):
    """In the environment: run the chains of path (a shard's), parallel at
    once, the longest first, until the deadline (see the module docstring);
    returns 0 if every one is done."""
    shard = _read_json(path)
    work = shard["work"]
    for folder in (PARTS, LOGS):
        os.makedirs(os.path.join(work, folder), exist_ok=True)
    queue = sorted(shard["chains"], key=lambda chain: (-chain["hours"], chain["id"]))
    total = len(queue)
    say("shard %s: %d chain(s), %d at once, the longest first; none started after %s, and "
        "those running stopped at %s" % (
            shard["index"], total, shard["parallel"],
            time.strftime("%H:%M", time.localtime(shard["deadline"])),
            time.strftime("%H:%M", time.localtime(shard["stop"]))))
    running = {}  # id: (process, started, chain)
    results = {}  # id: (what became of it, seconds)
    waiting_for_disk = False
    try:
        with process.interruptible():
            while queue or running:
                now = time.time()
                for chain_id in list(running):
                    child, started, chain = running[chain_id]
                    status = child.poll()
                    if status is None:
                        continue
                    del running[chain_id]
                    results[chain_id] = ("done" if status == 0 else "FAILED (exit status %d)"
                                         % status, now - started)
                    say("chain %s %s after %s (%d of %d)" % (
                        chain_id, results[chain_id][0], _hours(now - started), len(results),
                        total))
                if running and now >= shard["stop"]:
                    say("github.stop_hours is reached, so the chains still running are stopped")
                    for chain_id, (child, started, _) in running.items():
                        _stop(child)
                        results[chain_id] = ("STOPPED at github.stop_hours", now - started)
                    running = {}
                    break
                while queue and len(running) < shard["parallel"] and now < shard["deadline"]:
                    free = free_gb(queue[0]["data_dir"])
                    if running and free < shard["min_free_gb"]:
                        if not waiting_for_disk:
                            say("%.1f GB of disk is free, less than github.min_free_gb, so the "
                                "next chain waits for one to finish" % free)
                        waiting_for_disk = True
                        break
                    waiting_for_disk = False
                    chain = queue.pop(0)
                    running[chain["id"]] = (_start(path, chain, work), time.time(), chain)
                    say("chain %s started: %s %s/%s, about %.2f h (%.0f GB of disk free)" % (
                        chain["id"], chain["model"], chain["scenario"], chain["name"],
                        chain["hours"], free))
                if not running:
                    break
                time.sleep(poll)
    except process.Interrupted:
        for child, _, _ in running.values():
            _stop(child, wait=20)
        raise
    for chain in queue:
        results[chain["id"]] = ("NOT STARTED, past github.deadline_hours", 0.0)

    failed = [chain_id for chain_id, (what, _) in results.items() if what != "done"]
    say("shard %s: %d of %d chain(s) done" % (shard["index"], total - len(failed), total))
    for chain in shard["chains"]:
        what, seconds = results.get(chain["id"], ("?", 0.0))
        say("  %s  %-40s %s %s/%s" % (chain["id"], what, _hours(seconds), chain["scenario"],
                                      chain["name"]))
    for chain_id in failed:
        log = os.path.join(work, LOGS, chain_id + ".log")
        if os.path.isfile(log):
            say("the end of chain %s's log, %s:\n%s" % (chain_id, log,
                                                         process.log_tail(log, 30)))
    return 1 if failed else 0


def run_chain(path, chain_id, keep=False):
    """In the environment: one chain of path (a shard's), see the module
    docstring. With keep, the simulation and reconstruction are kept."""
    from . import measure
    from .executors import run_task
    shard = _read_json(path)
    chain = next((c for c in shard["chains"] if c["id"] == chain_id), None)
    if chain is None:
        raise GitHubError("%s has no chain %s" % (path, chain_id))
    simulation, reconstruction = chain["tasks"]
    say("chain %s: %s's simulation of %s/%s, its reconstruction and its measurement" % (
        chain_id, chain["model"], chain["scenario"], chain["name"]))
    seconds = {}
    scenario = chain["scenario"]
    measured, record = part_files(shard["work"], chain_id)
    base = reconstruction["params"]["base"]
    try:
        # (stopped, by the shard or GitHub, it still cleans up below)
        with process.interruptible():
            for name, task in (("simulate", simulation), ("reconstruct", reconstruction)):
                start = time.time()
                run_task(task, check=True)
                seconds[name] = round(time.time() - start, 1)
            start = time.time()
            measure.compute_simulator(chain["observables"], chain["simulator"],
                                      {scenario: [reconstruction["output"]]},
                                      measure.wanted(chain["observables"]), measured,
                                      Layout(chain["data_dir"]).relative,
                                      seeds={scenario: [chain["seed"]]})
            seconds["measure"] = round(time.time() - start, 1)
    finally:
        # The logs go with the measurement, whatever became of it; the
        # simulation and reconstruction, which are large, are deleted
        logs = os.path.join(shard["work"], LOGS)
        for log, kind in ((simulation["params"]["log"], "simulation"),
                          (base + ".log", "reconstruction")):
            if os.path.isfile(log):
                shutil.copyfile(log, os.path.join(logs, "%s.%s.log" % (chain_id, kind)))
        if not keep:
            process.remove(simulation["output"], *[base + ending for ending in (
                reconstruction["params"]["done_suffix"], REC_SUFFIX_LCIO_ONLY, "_AIDA.root",
                "_PfoAnalysis.root")])
    import numpy as np
    with np.load(measured, allow_pickle=False) as data:
        events = int(np.sum(data["%s__n_events" % scenario]))
    _write_json(record, {"id": chain_id, "model": chain["model"], "scenario": scenario,
                         "name": chain["name"], "reconstruction": chain["reconstruction"],
                         "status": "done", "events": events, "seconds": seconds,
                         "estimated_hours": chain["hours"], "host": socket.gethostname(),
                         "finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    say("chain %s done: %d events; %s" % (chain_id, events, ", ".join(
        "%s %s" % (name, _hours(value)) for name, value in seconds.items())))
    return 0


# ---------------------------------------------------------------------------
# The report


def _number(text):
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _shown(row, value_key="value", spread_key="uncertainty"):
    """A value of a metrics.csv row, with its uncertainty, for the summary."""
    if row is None:
        return "\u2013"
    value, spread = _number(row.get(value_key)), _number(row.get(spread_key))
    if value is None:
        return "\u2013"
    return "%.3g \u00b1 %.2g" % (value, spread) if spread else "%.3g" % value


def metrics_table(rows, model, label, published):
    """Markdown lines of a table of the model's metrics (rows of metrics.csv):
    each with its floor and projection, and Geant4's second sample's and the
    best of the published models' values."""
    by_key = {}
    for row in rows:
        key = (row["observable"], row["scenario"], row["profile"], row["metric"])
        by_key.setdefault(key, {})[row["simulator"]] = row
    lines = ["| Figure | Metric | %s | Floor at its size | Projected to full size | "
             "Geant4, second sample | Best published model |" % label,
             "| --- | --- | ---: | ---: | ---: | ---: | --- |"]
    for row in rows:
        if row["simulator"] != model:
            continue
        others = by_key[(row["observable"], row["scenario"], row["profile"], row["metric"])]
        candidates = [(abs(_number(other["value"])), name) for name, other in others.items()
                      if name in published and _number(other["value"]) is not None]
        best = min(candidates) if candidates else None
        figure = "%s%s, %s" % (row["observable"], " " + row["profile"] if row["profile"] else "",
                               row["scenario"])
        lines.append("| %s | %s | %s | %s | %s | %s | %s |" % (
            figure, row["metric"], _shown(row), _shown(row, "floor", "floor_uncertainty"),
            _shown(row, "projected", "projected_uncertainty"), _shown(others.get(SECOND_GEANT4)),
            "%s: %s" % (best[1], _shown(others[best[1]])) if best else "\u2013"))
    return lines


def chain_records(work, record, parts=None):
    """[(chain, its record or None)] of the plan's chains."""
    found = []
    for chain in record["chains"]:
        _, path = part_files(work, chain["id"], parts)
        try:
            found.append((chain, _read_json(path) if os.path.isfile(path) else None))
        except GitHubError:
            found.append((chain, None))
    return found


def speed_line(records, speed):
    """What the chains took against their estimates, in words, or None."""
    done = [(chain, kept) for chain, kept in records if kept and kept.get("status") == "done"]
    if not done:
        return None
    took = sum(sum((kept.get("seconds") or {}).values()) for _, kept in done) / 3600.0
    estimated = sum(chain["hours"] for chain, _ in done)
    if took <= 0:
        return None
    return ("the %d chain(s) took %.1f runner-hours, against %.1f estimated: the runners ran at "
            "%.2f of the speed of a core of the timed nodes (github.speed is %g)" % (
                len(done), took, estimated, speed * estimated / took, speed))


def sample_label(events):
    """A sample's events of each input, as its label in the figures has them:
    n=1000, or n=1000, tau_pairs 500 where the scenarios differ."""
    from .layout import most_common
    common = most_common(events)
    return ", ".join(["n=%d" % common] + ["%s %d" % (scenario, number)
                                          for scenario, number in events.items()
                                          if number != common])


def summary(record, rows, drawn, missing, labels, notes=()):
    """The job's summary, Markdown."""
    tested = [model for model in record["models"] if model in drawn]
    lines = ["## Low-cost test of %s" % (", ".join(
        "%s (`%s`)" % (labels.get(model, model), model) for model in record["models"])
        or "no models")]
    from .low_cost import events_text
    lines.append("")
    lines.append("Simulated, reconstructed and measured on %s, against the full samples of %s "
                 "as published%s. Each metric's floor is what Geant4 itself would measure with "
                 "as many events; the projection, what the model would measure with as many as "
                 "the full samples." % (
                     events_text(record["events"]), ", ".join(record["compared"]),
                     " (Zenodo record %s)" % ", ".join(record["records"])
                     if record["records"] else ""))
    for model, chains in missing.items():
        lines += ["", "**%s isn't drawn:** %d of its chains have no measurement (%s). Re-run the "
                  "failed jobs to make them." % (model, len(chains), ", ".join(
                      "%s %s/%s" % (c["id"], c["scenario"], c["name"]) for c in chains[:5])
                      + (", ..." if len(chains) > 5 else ""))]
    for note in notes:
        lines += ["", note]
    published = set(record["compared"]) - {GEANT4, SECOND_GEANT4}
    for model in tested:
        lines += ["", "### %s" % labels.get(model, model), ""]
        lines += metrics_table(rows, model, "%s (%s)" % (labels.get(model, model),
                                                         sample_label(record["events"])),
                               published)
    return "\n".join(lines) + "\n"


def read_metrics(path):
    if not os.path.isfile(path):
        return []
    with open(path) as stream:
        return list(csv.DictReader(stream))


def report(work, parts=None, out=None):
    """The report job (see the module docstring); returns its exit status."""
    from . import measure, plot
    from .graph import Graph
    record = read_plan(work)
    parts = parts or os.path.join(work, PARTS)
    out = out or os.path.join(work, RESULTS)
    if not record["models"]:
        say("the plan has no models to test, so there is nothing to report")
        return 0
    plan = make_plan(simulators=record["models"], events=record["events"], low_cost=True)
    plan["published"]["enabled"] = False
    graph = Graph(plan)
    layout = graph.layout
    labels = dict((model["name"], model["label"]) for model in plan["models"])
    drawn, made, missing = [], [], {}
    for model in record["models"]:
        target = layout.observables(model)
        task = graph.by_key["observables:" + target]
        params = task["params"]
        if model in record["reused"]:
            os.makedirs(os.path.dirname(target), exist_ok=True)
            shutil.copyfile(os.path.join(work, record["reused"][model]["file"]), target)
        else:
            chains = [chain for chain in record["chains"] if chain["model"] == model]
            absent = [chain for chain in chains if not part_is_done(work, chain["id"], parts)]
            if absent:
                missing[model] = absent
                say("%s isn't drawn: %d of its %d chain(s) have no measurement, e.g. %s"
                    % (model, len(absent), len(chains), absent[0]["reconstruction"]))
                continue
            measure.merge_parts(dict((chain["reconstruction"],
                                      part_files(work, chain["id"], parts)[0])
                                     for chain in chains), params["files"], target,
                                layout.relative)
            made.append(model)
        current, why = measure.is_up_to_date(target, measure.wanted(params["observables"]),
                                             params["files"], layout.relative)
        if not current:
            raise GitHubError("the measurements of %s, %s, aren't what this plan measures: %s"
                              % (model, target, why))
        drawn.append(model)
        say("%s: %s" % (model, "merged the measurements of its %d chain(s) into %s" % (
            sum(1 for c in record["chains"] if c["model"] == model), target)
            if model in made else "its sample of run %s, %s" % (record["reused"][model]["run"],
                                                                 target)))

    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    rows = []
    if drawn:
        plots = layout.low_cost_plots_dir
        shutil.rmtree(plots, ignore_errors=True)
        argv = ["--dir", layout.observables_dir, "--out", plots, "--no-pareto"]
        for model in drawn:
            argv += ["--sample", "%s:%s" % (model, record["suffix"])]
        if plot.main(argv) != 0:
            raise GitHubError("drawing the plots failed (see above)")
        for name in sorted(os.listdir(plots)):
            source = os.path.join(plots, name)
            (shutil.copytree if os.path.isdir(source) else shutil.copy2)(
                source, os.path.join(out, name))
        rows = read_metrics(os.path.join(out, "metrics.csv"))
        os.makedirs(os.path.join(out, SAMPLES))
        for model in drawn:
            shutil.copy2(layout.observables(model), os.path.join(out, SAMPLES))

    # Each sample made, to be kept for later runs of the same model
    keep = []
    shutil.rmtree(os.path.join(work, KEEP), ignore_errors=True)
    for model in made:
        artifact = record["artifacts"][model]
        folder = os.path.join(work, KEEP, artifact)
        os.makedirs(folder)
        shutil.copy2(layout.observables(model), folder)
        _write_json(os.path.join(folder, SAMPLE_RECORD), {
            "format": SAMPLE_FORMAT, "model": model, "fingerprint": record["fingerprints"][model],
            "events": record["events"], "suffix": record["suffix"], "run": _run_url()})
        keep.append(artifact)

    records = chain_records(work, record, parts)
    notes = []
    for model in drawn:
        if model in record["reused"]:
            notes.append("%s's sample was made by an earlier run, %s, of the same model, "
                         "software and settings." % (model, _run_url(
                             record["reused"][model]["run"])))
    line = speed_line(records, record["settings"]["speed"])
    if line:
        notes.append(line[0].upper() + line[1:] + ".")
    text = summary(record, rows, drawn, missing, labels, notes)
    with open(os.path.join(out, "summary.md"), "w") as stream:
        stream.write(text)
    _write_json(os.path.join(out, "run.json"), dict(
        record, drawn=drawn, made=made, missing=dict((model, [c["id"] for c in chains])
                                                     for model, chains in missing.items()),
        run=_run_url(), chain_records=[kept for _, kept in records if kept]))
    add_summary(text)
    if line:
        say(line)
    set_outputs(keep=keep, drawn=" ".join(drawn))
    if missing:
        say("%s %s chains without measurements, so %s drawn (see above)" % (
            " and ".join(missing), "has" if len(missing) == 1 else "have",
            "it isn't" if len(missing) == 1 else "they aren't"))
        return 1
    return 0


def _run_url(run=None):
    """The address of a workflow run: this one's, or run's of this repository."""
    server = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
    repository = os.environ.get("GITHUB_REPOSITORY")
    run = run or os.environ.get("GITHUB_RUN_ID")
    if not repository or not run:
        return run and "run %s" % run
    return "%s/%s/actions/runs/%s" % (server, repository, run)


# ---------------------------------------------------------------------------
# The command line


def plan_summary(record):
    """The plan job's summary, Markdown."""
    from .low_cost import events_text
    lines = ["## Low-cost plan", ""]
    if not record["models"]:
        lines.append("Every model in `models/` has its full measurements published, so there "
                     "is nothing to test. Run the workflow by hand, naming models, to test them "
                     "anyway.")
        return "\n".join(lines) + "\n"
    lines.append("Testing %s on %s, against the full samples of %s%s." % (
        ", ".join("`%s`" % model for model in record["models"]), events_text(record["events"]),
        ", ".join(record["compared"]) or "nothing",
        " (Zenodo record %s)" % ", ".join(record["records"]) if record["records"] else ""))
    lines += ["", "| Model | Why | Its sample |", "| --- | --- | --- |"]
    for model in record["models"]:
        why = record["why"].get(model) or ""
        if model in record["reused"]:
            sample = "made by an earlier run, %s" % _run_url(record["reused"][model]["run"])
        else:
            sample = "%d chains" % sum(1 for c in record["chains"] if c["model"] == model)
        lines.append("| `%s` | %s | %s |" % (model, "named" if why == "named"
                                              else "not published: " + why, sample))
    if record["chains"]:
        lines += ["", "| Scenario | Chains | Events of each input | Hours each, estimated | "
                  "Hours in all |", "| --- | ---: | ---: | ---: | ---: |"]
        by_scenario = {}
        for chain in record["chains"]:
            count, hours = by_scenario.get(chain["scenario"], (0, 0.0))
            by_scenario[chain["scenario"]] = (count + 1, hours + chain["hours"])
        for scenario, (count, hours) in by_scenario.items():
            lines.append("| %s | %d | %d | %.2f | %.1f |" % (
                scenario, count, record["events"][scenario], hours / count, hours))
        settings = record["settings"]
        lines += ["", "%d shard%s, %d chain%s at once on each, planned for %s h: about %.1f "
                  "runner-hours with their set-up (`github.budget_hours` is %g)." % (
                      len(record["shards"]), "" if len(record["shards"]) == 1 else "s",
                      settings["parallel"], "" if settings["parallel"] == 1 else "s",
                      ", ".join("%.1f" % shard["hours"] for shard in record["shards"]),
                      record["runner_hours"], settings["budget_hours"])]
    return "\n".join(lines) + "\n"


def shard_summary(record, index, work):
    """A shard job's summary, Markdown: what became of each of its chains."""
    lines = ["## Shard %d" % index, "",
             "| Chain | Model | Scenario | Simulation | Done | Took | Estimated |",
             "| --- | --- | --- | --- | --- | ---: | ---: |"]
    by_id = dict((chain["id"], chain) for chain in record["chains"])
    for chain_id in record["shards"][index]["chains"]:
        chain = by_id[chain_id]
        _, path = part_files(work, chain_id)
        try:
            kept = _read_json(path) if os.path.isfile(path) else {}
        except GitHubError:
            kept = {}
        done = kept.get("status") == "done"
        lines.append("| %s | %s | %s | %s | %s | %s | %.2f h |" % (
            chain_id, chain["model"], chain["scenario"], chain["name"],
            "yes" if done else "**no** (see the shard's log)",
            _hours(sum((kept.get("seconds") or {}).values())) if done else "",
            chain["hours"]))
    return "\n".join(lines) + "\n"


def main(argv):
    import argparse
    parser = argparse.ArgumentParser(
        prog="ddml-bench github", description="The low-cost mode on GitHub's runners "
        "(.github/workflows/low-cost.yml; see ddml_bench/github.py and docs/github.md).")
    parser.add_argument("command", choices=("keys", "plan", "check", "preflight", "shard",
                                            "report", "tasks-check", "preflight-tasks",
                                            "chains", "chain"))
    parser.add_argument("arguments", nargs="*")
    parser.add_argument("--dir", default=DEFAULT_DIR,
                        help="the folder of the run (default: low_cost_run in the repository)")
    parser.add_argument("--models", default="",
                        help="plan: the models to test, separated by spaces or commas (default: "
                             "every model in models/ without published full measurements)")
    parser.add_argument("-n", "--events", default="",
                        help="plan: the first N events of each input of every scenario "
                             "(default: low_cost.events)")
    parser.add_argument("--fresh", action="store_true",
                        help="plan: make every sample, even those kept by earlier runs")
    parser.add_argument("--any-stack", action="store_true",
                        help="plan: compare with the published measurements even where this "
                             "configuration pins other software")
    parser.add_argument("--parts", help="report: the folder of the chains' measurements "
                                        "(default: DIR/parts)")
    parser.add_argument("--out", help="report: where the results go (default: DIR/results)")
    parser.add_argument("--keep", action="store_true",
                        help="chain: keep the simulation and reconstruction")
    args = parser.parse_args(argv)
    work = os.path.abspath(args.dir)

    def arguments(count, names):
        if len(args.arguments) != count:
            parser.error("%s takes %s" % (args.command, names))
        return args.arguments

    if args.command == "keys":
        arguments(0, "no arguments")
        set_outputs(**cache_keys(load_config()))
        return 0
    if args.command == "plan":
        arguments(0, "no arguments")
        events = args.events.strip()
        if events and (not events.isdigit() or int(events) < 1):
            parser.error("-n needs a whole number, at least 1, not '%s'" % events)
        models = [m for m in args.models.replace(",", " ").split() if m]
        path = os.path.join(work, PLAN)
        process.remove(path)
        try:
            record = plan_run(work, models, int(events) if events else None, args.fresh,
                              args.any_stack)
        except Exception as error:
            # What was planned, if it was, and why it stopped
            add_summary((plan_summary(_read_json(path)) if os.path.isfile(path)
                         else "## Low-cost plan") + "\n\n**Stopped:** %s" % error)
            raise
        add_summary(plan_summary(record))
        set_outputs(shards=list(range(len(record["shards"]))), chains=len(record["chains"]),
                    report="true" if record["models"] else "false",
                    models=" ".join(record["models"]), black=black_requirement(load_config()))
        return 0
    if args.command == "check":
        arguments(0, "no arguments")
        return check(work)
    if args.command == "preflight":
        arguments(0, "no arguments")
        return preflight(work)
    if args.command == "preflight-tasks":
        return preflight_tasks(arguments(1, "the file of the preflight's tasks")[0])
    if args.command == "shard":
        index, = arguments(1, "the number of the shard")
        if not index.isdigit():
            parser.error("give the number of the shard, from 0, not '%s'" % index)
        return run_shard(work, int(index))
    if args.command == "report":
        arguments(0, "no arguments")
        return report(work, args.parts and os.path.abspath(args.parts),
                      args.out and os.path.abspath(args.out))
    if args.command == "tasks-check":
        return check_tasks(arguments(1, "the file of the chains")[0])
    if args.command == "chains":
        return run_chains(arguments(1, "the file of the shard")[0])
    path, chain_id = arguments(2, "the file of the shard and the chain")
    return run_chain(path, chain_id, keep=args.keep)
