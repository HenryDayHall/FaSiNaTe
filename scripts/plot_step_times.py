#!/usr/bin/env python3
"""Plot the time taken by each pipeline step, from the FaSiNaTe logs.

Simulation and reconstruction (steps 2 and 3) are timed from the log written
next to each output, which ends, when the job succeeded, in

    # finished  2026-10-16 12:59:38, after 1793 s

These logs are found in every data folder: the paths.data_dir of the FaSiNaTe
configuration (scripts/fasinate_config.py of the repository, with
configs/local.yaml and $FASINATE_CONFIG) and the data_dir in each pipeline
run's plan.json; or only those given with --data-dir.  Their folder gives the
scenario and simulator:

    sim_output/<scenario>/<simulator>/<sample>.log      2  simulation
    rec_output/<scenario>/<simulator>/<sample>.log      3  reconstruction

Test samples (<input>_n<N>, from -n N) and the scenario "custom" (files given
directly) are left out, and so are the logs of segments split from a whole
job's output by scripts/split_data.py, which didn't take time of their own:
the whole jobs' logs are in <data_dir>/.unsplit, which --data-dir can name.  A log is rewritten each time its output is made
again, so it times the latest attempt.

The other steps are timed from the slurm logs, every *.out file under the given
paths (default: ./slurm): the gap between the two ddml-bench lines of the job,

    ddml-bench: job 25369500 on max-wn148.desy.de, 2026-10-21 18:43:42: generate:...
    ddml-bench: job 25369500 finished 2026-10-21 18:44:26, with status 0

with step, model and scenario read off the log name (slurm logs of
simulation and reconstruction jobs are not used):

    001-generate-di_photon-diphoton_en20_..._seed0.out           1  -            di_photon
    003-geant4-di_photon-diphoton_en20_..._seed0_simseed1.out    2  geant4       di_photon
    009-models-caloclouds3-di_photon-..._simseed1.out            2  caloclouds3  di_photon
    015-reconstruct-geant4-di_photon-..._simseed1.out            3  geant4       di_photon
    028-observables-caloclouds3.out                              4  caloclouds3  (plan.json)
    029-plot.out                                                 5  -            (plan.json)

An observables job measures every scenario of its simulator and the plot job
plots them all, so their scenario is the list of measured scenarios in the
run's plan.json ("All scenarios" when there are several).  Without a plan.json,
a log whose name has no scenario takes the scenario of the other logs in its
run directory.

Only Geant4 and the models described in models/ now (models/<name>.yaml, of
the repository: --repo, $FASINATE_REPO_DIR, or the one this script is in) are
plotted, each labelled by its description's name; --all-models plots every
model found in the logs.  With --since DATE, jobs that started before DATE are
left out, e.g. those from before the inputs were split into segments.

All jobs that share step, model and scenario make one point - repeats in other
runs, other seeds and simseeds, other input configurations - with their mean as
the marker and their spread as the error bar.  Colour = model, marker =
scenario; steps not tied to one model (input generation, plotting) are grey.

examples, from the data folder:
  python3 plot_step_times.py                            # ./slurm -> step_times.png
  python3 plot_step_times.py --since 2026-11-12         # only jobs started since then
  python3 plot_step_times.py slurm/pipeline_* -o step_times.pdf --csv jobs.csv
  python3 plot_step_times.py --repo ~/training/analysis/FaSiNaTe
  python3 plot_step_times.py --data-dir /data/dust/user/dayhallh/analysis/data
  python3 plot_step_times.py --error minmax --linear
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import re
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

# -----------------------------------------------------------------------------
# Configuration: edit here to rename steps or restyle the plot.

STEP_NAMES = {
    1: "Input slcio\ngeneration",
    2: "Simulation",
    3: "Reconstruction",
    4: "Observable\nextraction",
    5: "Plotting",
}

# First word of a log name (after the "NNN-" index) -> step.  A first word that
# is a model name, as in "003-geant4-...", also means simulation.
# More can be added from the command line with --stage WORD=STEP.
STAGE_STEPS = {
    "generate": 1, "generation": 1,
    "models": 2, "simulate": 2, "simulation": 2,
    "reconstruct": 3, "reconstruction": 3, "reco": 3,
    "observables": 4, "observable": 4,
    "plot": 5, "plots": 5, "plotting": 5,
}

# Name in the log names -> (legend label, colour).  Dict order = legend order.
# The models of models/ get the spare colours in order of their names, and the
# name in their description as label.  "" is for steps not tied to one model.
MODELS = {
    "": ("Model-independent", "#6b6b6b"),
    "geant4": ("Geant4", "#0072B2"),
}
# Name in the log names -> (legend label, marker).  Scenarios not listed are
# picked up from "generate-<scenario>-..." etc. and get a spare marker.
SCENARIOS = {
    "single_photon": ("Single photon gun", "o"),
    "photon": ("Single photon gun", "o"),
    "di_photon": ("Di-photon", "s"),
    "tau_pairs": ("τ pairs (physics sample)", "v"),
}
ALL, UNKNOWN = "<all>", "<unknown>"  # logs whose name and run give no single scenario
PSEUDO_SCENARIOS = {ALL: ("All scenarios", "D"), UNKNOWN: ("Scenario unknown", "X")}
# (in this order, neighbours stay apart for colour-blind readers too)
SPARE_COLOURS = ["#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9", "#000000"]
# The repository this script is in, whose models/ and configuration are used
# unless --repo or $FASINATE_REPO_DIR names another
SCRIPT_REPO = Path(__file__).resolve().parent.parent
SPARE_MARKERS = ["o", "^", "v", "P", "h", "*", "<", ">"]

ERROR_TEXT = {
    "std": "marker: mean over jobs, bar: ±1 standard deviation",
    "sem": "marker: mean over jobs, bar: ±1 standard error of the mean",
    "minmax": "marker: mean over jobs, bar: min to max",
}
UNIT_SECONDS = {"s": 1.0, "min": 60.0, "h": 3600.0}

# -----------------------------------------------------------------------------

# As job.sh of slurm_pipeline.sh prints them (the job ID is empty outside slurm)
START_RE = re.compile(
    r"ddml-bench: job (?P<job>\S*) on (?P<host>[^,\s]+), "
    r"(?P<time>\d{4}-\d\d-\d\d[ T]\d\d:\d\d:\d\d):?\s*(?P<task>.*)"
)
END_RE = re.compile(
    r"ddml-bench: job (?P<job>\S*) finished "
    r"(?P<time>\d{4}-\d\d-\d\d[ T]\d\d:\d\d:\d\d),? with status (?P<status>-?\d+)"
)
RUN_STAMP = re.compile(r"_\d{8}_\d{6}_\d+$")  # pipeline_20261021_184320_3142239
INDEX = re.compile(r"^\d+-")                  # 001-
GATHERING_STEPS = {4, 5}  # one job covers every measured scenario: observables, plot

# The logs next to the outputs (ddml_bench/layout.py, stages/simulate.py and
# stages/reconstruct.py): <data_dir>/<folder>/<scenario>/<simulator>/<sample>.log
OUTPUT_STEPS = {"sim_output": 2, "rec_output": 3}
OUTPUT_STARTED_RE = re.compile(
    r"^# started\s+(?P<time>\d{4}-\d\d-\d\d[ T]\d\d:\d\d:\d\d) on (?P<host>\S+)", re.M)
OUTPUT_END_RE = re.compile(r"^# (?P<what>finished|FAILED|INTERRUPTED)\b", re.M)
OUTPUT_FINISHED_RE = re.compile(
    r"# finished\s+(?P<time>\d{4}-\d\d-\d\d[ T]\d\d:\d\d:\d\d), after (?P<seconds>\d+(?:\.\d+)?) s")
# The log of a segment split from a whole job's output by scripts/split_data.py,
# which has that job's header but no time of its own
OUTPUT_SEGMENT_RE = re.compile(r"^# segment\s", re.M)
TEST_SAMPLE = re.compile(r"(?:^|_)n\d+(?=_|$)")  # <input>_n<N>, from -n N
CUSTOM_SCENARIO = "custom"


@dataclass
class Log:
    path: Path
    run_dir: Path                 # e.g. slurm/pipeline_20261021_184320_3142239
    run: str                      # its name
    words: list                   # log name split at "-", without the index
    step: int = 0
    model: str = ""               # "" = not tied to one model
    scenario: str | None = None   # None until worked out from the run
    config: str = ""              # rest of the name (input, seeds)


@dataclass
class Job:
    log: Log
    seconds: float
    status: int
    start: datetime
    end: datetime
    host: str
    slurm_job: str
    task: str


@dataclass
class Point:
    step: int
    model: str
    scenario: str
    seconds: list = field(default_factory=list)
    low: float = 0.0              # error bar below / above the mean [s]
    high: float = 0.0

    @property
    def n(self):
        return len(self.seconds)

    @property
    def mean(self):
        return statistics.fmean(self.seconds)

    @property
    def std(self):
        return statistics.stdev(self.seconds) if self.n > 1 else 0.0


# ------------------------------------------------------------------ log names

def find_logs(paths, pattern):
    logs, seen = [], set()
    for root in paths:
        if root.is_dir():
            files = sorted(root.rglob(pattern))
        elif root.is_file():
            files = [root]
        else:
            print(f"warning: {root} does not exist", file=sys.stderr)
            continue
        for path in files:
            if path.is_file() and path.resolve() not in seen:
                seen.add(path.resolve())
                words = [w for w in INDEX.sub("", path.stem).split("-") if w]
                run_dir = run_of(path)
                logs.append(Log(path, run_dir, run_dir.name, words))
    return logs


def run_of(path):
    """The run directory a log belongs to, e.g. .../pipeline_20261021_184320_3142239."""
    for parent in path.parents:
        if RUN_STAMP.search(parent.name):
            return parent
    return path.parent


_plans = {}


def read_plan(run_dir):
    """A pipeline run's plan.json, or {} without one."""
    if run_dir not in _plans:
        try:
            with open(run_dir / "plan.json") as fh:
                plan = json.load(fh)
            _plans[run_dir] = plan if isinstance(plan, dict) else {}
        except (OSError, ValueError):
            _plans[run_dir] = {}
    return _plans[run_dir]


def measured_scenarios(run_dir):
    """The scenarios the observables of a pipeline run measure, as its plan.json
    lists them (plan["observables"]["scenarios"]), or None without one."""
    try:
        found = read_plan(run_dir)["observables"]["scenarios"]
        return [s for s in found if isinstance(s, str)] or None
    except (KeyError, TypeError):
        return None


# ------------------------------------------------- logs next to the outputs

def config_data_dir(repo, config_files):
    """paths.data_dir of the FaSiNaTe configuration in repo, as
    scripts/fasinate_config.py merges it, or (None, why not)."""
    script = Path(repo).expanduser() / "scripts" / "fasinate_config.py"
    if not script.is_file():
        return None, f"no {script}"
    env = dict(os.environ)
    if config_files:
        env["FASINATE_CONFIG"] = ":".join(
            [f for f in env.get("FASINATE_CONFIG", "").split(":") if f] + list(config_files))
    python = os.environ.get("FASINATE_PYTHON") or sys.executable
    try:
        result = subprocess.run([python, str(script), "--get", "paths.data_dir"],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                universal_newlines=True, env=env, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, f"{script} failed: {exc}"
    if result.returncode != 0 or not result.stdout.strip():
        why = (result.stderr or result.stdout).strip().splitlines()
        return None, f"{script} failed: {why[-1] if why else result.returncode}"
    return Path(result.stdout.strip().splitlines()[-1]), ""


def find_repo(args, run_dirs):
    """The FaSiNaTe repository: --repo, $FASINATE_REPO_DIR, the one this script
    is in, or the one most of the runs' plan.json name; None if none is found."""
    if args.repo or os.environ.get("FASINATE_REPO_DIR"):
        return Path(args.repo or os.environ["FASINATE_REPO_DIR"]).expanduser()
    if (SCRIPT_REPO / "scripts" / "fasinate_config.py").is_file():
        return SCRIPT_REPO
    repos = Counter(p["repo_dir"] for p in map(read_plan, run_dirs)
                    if isinstance(p.get("repo_dir"), str))
    return Path(repos.most_common(1)[0][0]) if repos else None


NAME_LINE = re.compile(r"^name:[ \t]*(?P<name>.*?)[ \t]*(?:[ \t]#.*)?$", re.M)


def current_models(models_dir):
    """{model: legend label} of the models described in models_dir now
    (<model>.yaml; the label is its top-level name:), or None if there is no
    such folder."""
    if not models_dir.is_dir():
        return None
    found = {}
    for path in sorted(models_dir.glob("*.yaml")):
        if path.name.startswith(".") or not path.is_file():
            continue
        try:
            match = NAME_LINE.search(path.read_text(errors="replace"))
        except OSError:
            match = None
        label = match["name"].strip("'\"") if match and match["name"] else ""
        found[path.stem] = label or path.stem
    return found


def since_arg(text):
    for form in ("%Y-%m-%d", "%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text.strip().replace("T", " "), form)
        except ValueError:
            pass
    raise argparse.ArgumentTypeError(
        f"expected a date, YYYY-MM-DD or 'YYYY-MM-DD HH:MM', got {text!r}")


def find_data_dirs(args, run_dirs, repo):
    """[(data folder, where it came from)]: those given, or the configuration's
    and those in the runs' plan.json."""
    if args.data_dir:
        found = [(d, "--data-dir") for d in args.data_dir]
    else:
        plans = [read_plan(r) for r in run_dirs]
        found = [(Path(p["data_dir"]), "plan.json") for p in plans
                 if isinstance(p.get("data_dir"), str)]
        if repo:
            data_dir, why = config_data_dir(repo, args.fasinate_config)
            if data_dir is None:
                print(f"warning: can't read the FaSiNaTe configuration: {why}", file=sys.stderr)
            else:
                found.insert(0, (data_dir, f"configuration in {repo}"))
        elif not found:
            print("warning: no FaSiNaTe repository to read the configuration from; "
                  "give one with --repo, or the data folder with --data-dir", file=sys.stderr)
    unique, seen = [], set()
    for path, source in found:
        key = os.path.realpath(path)
        if key not in seen:
            seen.add(key)
            unique.append((path, source))
    return unique


def read_ends(path, size=16384):
    """The first and last few KB of a file, as text (the logs can be large)."""
    with open(path, "rb") as fh:
        head = fh.read(size)
        fh.seek(0, os.SEEK_END)
        end = fh.tell()
        fh.seek(max(0, end - size))
        tail = fh.read()
    return head.decode("utf-8", "replace"), tail.decode("utf-8", "replace")


def output_logs(data_dirs, args, parse_time, skip):
    """Jobs timed from the logs next to the simulation and reconstruction outputs."""
    jobs, seen = [], set()
    for data_dir, _ in data_dirs:
        for folder, step in OUTPUT_STEPS.items():
            for path in sorted((data_dir / folder).glob("*/*/*.log")):
                key = os.path.realpath(path)
                if key in seen or path.name.startswith(".") or not path.is_file():
                    continue
                seen.add(key)
                scenario, simulator = path.parent.parent.name, path.parent.name
                sample = path.stem
                log = Log(path, data_dir, folder, [], step, simulator, scenario, sample)
                if scenario == CUSTOM_SCENARIO:
                    skip("scenario 'custom' (files given directly)", log)
                    continue
                if TEST_SAMPLE.search(sample) and not args.include_test:
                    skip("test sample (_n<N>; use --include-test)", log)
                    continue
                try:
                    head, tail = read_ends(path)
                except OSError as exc:
                    skip(f"unreadable ({exc.strerror})", log)
                    continue
                if OUTPUT_SEGMENT_RE.search(head):
                    skip("split from a whole job's output by split_data.py, so not timed (the "
                         "whole job's log is in .unsplit: --data-dir <data_dir>/.unsplit)", log)
                    continue
                ends = list(OUTPUT_END_RE.finditer(tail))
                if not ends:
                    skip("no '# finished' line (still running, or killed)", log)
                    continue
                if ends[-1]["what"] != "finished":
                    skip(f"its last attempt ended {ends[-1]['what']}", log)
                    continue
                done = OUTPUT_FINISHED_RE.match(tail, ends[-1].start())
                if done is None:
                    skip("unreadable '# finished' line", log)
                    continue
                seconds = float(done["seconds"])
                end = parse_time(done["time"])
                started = OUTPUT_STARTED_RE.search(head)
                start = parse_time(started["time"]) if started else end - timedelta(seconds=seconds)
                host = started["host"] if started else ""
                jobs.append(Job(log, seconds, 0, start, end, host, "", ""))
    return jobs


def run_words(run):
    """simulate_geant4_20261016_122817_3474267 -> ['simulate', 'geant4']."""
    base = RUN_STAMP.sub("", run)
    return base.split("_") if base != run else []


def learn_names(logs, stage_steps):
    """The listed models/scenarios plus those read off names whose layout is
    unambiguous: generate-<scenario>-<input>, <model>-<scenario>-<input> and
    models-<model>-<scenario>-<input>."""
    models = {m for m in MODELS if m}
    scenarios = set(SCENARIOS)
    for log in logs:
        stage, rest = (log.words or [""])[0], log.words[1:]
        if len(rest) >= 2 and (stage_steps.get(stage) == 1 or stage in models):
            scenarios.add(rest[0])
    for log in logs:
        stage, rest = (log.words or [""])[0], log.words[1:]
        if stage_steps.get(stage) == 2 and len(rest) >= 2 and rest[0] not in scenarios:
            models.add(rest[0])
            if len(rest) >= 3:
                scenarios.add(rest[1])
    return models, scenarios


def classify(words, stage_steps, models, scenarios):
    """(step, model, scenario, config) for a log name, or None if its first word
    is not a known stage or model."""
    if not words:
        return None
    stage, rest = words[0], words[1:]
    if stage in stage_steps:
        step, found_models = stage_steps[stage], set()
    elif stage in models:                        # e.g. 003-geant4-...
        step, found_models = 2, {stage}
    else:
        return None
    found_scenarios, other = set(), []
    for word in rest:
        if word in models:
            found_models.add(word)
        elif word in scenarios:
            found_scenarios.add(word)
        else:
            other.append(word)
    model = found_models.pop() if len(found_models) == 1 else ""
    if len(found_scenarios) == 1:
        scenario = found_scenarios.pop()
    else:
        scenario = ALL if found_scenarios else None  # None: taken from the run
    return step, model, scenario, "-".join(other)


# --------------------------------------------------------------------- timing

def time_parser(tz_name):
    tz = None
    if tz_name:
        try:
            from zoneinfo import ZoneInfo
            tz = ZoneInfo(tz_name)
        except Exception as exc:  # Python < 3.9, or no time zone database
            print(f"warning: time zone {tz_name!r} unavailable ({exc}); durations "
                  "across a clock change will be off by an hour", file=sys.stderr)

    def parse(text):
        t = datetime.strptime(text.replace("T", " "), "%Y-%m-%d %H:%M:%S")
        return t.replace(tzinfo=tz) if tz else t

    return parse


def duration(start, end):
    """(seconds from start to end, end) honouring daylight-saving changes; the end
    is moved to the second pass of the repeated autumn hour if that is the only
    reading that comes after the start."""
    if start.tzinfo is None:
        return (end - start).total_seconds(), end
    utc0 = start.astimezone(timezone.utc)
    seconds = (end.astimezone(timezone.utc) - utc0).total_seconds()
    if seconds < 0:
        end = end.replace(fold=1)
        seconds = (end.astimezone(timezone.utc) - utc0).total_seconds()
    return seconds, end


def read_attempts(path, excludes):
    """[(start match, end match or None), ...] for each start line in the log,
    and whether one of the --exclude patterns matched."""
    attempts, start = [], None
    with open(path, "rb") as fh:
        for raw in fh:
            if excludes:
                text = raw.decode("utf-8", "replace")
                if any(rx.search(text) for rx in excludes):
                    return [], True
            if b"ddml-bench: job" not in raw:
                continue
            line = raw.decode("utf-8", "replace")
            match = START_RE.search(line)
            if match:
                if start is not None:  # an earlier attempt that never finished
                    attempts.append((start, None))
                start = match
                continue
            match = END_RE.search(line)
            if match and start is not None:
                attempts.append((start, match))
                start = None
    if start is not None:
        attempts.append((start, None))
    return attempts, False


# ------------------------------------------------------------ pooling, styles

def summarise(jobs, error):
    pooled = defaultdict(list)
    for job in jobs:
        pooled[(job.log.step, job.log.model, job.log.scenario)].append(job.seconds)
    points = []
    for (step, model, scenario), seconds in pooled.items():
        p = Point(step, model, scenario, seconds)
        if error == "minmax":
            p.low, p.high = p.mean - min(seconds), max(seconds) - p.mean
        else:
            p.low = p.high = p.std / (math.sqrt(p.n) if error == "sem" else 1.0)
        points.append(p)
    return sorted(points, key=lambda p: (p.step, model_rank(p.model), scenario_rank(p.scenario)))


def auto_unit(points):
    longest = max(p.mean for p in points)
    return "s" if longest < 300 else "min" if longest < 10 * 3600 else "h"


def model_rank(model):
    return (0, list(MODELS).index(model), "") if model in MODELS else (1, 0, model)


def scenario_rank(scenario):
    if scenario in SCENARIOS:
        return (0, list(SCENARIOS).index(scenario), "")
    if scenario in PSEUDO_SCENARIOS:
        return (2, list(PSEUDO_SCENARIOS).index(scenario), "")
    return (1, 0, scenario)


def styles(names, listed, spares):
    """name -> (label, colour or marker): listed names as given, others get spares."""
    used = {listed[n][1] for n in names if n in listed}
    free = itertools.cycle([s for s in spares if s not in used] or spares)
    return {n: listed[n] if n in listed else (n.replace("_", " ").capitalize(), next(free))
            for n in names}


# -------------------------------------------------------------------- outputs

def print_table(points, unit, model_style, scenario_style):
    k = UNIT_SECONDS[unit]
    wm = max([len("model")] + [len(model_style[p.model][0]) for p in points]) + 2
    ws = max([len("scenario")] + [len(scenario_style[p.scenario][0]) for p in points]) + 2
    print(f"\n{'step':<26}{'model':<{wm}}{'scenario':<{ws}}{'n':>5}"
          f"{'mean':>9}{'std':>9}{'min':>9}{'max':>9}  [{unit}]")
    for p in points:
        step = f"{p.step} " + STEP_NAMES[p.step].replace("\n", " ")
        print(f"{step:<26}{model_style[p.model][0]:<{wm}}{scenario_style[p.scenario][0]:<{ws}}"
              f"{p.n:>5}{p.mean / k:>9.2f}{p.std / k:>9.2f}"
              f"{min(p.seconds) / k:>9.2f}{max(p.seconds) / k:>9.2f}")
    print()


def write_csv(jobs, path):
    with open(path, "w", newline="") as fh:
        out = csv.writer(fh)
        out.writerow(["run", "log", "step", "model", "scenario", "config", "seconds",
                      "status", "start", "end", "host", "slurm_job", "task"])
        for j in jobs:
            out.writerow([j.log.run, j.log.path, j.log.step, j.log.model, j.log.scenario,
                          j.log.config, round(j.seconds), j.status, j.start.isoformat(" "),
                          j.end.isoformat(" "), j.host, j.slurm_job, j.task])
    print(f"wrote {path}")


def plot(points, unit, model_style, scenario_style, args, source):
    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import FuncFormatter

    k = UNIT_SECONDS[unit]
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    fig.subplots_adjust(left=0.08, right=0.75, bottom=0.17, top=0.9)

    tops, bottoms = [], []
    for step in STEP_NAMES:
        here = [p for p in points if p.step == step]  # sorted by model, then scenario
        gap = min(0.17, 0.7 / max(len(here) - 1, 1))  # side-by-side so bars don't overlap
        for i, p in enumerate(here):
            x = step + (i - (len(here) - 1) / 2) * gap
            y, low, high = p.mean / k, p.low / k, p.high / k
            ax.errorbar(x, y, yerr=[[low], [high]] if low or high else None,
                        fmt=scenario_style[p.scenario][1], color=model_style[p.model][1],
                        ms=8, mec="white", mew=1.3, elinewidth=1.5, capsize=3.5,
                        capthick=1.5, zorder=3)
            tops.append(y + high)
            bottoms += [v for v in (y - low, y) if v > 0]
            if args.counts:
                ax.annotate(f"n={p.n}", (x, y + high), xytext=(0, 3),
                            textcoords="offset points", ha="center", va="bottom",
                            fontsize=6.5, color="0.4")

    if args.linear:
        ax.set_ylim(0, max(tops) * 1.15)
    else:
        ax.set_yscale("log")
        ax.set_ylim(min(bottoms, default=max(tops) / 100) / 2, max(tops) * 3)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))

    ax.set_xlim(0.45, len(STEP_NAMES) + 0.55)
    ax.set_xticks(list(STEP_NAMES))
    ax.set_xticklabels([f"{s}\n{name}" for s, name in STEP_NAMES.items()])
    ax.tick_params(axis="x", length=0)
    for s in list(STEP_NAMES)[1:]:
        ax.axvline(s - 0.5, color="0.85", lw=0.8, zorder=1)
    ax.grid(axis="y", color="0.92", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("0.45")
    ax.set_xlabel("Step")
    ax.set_ylabel(f"Wall time per job [{unit}]")
    ax.set_title(args.title, loc="left")
    ax.set_title(f"{ERROR_TEXT[args.error]}\n{source}", loc="right", fontsize=8, color="0.35")

    # Hand-made legends: colour <-> model, marker shape <-> scenario.
    def legend(handles, title, top, **kw):
        kw.update(handles=handles, title=title, loc="upper left", bbox_to_anchor=(1.02, top),
                  borderaxespad=0, frameon=False)
        try:
            return ax.legend(alignment="left", **kw)
        except TypeError:  # matplotlib < 3.6
            return ax.legend(**kw)

    by_colour = legend([Patch(facecolor=c, edgecolor="none", label=label)
                        for label, c in dict.fromkeys(model_style.values())],
                       "Model (colour)", 1.0, handlelength=1.3, handleheight=1.0)
    ax.add_artist(by_colour)
    by_colour.set_clip_on(False)  # add_artist clips to the axes, which hides it from the crop
    fig.canvas.draw()
    below = by_colour.get_window_extent().transformed(ax.transAxes.inverted()).y0 - 0.06
    legend([Line2D([], [], ls="none", marker=m, ms=8, mfc="white", mec="0.2", mew=1.1,
                   label=label)
            for label, m in dict.fromkeys(scenario_style.values())],
           "Scenario (marker)", below)

    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"wrote {args.out}")
    if args.show:
        plt.show()


# ----------------------------------------------------------------------- main

def stage_arg(text):
    word, _, step = text.partition("=")
    if not word or step not in {str(s) for s in STEP_NAMES}:
        raise argparse.ArgumentTypeError(
            f"expected WORD=STEP with STEP one of {list(STEP_NAMES)}, got {text!r}")
    return word, int(step)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="*", type=Path, default=[Path("slurm")],
                   help="slurm directories, run directories or single logs (default: ./slurm)")
    p.add_argument("--data-dir", metavar="DIR", type=Path, action="append", default=[],
                   help="data folder holding sim_output/ and rec_output/ (repeatable); by "
                        "default the configuration's paths.data_dir and those in plan.json")
    p.add_argument("--repo", metavar="DIR",
                   help="FaSiNaTe repository whose configuration names the data folder and "
                        "whose models/ names the models to plot (default: $FASINATE_REPO_DIR, "
                        "else the one this script is in, else repo_dir in plan.json)")
    p.add_argument("--since", metavar="DATE", type=since_arg,
                   help="leave out jobs that started before DATE (YYYY-MM-DD or "
                        "'YYYY-MM-DD HH:MM', in --tz)")
    p.add_argument("--models-dir", metavar="DIR", type=Path,
                   help="plot only Geant4 and the models described in DIR (<model>.yaml) "
                        "(default: models/ of the repository)")
    p.add_argument("--all-models", action="store_true",
                   help="plot every model found in the logs, described in models/ or not")
    p.add_argument("--fasinate-config", metavar="FILE", action="append", default=[],
                   help="configuration file to merge over the repository's, as its scripts' "
                        "--config (repeatable)")
    p.add_argument("--include-test", action="store_true",
                   help="also time test samples (<input>_n<N>, made with -n N)")
    p.add_argument("-o", "--out", type=Path, default=Path("step_times.png"),
                   help="figure to write; the suffix sets the format (default: step_times.png)")
    p.add_argument("--csv", type=Path, help="also write one row per timed job to this CSV")
    p.add_argument("--error", choices=list(ERROR_TEXT), default="std",
                   help="error bar: standard deviation (default), standard error of the "
                        "mean, or min to max")
    p.add_argument("--unit", choices=["auto", *UNIT_SECONDS], default="auto",
                   help="time unit (default: picked from the data)")
    p.add_argument("--linear", action="store_true", help="linear y axis instead of log")
    p.add_argument("--include-failed", action="store_true",
                   help="also use jobs that finished with a non-zero status")
    p.add_argument("--exclude", metavar="REGEX", action="append", default=[],
                   help="skip logs with a line matching REGEX (repeatable)")
    p.add_argument("--stage", metavar="WORD=STEP", action="append", default=[], type=stage_arg,
                   help="treat logs whose name starts with WORD as step STEP, e.g. --stage figures=5")
    p.add_argument("--tz", default="Europe/Berlin",
                   help="time zone of the log timestamps, so durations across a "
                        "daylight-saving change are right ('' to ignore)")
    p.add_argument("--glob", default="*.out", help="log file pattern (default: *.out)")
    p.add_argument("--title", default="Time taken per pipeline step")
    p.add_argument("--counts", action="store_true",
                   help="write the number of jobs above each point (always in the printed table)")
    p.add_argument("--show", action="store_true", help="also open the plot in a window")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    stage_steps = {**STAGE_STEPS, **dict(args.stage)}
    parse_time = time_parser(args.tz)
    excludes = [re.compile(rx) for rx in args.exclude]

    jobs, skipped, example = [], Counter(), {}

    def skip(reason, log):
        skipped[reason] += 1
        example.setdefault(reason, log.path)

    logs = find_logs(args.paths, args.glob)
    if not logs:
        print(f"warning: no {args.glob} files under {' '.join(map(str, args.paths))}, so "
              "only simulation and reconstruction are timed; run from the directory that "
              "holds slurm/ or pass its path", file=sys.stderr)

    run_dirs = sorted({log.run_dir for log in logs})
    repo = find_repo(args, run_dirs)
    models_dir = args.models_dir or (repo / "models" if repo else None)
    labels = current_models(models_dir) if models_dir else None  # {model: legend label}
    allowed = None  # the models to plot, None for all
    if labels is None:
        if not args.all_models:
            print(f"warning: no models folder ({models_dir or 'no repository found'}), so every "
                  "model in the logs is plotted; give one with --models-dir or --repo",
                  file=sys.stderr)
    elif not args.all_models:
        allowed = set(labels) | {"geant4"}
        print(f"models in {models_dir}: {', '.join(labels) or 'none'} (and Geant4)")
    since = None
    if args.since:
        since = parse_time(f"{args.since:%Y-%m-%d %H:%M:%S}")
        print(f"only jobs started since {since:%Y-%m-%d %H:%M} ({args.tz or 'local time'})")

    # 1. Simulation and reconstruction, from the logs next to their outputs.
    data_dirs = find_data_dirs(args, run_dirs, repo)
    for data_dir, source in data_dirs:
        print(f"data folder {data_dir}  (from the {source})")
    out_jobs = output_logs(data_dirs, args, parse_time, skip)
    n_out_logs = len(out_jobs) + sum(skipped.values())
    jobs += out_jobs
    if not out_jobs:
        print("warning: no finished simulation or reconstruction logs found, so steps 2 and 3 "
              "are empty; give the data folder with --data-dir", file=sys.stderr)

    # 2. Step, model and scenario of each slurm log, from its name.
    models, scenarios = learn_names(logs, stage_steps)
    models |= {j.log.model for j in out_jobs}  # e.g. observables-<model> of a new model
    scenarios |= {j.log.scenario for j in out_jobs}
    named, unknown_words, replaced = [], Counter(), 0
    for log in logs:
        parsed = classify(log.words, stage_steps, models, scenarios)
        if parsed is None and run_words(log.run):  # e.g. simulate_geant4_<stamp>/
            parsed = classify(run_words(log.run) + log.words, stage_steps, models, scenarios)
        if parsed is None:
            unknown_words[log.words[0] if log.words else log.path.name] += 1
            continue
        log.step, log.model, log.scenario, log.config = parsed
        if log.step in OUTPUT_STEPS.values():
            replaced += 1  # timed from the log next to its output instead
            continue
        named.append(log)
    # A name without a scenario: an observables or plot job covers every scenario
    # its run measures (plan.json); otherwise go by the other logs of the run.
    run_scenarios = defaultdict(set)
    for log in named:
        if log.scenario not in (None, ALL):
            run_scenarios[log.run_dir].add(log.scenario)
    for log in named:
        if log.scenario is None:
            found = measured_scenarios(log.run_dir) if log.step in GATHERING_STEPS else None
            found = set(found or run_scenarios[log.run_dir])
            log.scenario = next(iter(found)) if len(found) == 1 else ALL if found else UNKNOWN

    # 3. Time of each other job, from its ddml-bench lines.
    for log in named:
        attempts, excluded = read_attempts(log.path, excludes)
        if excluded:
            skip("matched --exclude", log)
        elif not attempts:
            skip("no 'ddml-bench: job ... on ...' start line", log)
        for start, end in attempts:
            if end is None:
                skip("no 'finished' line (still running, killed or timed out)", log)
                continue
            status = int(end["status"])
            if status != 0 and not args.include_failed:
                skip("finished with a non-zero status", log)
                continue
            t0, t1 = parse_time(start["time"]), parse_time(end["time"])
            seconds, t1 = duration(t0, t1)
            if seconds < 0:
                skip("finish time before start time", log)
                continue
            jobs.append(Job(log, seconds, status, t0, t1, start["host"], start["job"],
                            start["task"].strip()))

    # 4. Only the models described now, and only the jobs started since --since.
    kept = []
    for job in jobs:
        if allowed is not None and job.log.model and job.log.model not in allowed:
            skip(f"model {job.log.model} has no description in models/ now", job.log)
        elif since is not None and job.start < since:
            skip(f"started before {since:%Y-%m-%d %H:%M} (--since)", job.log)
        else:
            kept.append(job)
    jobs = kept
    out_jobs = [j for j in jobs if j.log.step in OUTPUT_STEPS.values()]

    n_runs = len({log.run_dir for log in logs})
    n_slurm = len(jobs) - len(out_jobs)
    print(f"{n_out_logs} simulation/reconstruction logs + {len(logs)} slurm logs in "
          f"{n_runs} run directories -> {len(jobs)} timed jobs")
    if replaced:
        print(f"  not used {replaced:4d}: slurm logs of simulation/reconstruction jobs "
              "(timed from the logs next to their outputs)")
    for reason, n in skipped.most_common():
        print(f"  skipped  {n:4d}: {reason}, e.g. {example[reason]}")
    if unknown_words:
        words = ", ".join(f"{w} ({n})" for w, n in unknown_words.most_common())
        print(f"  skipped  {sum(unknown_words.values()):4d}: name starts with an unknown word: "
              f"{words}\n                  (map one to a step with --stage WORD=STEP)")
    if not jobs:
        sys.exit("No finished jobs to plot.")

    # 5. Pool, report and plot.
    points = summarise(jobs, args.error)
    unit = auto_unit(points) if args.unit == "auto" else args.unit
    model_style = styles(sorted({p.model for p in points}, key=model_rank), MODELS,
                         SPARE_COLOURS)
    model_style = {m: ((labels or {}).get(m, label), colour)
                   for m, (label, colour) in model_style.items()}
    scenario_style = styles(sorted({p.scenario for p in points}, key=scenario_rank),
                            {**SCENARIOS, **PSEUDO_SCENARIOS}, SPARE_MARKERS)
    print_table(points, unit, model_style, scenario_style)
    if args.csv:
        write_csv(jobs, args.csv)
    runs = len({j.log.run_dir for j in jobs if j.log.step not in OUTPUT_STEPS.values()})
    source = (f"steps 2-3: {len(out_jobs)} output log{'s' if len(out_jobs) != 1 else ''}; "
              f"steps 1, 4-5: {n_slurm} slurm job{'s' if n_slurm != 1 else ''} "
              f"from {runs} run{'s' if runs != 1 else ''}")
    plot(points, unit, model_style, scenario_style, args, source)


if __name__ == "__main__":
    main()
