"""The low-cost mode: models tested on the first events of each input, against
the full samples of Geant4, its second sample and the other models, as
published, so that only those models are simulated and reconstructed, and on
a part of the events. It runs anywhere the pipeline does, and on GitHub's
runners for a model pushed to a fork (ddml_bench/github.py, which splits it
into the jobs of .github/workflows/low-cost.yml):

  ddml-bench low-cost [options] MODEL ... [-- SBATCH_OPTION ...]   (./low_cost.sh)

A run, in order (plan_run()):

 1. Reads the manifests of the published records (published in the
    configuration), and checks that the software the published measurements
    were made with is what the configuration pins (key4hep.release,
    ddml.commit, detector_configs.ildconfig.commit): a model simulated with
    other software isn't compared like with like, so it stops, saying what
    to pin in configs/local.yaml (--any-stack goes on, with a warning).
 2. Fetches what the full benchmark would fetch (ddml_bench/fetch.py): the
    inputs, the full samples' measurements (observables/<simulator>.npz) that
    are what this configuration would make, and the timing table, each where
    it isn't there yet. A data folder whose full samples are measured already
    (e.g. by the full benchmark on a cluster) fetches none of them, and
    --no-fetch compares with what is there.
 3. Checks that the reference, Geant4, has its full measurements (here, or
    to be fetched), and says which others there are to compare with.
 4. Prints what the sample costs, estimated (estimate(), low_cost.estimate).
 5. Plans the models' tasks of the sample, the first N events of each input
    of each scenario (low_cost.events; -n N for every scenario): simulating
    them, reconstructing them and measuring them, into
    observables/<model><sample>.npz (layout.sample_suffix(): _n<N>, and
    _<scenario><M> for a scenario with another number), and drawing them
    against the full samples of every other simulator, into
    observables/plots/low_cost<sample>/ (the plot stage, ddml_bench/plot.py's
    --sample), with the statistical floor of each metric at the sample's
    size (ddml_bench/floors.py). The full sample's own plots aren't touched.

ddml_bench/cli.py runs the plan with the local executor, or as slurm jobs.
Everything here runs where the plan is made, with the standard library (and
PyYAML for the configuration).
"""

import math
import os
import sys

from .config import ConfigError, load_config
from .graph import PlanError, make_plan, scenario_events
from .layout import GEANT4, SECOND_GEANT4, TIMING_DIR, TIMING_SUMMARY, Layout, sample_suffix
from .models import model_names


class LowCostError(Exception):
    pass


def say(message):
    print("ddml-bench: " + message)
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# The sample


def enabled_scenarios(config):
    """([the scenarios enabled], [those disabled]), in the configuration's
    order."""
    enabled, disabled = [], []
    for name, settings in (config.get("scenarios") or {}).items():
        on = settings.get("enabled", True) if isinstance(settings, dict) else True
        (enabled if on is not False else disabled).append(name)
    return enabled, disabled


def configured_events(config, events=None):
    """{scenario: N} of the low-cost sample, of each scenario enabled: N of
    every scenario with events, else the configuration's low_cost.events."""
    enabled, disabled = enabled_scenarios(config)
    if events is not None:
        return dict((name, events) for name in enabled)
    numbers = (config.get("low_cost") or {}).get("events")
    if not isinstance(numbers, dict):
        raise ConfigError("low_cost.events should be the number of events of each input of each "
                          "scenario, e.g. single_photon: 1000")
    try:
        return scenario_events(numbers, enabled, disabled)
    except PlanError as error:
        raise ConfigError(str(error))


def events_text(numbers):
    """{scenario: N} in words: the first 1000 events of each input, and 500
    of tau_pairs's."""
    from .layout import most_common
    common = most_common(numbers)
    rest = ["%d of %s's" % (number, scenario) for scenario, number in numbers.items()
            if number != common]
    return "the first %d events of each input%s" % (
        common, (", and " + ", ".join(rest)) if rest else "")


def configured_suffix(config=None):
    """The suffix of the configuration's low-cost sample (layout.sample_suffix())."""
    return sample_suffix(configured_events(config or load_config()))


# ---------------------------------------------------------------------------
# What it is compared with


def check_stack(settings, any_stack=False):
    """Stop (LowCostError) if the configuration doesn't pin the software the
    published measurements were made with (settings: a plan's published),
    saying what to pin; with any_stack, say so and go on. Nothing published,
    nothing checked."""
    from .fetch import read_manifests, stack_differences
    if not settings.get("enabled") or not settings.get("records"):
        return []
    differ = stack_differences(read_manifests(settings), settings.get("pins") or {})
    if not differ:
        return []
    listed = "; ".join('%s: "%s" (here %s)' % (setting, theirs, mine or "not pinned")
                       for setting, theirs, mine in differ)
    if any_stack:
        say("warning: the published measurements were made with other software than this "
            "configuration pins (%s), so the models are compared with them as they are "
            "(--any-stack)" % listed)
        return differ
    raise LowCostError(
        "the published measurements were made with other software than this configuration "
        "pins (%s), so a model simulated here wouldn't be compared like with like. Pin theirs "
        "in configs/local.yaml (the next run builds DDML again at that commit, if needed):\n\n"
        "%s\n(or run with --any-stack to compare with them anyway)" % (listed, pins_yaml(differ)))


def pins_yaml(differ):
    """The settings of stack_differences() as YAML, to paste into
    configs/local.yaml."""
    tree = {}
    for setting, theirs, _ in differ:
        node = tree
        parts = setting.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = theirs
    lines = []

    def write(node, depth):
        for key, value in node.items():
            if isinstance(value, dict):
                lines.append("%s%s:" % ("  " * depth, key))
                write(value, depth + 1)
            else:
                lines.append('%s%s: "%s"' % ("  " * depth, key, value))
    write(tree, 0)
    return "\n".join(lines) + "\n"


def full_measurements(full):
    """[(simulator, its full measurements, "here", "to fetch" or None)] of
    the full benchmark's plan (after fetch.prepare()): what the low-cost
    sample is compared with."""
    fetching = full.get("to_fetch") or {}
    layout = Layout(full["data_dir"])
    found = []
    for simulator in [full["reference"], full["second_geant4"]] + full["models"]:
        path = layout.observables(simulator["name"])
        state = "here" if os.path.isfile(path) else "to fetch" if path in fetching else None
        found.append((simulator["name"], path, state))
    return found


# ---------------------------------------------------------------------------
# What it costs


def estimate(plan, config):
    """[(scenario, simulations, events of each input, events, the
    simulation's seconds per event, where that is from, the
    reconstruction's, core-hours)] of one model's tasks of the plan's sample,
    estimated from the configuration's low_cost.estimate: the time per event
    of a published model that stands in for the model's (in the timing
    table, timing/summary.csv, on its kind of node, or else the configured
    one), the reconstruction's, and the overhead of each simulation."""
    settings = (config.get("low_cost") or {}).get("estimate") or {}
    stand_in, node = settings.get("stand_in"), settings.get("node") or "cpu"
    times = {}
    table = os.path.join(plan["data_dir"], TIMING_DIR, TIMING_SUMMARY)
    if stand_in and os.path.isfile(table):
        from .pareto import TimingError, read_timing
        try:
            times = read_timing(table)
        except (OSError, TimingError):
            times = {}
    layout = Layout(plan["data_dir"], plan["events"])
    overhead = float(settings.get("overhead_s") or 0)
    rows = []
    for group in plan["scenarios"]:
        scenario = group["scenario"]
        simulations = sum(len(entry["seeds"]) for entry in group["inputs"])
        each = layout.events_of(scenario)
        timed = times.get((node, scenario, stand_in))
        if timed:
            simulation, source = timed["time"], "timing table"
        else:
            simulation = _number((settings.get("simulation_s") or {}).get(scenario))
            source = "configured" if simulation is not None else "unknown"
        reconstruction = _number((settings.get("reconstruction_s") or {}).get(scenario))
        seconds = None
        if simulation is not None and reconstruction is not None:
            seconds = simulations * (each * (simulation + reconstruction) + overhead)
        rows.append((scenario, simulations, each, simulations * each, simulation, source,
                     reconstruction, None if seconds is None else seconds / 3600.0))
    return rows


def _number(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def estimate_lines(rows, config, models):
    """The estimate (estimate()) as the run prints it."""
    settings = (config.get("low_cost") or {}).get("estimate") or {}
    lines = ["what the sample costs, estimated for each model in core-hours of one core of the "
             "%s nodes (%s's time per event there stands in for the model's; low_cost.estimate):"
             % (settings.get("node") or "cpu", settings.get("stand_in") or "no model")]
    lines.append("  %-16s %11s %13s %9s %15s %15s %10s" % (
        "scenario", "simulations", "events/input", "events", "simulation s/ev",
        "reconstr. s/ev", "core-hours"))
    total = 0.0
    for scenario, simulations, each, events, simulation, source, reconstruction, hours in rows:
        lines.append("  %-16s %11d %13d %9d %15s %15s %10s" % (
            scenario, simulations, each, events,
            "%.3g (%s)" % (simulation, "table" if source == "timing table" else "config")
            if simulation is not None else "?",
            "%.3g" % reconstruction if reconstruction is not None else "?",
            "%.1f" % hours if hours is not None else "?"))
        total = total + hours if hours is not None and total is not None else None
    lines.append("  %-16s %11d %13s %9d %15s %15s %10s" % (
        "all", sum(row[1] for row in rows), "", sum(row[3] for row in rows), "", "",
        "%.1f" % total if total is not None else "?"))
    if len(models) > 1 and total is not None:
        lines.append("  for the %d models: %.1f core-hours" % (len(models), total * len(models)))
    return lines


# ---------------------------------------------------------------------------
# A run


def plan_run(models, events=None, force=(), slurm=False, dry_run=False, no_fetch=False,
             any_stack=False):
    """The plan of a low-cost run of the models (see the module docstring),
    after checking the software, fetching what is published of the full
    benchmark and saying what the sample is compared with and costs. Raises
    LowCostError, ConfigError, PlanError or FetchError."""
    from . import fetch
    config = load_config()
    names = model_names()
    if not models:
        raise LowCostError("give the models to test, by their names in models/: %s"
                           % (", ".join(names) or "it has none"))
    for model in models:
        if model in (GEANT4, SECOND_GEANT4):
            raise LowCostError("the low-cost mode tests models against the full samples of "
                               "Geant4 and the rest, so %s can't be one of them" % model)
        if model not in names:
            raise LowCostError("there is no model '%s'; models/ has: %s"
                               % (model, ", ".join(names) or "nothing"))
    numbers = configured_events(config, events)

    # The full benchmark, for what is published of it: its inputs, and the
    # measurements of the full samples that the models are compared with
    full = make_plan()
    if no_fetch:
        full["published"]["enabled"] = False
    check_stack(full["published"], any_stack)
    fetch.prepare(full, dry_run, note_stack=False)

    plan = make_plan(simulators=models, events=numbers, force=force, slurm=slurm,
                     low_cost=True)
    # (fetched above, as the full benchmark's)
    plan["published"]["enabled"] = False
    for key in ("to_fetch", "timing_to_fetch"):
        if full.get(key):
            plan[key] = full[key]

    found = full_measurements(full)
    reference = next(item for item in found if item[0] == GEANT4)
    if reference[2] is None:
        raise LowCostError("there are no full measurements of Geant4 (%s) to compare with, here or "
                           "published as this configuration would make them (see above); make "
                           "them (./slurm_pipeline.sh geant4) or fetch them (published)"
                           % reference[1])
    compared = ["%s (%s)" % (name, state) for name, _, state in found
                if state and name not in models]
    absent = [name for name, _, state in found if state is None and name not in models]
    layout = Layout(plan["data_dir"], plan["events"])
    say("low-cost run of %s: %s, against the full samples of %s; measured as "
        "observables/%s%s.npz, plotted in %s" % (
            ", ".join(models), events_text(numbers), ", ".join(compared),
            models[0] if len(models) == 1 else "<model>", layout.suffix,
            layout.relative(layout.low_cost_plots_dir)))
    if absent:
        say("%s %s no full measurements here, nor published as this configuration would make "
            "them, so %s drawn" % (", ".join(absent), "has" if len(absent) == 1 else "have",
                                   "it isn't" if len(absent) == 1 else "they aren't"))
    for line in estimate_lines(estimate(plan, config), config, models):
        say(line)
    return plan
