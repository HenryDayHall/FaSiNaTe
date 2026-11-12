"""The Pareto fronts of accuracy against time per event: of each scenario, on
each kind of node the timing benchmark timed it on (docs/timing.md), every
metric of the scenario on the same axes, in two versions (VERSIONS), by how
the distributions' accuracy is measured: by the Jensen-Shannon distance, or by
the Wasserstein distance. The figures that aren't of distributions, the energy
figures and the pi0 categories, are measured by their chi^2
(plot.chi_squared()) in both.

plot.py draws them after the other figures (plot.main()), whenever there is a
timing table, <paths.data_dir>/timing/summary.csv (timing_summary.sh makes
it), or the one --timing gives: pareto_<version>_<node>__<scenario><sample>,
e.g. pareto_js_cpu__single_photon and pareto_wasserstein_cpu__single_photon.
Each model is drawn in its colour at its time per event on those nodes, a
point for each metric of the scenario (a figure of plot.py, e.g. the radial
profile r30) at its distance from Geant4 there, each metric with a marker of
its own. Geant4 is drawn likewise, at its own time and at its own distance:
that of its second sample (layout.SECOND_GEANT4, geant4_2: every input
simulated with Geant4 again, with other seeds), which plot.py compares with
Geant4 as it does the models. So Geant4's point is how far apart two Geant4
samples of the same events are, the floor that the models' distances are
read against, and it is drawn as the second sample is, nearly black;
where the second sample hasn't measured a metric, Geant4 isn't on its front.
The time is the table's row of the scenario over all its inputs (input "all",
or of several numbers of events timed the one of the most events, all_t<N>):
the mean of the inputs' mean wall-clock times per event, each input weighing
the same, with its standard error. So every metric of a scenario has the same
times, its di-photon energies too. Geant4 is timed on the CPU nodes only
(unless timing.devices says otherwise), so where it wasn't timed on the
figure's kind of node, it is drawn at its time on the CPU nodes, as the
table's speed-ups are, and its legend entry says so. (Its second sample isn't
timed: it is Geant4, so its time is Geant4's.)

The distance is, in the js version, the Jensen-Shannon distance, the square
root of the divergence in metrics.csv (JS; a metric in the mathematical sense,
which the divergence isn't); in the wasserstein version, the Wasserstein
distance in metrics.csv (W1, plot.wasserstein(), in the units of the figure's
bins); and in both, of the energy figures and the pi0 categories, the chi^2 in
metrics.csv (chi2, over the figure's points, with the uncertainties drawn; no
distance in the mathematical sense, but growing as a simulator is farther
from Geant4). Each metric's distances, Geant4's own too, are divided by their
mean over its models: every model with that metric in metrics.csv, timed or
not (not Geant4's second sample). So the models' mean is at 1 for every
metric, which puts them all on one axis, and the figures of the CPU nodes and
of the GPU nodes give each the same values. Every figure of a version also
has the same range of distance, so that they can be compared side by side.
(The square root and the division move the points, but don't change which are
on the front.) Where metrics.csv gives a metric an uncertainty (the spread
over the seeds, for the separation and the pi0s), it is carried to the
distance (to first order, sigma / (2 sqrt(divergence)), for the Jensen-Shannon
distance) and divided by the same mean.

Each metric has its Pareto front: a point, Geant4's or a model's, is on it
when no other point of that metric is as fast and as close to Geant4 and
either faster or closer, by their central values. Those on it are filled, the
others hollow; Geant4 is off a front only where a model faster than it is as
close to it as its own second sample. The legend gives each model's speed-up
on Geant4 as drawn (Geant4's time over the model's), and each metric's marker,
with the models' mean distance in it (in the units of the bins, for a
Wasserstein distance). Everything drawn is in pareto<sample>.csv: a row per
point of each metric of each figure of each version, with the distance and
the time, their uncertainties, and whether it is on its metric's front;
Geant4's rows have its second sample's distances, at its own time.

The closure tests, which compare a model with its twin, have no front.
"""

import csv
import math
import os
import re
import types

from . import plot
from .dashboard import DATA
from .layout import figure_base, second_sample_of
from .models import SECOND_GEANT4_STYLE

NAME = "pareto"


def js_distance(divergence, uncertainty=None):
    """The Jensen-Shannon distance, the square root of the divergence, and its
    uncertainty from the divergence's, to first order (sigma / (2 distance));
    None where it isn't known."""
    distance = math.sqrt(max(divergence, 0.0))
    if uncertainty is None or distance <= 0:
        return distance, None
    return distance, uncertainty / (2 * distance)


def as_distance(value, uncertainty=None):
    """A metric taken as a distance as it is, and its uncertainty."""
    return value, uncertainty


# The versions of the fronts: of each, its name (in the figures' names), and
# the metrics of metrics.csv that measure a figure's accuracy in it, the first
# of them the figure has (a figure has JS and W1 if it is of a distribution,
# else chi2): of each, its name in metrics.csv, the distance from its value
# and uncertainty, and whether that is in the units of the figure's bins
VERSIONS = (
    ("js", (("JS", js_distance, False), (plot.CHI2, as_distance, False))),
    ("wasserstein", ((plot.WASSERSTEIN, as_distance, True), (plot.CHI2, as_distance, False))),
)
# How the axis and the legend name each metric's distance
DISTANCE_NAMES = {"JS": "JS distance", plot.WASSERSTEIN: "Wasserstein distance",
                  plot.CHI2: r"$\chi^2$"}
# The markers of a scenario's metrics, in the order of its figures in
# metrics.csv, as different from each other as can be
METRIC_MARKERS = ("s", "^", "D", "P", "X", "v", "p", "h", "*", "<", ">", "d")
MARKER_SIZE = 7.0
# Markers that look small at the same size, made bigger to look the same
MARKER_SCALE = {"*": 1.5, "p": 1.15, "h": 1.15, "P": 1.05, "X": 1.05}
# How far apart, in points, a simulator's metrics are drawn about its time,
# so that its points and their error bars don't hide each other (the time is
# the same for all of them)
DODGE = 3.5
# The figures are wider than plot.py's, for the legend beside the axes
FIGURE_SIZE = (7.6, 4.8)
# The timing table's rows of a scenario over all its inputs: "all", or of each
# number of events timed, where there are several, all_t<N>
ALL_INPUTS = re.compile(r"^all(_t\d+)?$")
# The CPU nodes, where Geant4 is timed (timing.devices in the configuration)
CPU = "cpu"
# What a kind of node can be called, to be in a figure's name
_NODE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+-]*$")
COLUMNS = ("version", "node", "scenario", "observable", "profile", "simulator", "time_s",
           "time_error_s", "timed_on", "metric", "value", "distance", "distance_error", "unit",
           "relative_distance", "relative_distance_error", "mean_distance", "models", "speedup",
           "on_front")


class TimingError(Exception):
    pass


def number(text):
    """A number of a CSV file, or None for an empty or a non-finite one."""
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def read_timing(path):
    """{(node, scenario, simulator): time} from the timing benchmark's table
    (ddml_timing/summary.py): of each kind of node, scenario and simulator, its
    row over all the scenario's inputs (all, or all_t<N> of the most events), a
    dict of "time" and "error", its mean time per event and the standard error
    of that, in seconds, and "events", the events timed."""
    times = {}
    with open(path, newline="") as stream:
        reader = csv.DictReader(stream)
        wanted = ("device", "scenario", "simulator", "input", "events", "mean_s", "sem_s")
        missing = [column for column in wanted if column not in (reader.fieldnames or ())]
        if missing:
            raise TimingError("%s has no column %s, so it isn't the timing benchmark's table"
                              % (path, ", ".join(missing)))
        for row in reader:
            if not ALL_INPUTS.match(row["input"] or ""):
                continue
            time = number(row["mean_s"])
            if time is None or time <= 0:
                continue
            key = (row["device"], row["scenario"], row["simulator"])
            entry = {"time": time, "error": number(row["sem_s"]),
                     "events": int(number(row["events"]) or 0)}
            if key not in times or entry["events"] > times[key]["events"]:
                times[key] = entry
    return times


def read_metrics(path):
    """{(observable, scenario, profile): {metric: {simulator: (value,
    uncertainty or None)}}} of metrics.csv, the figures in their order."""
    groups = {}
    with open(path, newline="") as stream:
        for row in csv.DictReader(stream):
            value = number(row.get("value"))
            if value is None:
                continue
            key = (row["observable"], row["scenario"], row["profile"])
            groups.setdefault(key, {}).setdefault(row["metric"], {})[row["simulator"]] = (
                value, number(row.get("uncertainty")))
    return groups


def relative_distances(values, to_distance=js_distance, leave_out=()):
    """({simulator: (value, distance, its uncertainty, distance / mean, its
    uncertainty)}, mean) of one figure's metrics, {simulator: (value,
    uncertainty)}: each simulator's distance (to_distance(), of the metric's
    value and uncertainty), and divided by the mean of the models' (all of
    them but those left out, e.g. the reference's second sample; taken as
    exact). ({}, mean) if the mean isn't above 0."""
    distances = dict((simulator, (value,) + tuple(to_distance(value, uncertainty)))
                     for simulator, (value, uncertainty) in values.items())
    models = [d[1] for simulator, d in distances.items() if simulator not in leave_out]
    mean = sum(models) / len(models) if models else 0.0
    if not mean > 0:
        return {}, mean
    return (dict((simulator, (value, distance, error, distance / mean,
                              None if error is None else error / mean))
                 for simulator, (value, distance, error) in distances.items()), mean)


def pareto_front(points):
    """Whether each point, (time, distance), is on the Pareto front: no other
    point is as fast and as close, and faster or closer."""
    return [not any(t <= time and d <= distance and (t < time or d < distance)
                    for t, d in points)
            for time, distance in points]


def speedup_text(speedup):
    """A speed-up on Geant4 as the legend says it: ×47.4, ×0.706."""
    return "×%.3g" % speedup


def number_text(value):
    """A distance as the figures say it: a chi^2 of thousands in full."""
    return "%.0f" % value if value >= 1000 else "%.3g" % value


def style_of(name, simulators, default=None):
    """The simulator of that name among those drawn (with its label, colour and
    line style), or a stand-in with the default style ({label, color,
    linestyle}), or plot's."""
    found = next((s for s in simulators if s.name == name), None)
    style = dict({"label": name, "color": "#1f6fb4", "linestyle": "-"}, **(default or {}))
    return found or types.SimpleNamespace(name=name, **style)


def node_times(times, node, scenario, reference_name):
    """(the models' {simulator: time} on a kind of node for a scenario, and the
    reference's (time, the kind of node it was timed on) or None): the
    reference's on the same kind of node, or else on the CPU nodes, or any."""
    models = dict((simulator, time) for (n, s, simulator), time in times.items()
                  if n == node and s == scenario and simulator != reference_name)
    timed = dict((n, time) for (n, s, simulator), time in times.items()
                 if s == scenario and simulator == reference_name)
    for wanted in [node, CPU] + sorted(timed):
        if wanted in timed:
            return models, (timed[wanted], wanted)
    return models, None


def metric_label(metric, version_metric):
    """How the legend names one of a figure's metrics: its observable and
    profile, and its distance, if not the version's."""
    label = "%s %s" % (metric["observable"].replace("_", " "), metric["profile"])
    if metric["metric"] != version_metric:
        label += " (%s)" % DISTANCE_NAMES.get(metric["metric"], metric["metric"])
    unit = metric["unit"]
    return "%s: %s%s" % (label, number_text(metric["mean"]),
                         (" " + unit) if unit and unit != "°" else unit)


def plan_fronts(groups, times, simulators, reference_name, version=VERSIONS[0]):
    """The figures of one version (of VERSIONS): a dict for each scenario of
    groups (read_metrics()) and kind of node its models are timed on, with the
    scenario, the node, the reference, its time on those nodes or the CPU
    nodes ("reference_time", (time, where) or None), the simulator its points
    are drawn as ("own": its second sample, whose distances they are, or
    None), the models drawn (those timed there, in their order), and its
    metrics: dicts of the figure (observable, profile), the metric of
    metrics.csv it is measured by, the unit of its distance, the models' mean
    distance and their number, its marker, and its points: dicts of name (the
    simulator it is the point of), simulator (the one it is drawn as),
    whether it is the reference's, time, its error, the node it was timed on,
    the metric's value, the distance and the relative distance and their
    uncertainties, and whether it is on the metric's front (the reference's
    first, if it is timed, at its second sample's distance, if that is
    measured)."""
    name, measures = version
    reference = style_of(reference_name, simulators)
    # The reference's own distance, its second sample's (layout.SECOND_GEANT4),
    # which isn't one of the models
    second_name = second_sample_of(reference_name)
    own = style_of(second_name, simulators, SECOND_GEANT4_STYLE) if second_name else None
    not_models = (reference_name, second_name)
    order = dict((s.name, k) for k, s in enumerate(
        s for s in simulators if s.name not in not_models))
    scenarios, untimed = {}, []
    for (observable, scenario, profile), metrics in groups.items():
        measure = next((m for m in measures if m[0] in metrics), None)
        if measure is None:
            continue
        metric, to_distance, in_units = measure
        distances, mean = relative_distances(metrics[metric], to_distance, not_models)
        if not distances:
            plot.say("%s %s of %s: the models' mean %s isn't above 0, so it has no Pareto "
                     "front" % (observable, profile, scenario, metric))
            continue
        scenarios.setdefault(scenario, []).append({
            "observable": observable, "profile": profile, "metric": metric,
            "unit": plot.unit_of(observable, profile) if in_units else "", "mean": mean,
            "models": len([n for n in distances if n not in not_models]),
            "distances": distances})
    if second_name is None:
        plot.say("%s has no second sample, so its own distance isn't known, and it isn't on the "
                 "%s Pareto fronts" % (reference.label, name))
    elif scenarios and not any(second_name in m["distances"] for metrics in scenarios.values()
                               for m in metrics):
        plot.say("there are no measurements of %s, %s's second sample, whose distance is %s's "
                 "own, so %s isn't on the %s Pareto fronts" % (
                     second_name, reference.label, reference.label, reference.label, name))
    planned = []
    for scenario, metrics in scenarios.items():
        nodes = sorted({n for (n, s, simulator) in times if s == scenario
                        and any(simulator in m["distances"] for m in metrics)
                        and simulator not in not_models})
        # What the second sample hasn't measured, where it has measured
        # something (else it was said above)
        lacking = [m for m in metrics if second_name not in m["distances"]]
        if second_name and lacking and nodes and any(
                second_name in m["distances"] for ms in scenarios.values() for m in ms):
            plot.say("%s has no %s of %s, so %s isn't on %s Pareto front%s" % (
                second_name, ", ".join("%s %s" % (m["observable"], m["profile"])
                                       for m in lacking), scenario, reference.label,
                "its" if len(lacking) == 1 else "their", "" if len(lacking) == 1 else "s"))
        if not nodes and scenario not in untimed:
            untimed.append(scenario)
            plot.say("none of the models of %s is timed, so it has no %s Pareto fronts"
                     % (scenario, name))
        for node in nodes:
            if not _NODE.match(node):
                plot.say("the kind of node %r can't be in a figure's name, so it has no Pareto "
                         "fronts" % node)
                continue
            model_times, reference_time = node_times(times, node, scenario, reference_name)
            drawn = [style_of(n, simulators) for n in sorted(
                {n for m in metrics for n in m["distances"]
                 if n in model_times and n not in not_models},
                key=lambda n: (order.get(n, len(order)), n))]
            figure_metrics = []
            for k, metric in enumerate(metrics):
                points = []
                if reference_time is not None and second_name in metric["distances"]:
                    # The reference at its time, at its second sample's distance
                    time, timed_on = reference_time
                    points.append(point(reference_name, own, True, time, timed_on,
                                        metric["distances"][second_name]))
                for simulator in drawn:
                    if simulator.name in metric["distances"]:
                        points.append(point(simulator.name, simulator, False,
                                            model_times[simulator.name], node,
                                            metric["distances"][simulator.name]))
                if all(p["reference"] for p in points):
                    continue
                for p, on in zip(points, pareto_front([(p["time"], p["relative"])
                                                       for p in points])):
                    p["on_front"] = on
                shown = dict((key, value) for key, value in metric.items()
                             if key != "distances")
                shown.update(marker=METRIC_MARKERS[k % len(METRIC_MARKERS)], points=points)
                figure_metrics.append(shown)
            planned.append({"version": version, "scenario": scenario, "node": node,
                            "reference": reference, "reference_time": reference_time,
                            "own": own, "drawn": drawn, "metrics": figure_metrics})
    return planned


def point(name, simulator, is_reference, time, timed_on, distances):
    """A point of a metric's front: of the simulator named, drawn as simulator,
    at its time (from read_timing()) on the node it was timed on, at its
    distances (relative_distances()' (value, distance, its uncertainty,
    relative distance, its uncertainty))."""
    value, distance, error, relative, relative_error = distances
    return {"name": name, "simulator": simulator, "reference": is_reference,
            "time": time["time"], "time_error": time["error"], "timed_on": timed_on,
            "value": value, "distance": distance, "distance_error": error,
            "relative": relative, "relative_error": relative_error}


def fronts(out, suffix, formats, simulators, reference_name, timing_path, metrics_path=None):
    """Draw the Pareto fronts of every version (VERSIONS) of every scenario in
    metrics<suffix>.csv in out (or metrics_path), for every kind of node the
    timing table at timing_path has the scenario's models timed on, into out,
    and write pareto<suffix>.csv there; simulators are those drawn (for their
    labels and styles, the reference first, then the models in their order).
    Returns the rows of pareto<suffix>.csv ([] if no front is drawn)."""
    metrics_path = metrics_path or os.path.join(out, "metrics%s.csv" % suffix)
    if not os.path.isfile(timing_path):
        plot.say("there is no timing table (%s), so no Pareto fronts; ./timing_summary.sh makes "
                 "it from the timing benchmark's results" % timing_path)
        return []
    if not os.path.isfile(metrics_path):
        plot.say("there are no metrics (%s), so no Pareto fronts" % metrics_path)
        return []
    try:
        times = read_timing(timing_path)
    except (OSError, TimingError) as error:
        plot.say("%s, so no Pareto fronts" % error)
        return []
    groups = read_metrics(metrics_path)
    rows = []
    for version in VERSIONS:
        planned = [figure for figure in plan_fronts(groups, times, simulators, reference_name,
                                                    version) if figure["metrics"]]
        if not planned:
            plot.say("no metric of a model timed in %s, so there are no %s Pareto fronts"
                     % (timing_path, version[0]))
            continue
        # Every figure's axis of distance the same: up to the farthest point,
        # and a little below 0, so that Geant4's marker isn't cut in half
        farthest = max(p["relative"] + (p["relative_error"] or 0) for figure in planned
                       for metric in figure["metrics"] for p in metric["points"]) or 1.0
        top = farthest * 1.08
        if plot.collector is not None:
            plot.collector.section(out, planned[0]["reference"])
        for figure in planned:
            plot_fronts(figure, (-0.03 * top, top), out, suffix, formats)
            rows += figure_rows(figure)
    if not rows:
        return []
    path = os.path.join(out, "%s%s.csv" % (NAME, suffix))
    with open(path, "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(COLUMNS)
        for row in rows:
            writer.writerow([_cell(row[column]) for column in COLUMNS])
    for row in rows:
        print("%-11s %-6s %-20s %-13s %-10s %-23s %8.3g s, distance %.3g%s" % (
            row["version"], row["node"], row["observable"], row["scenario"], row["profile"],
            row["simulator"], row["time_s"], row["relative_distance"],
            ", on the front" if row["on_front"] else ""))
    plot.say("Pareto fronts, and %s, in %s" % (os.path.basename(path), out))
    return rows


def figure_rows(figure):
    """The rows of pareto.csv of a figure: one per point of each metric."""
    reference_time = figure["reference_time"][0]["time"] if figure["reference_time"] else None
    rows = []
    for metric in figure["metrics"]:
        for p in metric["points"]:
            is_reference = p["reference"]
            rows.append({
                "version": figure["version"][0], "node": figure["node"],
                "scenario": figure["scenario"], "observable": metric["observable"],
                "profile": metric["profile"], "simulator": p["name"],
                "time_s": p["time"], "time_error_s": p["time_error"], "timed_on": p["timed_on"],
                "metric": metric["metric"], "value": p["value"], "distance": p["distance"],
                "distance_error": p["distance_error"], "unit": metric["unit"],
                "relative_distance": p["relative"],
                "relative_distance_error": p["relative_error"],
                "mean_distance": metric["mean"], "models": metric["models"],
                "speedup": None if is_reference or reference_time is None
                else reference_time / p["time"],
                "on_front": p["on_front"]})
    return rows


def _cell(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return "%.6g" % value
    return value


def plot_fronts(figure, ylim, out, suffix, formats):
    """One figure of plan_fronts(): every metric's points of a scenario on one
    kind of node, with the distance from ylim[0] to ylim[1]."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    from matplotlib.transforms import offset_copy

    (name, measures), reference = figure["version"], figure["reference"]
    reference_time = figure["reference_time"]
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    axes = [[ax]]
    every = [p for metric in figure["metrics"] for p in metric["points"]]
    count = len(figure["metrics"])
    for k, metric in enumerate(figure["metrics"]):
        # Each metric a little to the side of the others, in points
        shifted = offset_copy(ax.transData, fig=fig, x=(k - (count - 1) / 2) * DODGE,
                              units="points")
        size = MARKER_SIZE * MARKER_SCALE.get(metric["marker"], 1.0)
        for p in metric["points"]:
            # On its metric's front: filled, with a white ring; off it: hollow.
            # (The reference's points are drawn as its second sample, whose
            # distances they are.)
            simulator, on = p["simulator"], p["on_front"]
            drawn = ax.errorbar(
                [p["time"]], [p["relative"]],
                xerr=None if p["time_error"] is None else [p["time_error"]],
                yerr=None if p["relative_error"] is None else [p["relative_error"]],
                linestyle="none", marker=metric["marker"], markersize=size,
                color=simulator.color, markerfacecolor=simulator.color if on else "white",
                markeredgecolor="white" if on else simulator.color,
                markeredgewidth=0.9 if on else 1.4, elinewidth=1.0, capsize=0, zorder=5,
                transform=shifted)
            plot.mark(simulator, DATA, drawn)

    ax.set_xscale("log")
    # The times, with their errors, and a margin on either side of a sixth of
    # the decades they span, or at least 0.15 of a decade
    low = min(p["time"] - p["time_error"] if p["time_error"] and p["time_error"] < p["time"]
              else p["time"] for p in every)
    high = max(p["time"] + (p["time_error"] or 0) for p in every)
    margin = 10 ** max(math.log10(high / low) / 6, 0.15)
    low, high = low / margin, high * margin
    ax.set_xlim(low, high)
    # Ticks that say the time in seconds (0.05, 1, ...), as many as fit
    decades = math.log10(high / low)
    subs = ((1.0, 2.0, 3.0, 5.0, 7.0) if decades <= 1.3 else (1.0, 2.0, 5.0) if decades <= 2.5
            else (1.0, 3.0) if decades <= 4 else (1.0,))
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=subs))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: "%g" % value))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(*ylim)
    ax.set_xlabel("Time per event on the %s nodes [s]" % figure["node"], color=plot.INK)
    shown = [m[0] for m in measures if any(metric["metric"] == m[0]
                                           for metric in figure["metrics"])]
    ax.set_ylabel("%s to %s / the models' mean" % (
        " or ".join(DISTANCE_NAMES.get(m, m) for m in shown), reference.label), color=plot.INK)

    # The legend, beside the axes: the reference's colour (its second
    # sample's) and the models', then the metrics' markers
    entries = []
    if any(p["reference"] for p in every):
        own = figure["own"]
        label = own.label
        if reference_time[1] != figure["node"]:
            label += ", timed on %s nodes" % reference_time[1]
        entries.append((Line2D([], [], color=own.color, linewidth=5, solid_capstyle="butt"),
                        label, own))
    for simulator in figure["drawn"]:
        label = simulator.label
        mine = next((p for metric in figure["metrics"] for p in metric["points"]
                     if p["simulator"] is simulator), None)
        if reference_time is not None and mine is not None:
            label += ", %s" % speedup_text(reference_time[0]["time"] / mine["time"])
        entries.append((Line2D([], [], color=simulator.color, linewidth=5,
                               solid_capstyle="butt"), label, simulator))
    entries.append((Line2D([], [], linestyle="none"), "Metrics (the models' mean):", None))
    for metric in figure["metrics"]:
        entries.append((Line2D([], [], linestyle="none", marker=metric["marker"],
                               markersize=MARKER_SIZE * MARKER_SCALE.get(metric["marker"], 1.0),
                               color=plot.MUTED, markeredgecolor="white", markeredgewidth=0.9),
                        metric_label(metric, measures[0][0]), None))
    plot.legend(ax, entries, fontsize=7.5, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0.0)
    note = "%s, %s nodes\nhollow: off its metric's Pareto front\n%s" % (
        figure["scenario"], figure["node"],
        "%s isn't timed" % reference.label if reference_time is None
        else "×N: speed-up on %s" % reference.label)
    if count > 1:
        note += "\na simulator's metrics side by side\nabout its time"
    if any(p["relative_error"] is not None for p in every):
        note += "\nerrors: the metrics' uncertainties"
    ax.text(1.02, 0.0, note, transform=ax.transAxes, fontsize=8, color=plot.MUTED,
            va="bottom", ha="left")
    figure_name = "%s_%s" % (NAME, name)
    if plot.collector is not None:
        plot.collector.context(figure_name, figure["scenario"])
    base = figure_base(out, figure_name, figure["node"], figure["scenario"], suffix)
    plot.finish(fig, axes, base, formats)
    plot.say("%s: %s" % (", ".join([reference.label] + [s.label for s in figure["drawn"]]),
                         ", ".join("%s.%s" % (os.path.basename(base), f) for f in formats)))
