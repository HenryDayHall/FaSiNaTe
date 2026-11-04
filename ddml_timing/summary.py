"""The timing results gathered into one table (timing_summary.sh):
<paths.data_dir>/timing/summary.csv, and printed.

A row for each task's results: the mean wall-clock time per event over the
events timed, its standard deviation and standard error (sem, and rel_sem_pct,
the standard error as a percentage of the mean), the median, the mean CPU time
per event, the model's showers per event (DDML's line for each shower it places
in the barrel), the model's embedded-Python inference per event where it was
timed,
how many events a 10% standard error needs at the spread seen (events_for_10pct),
the speed-up on Geant4 (Geant4's mean time per event of the same input over
the model's, on the same kind of node, or else on the CPU nodes, with its
uncertainty from both standard errors), the node, and drift_z, the difference
between the means of the first and the second half of the events timed in
units of its uncertainty (more than about 3 says the times drift, e.g. a
warm-up longer than the events left out). Then a row for each kind of node,
scenario and simulator over all its inputs (input "all"): the mean of the
inputs' means, each input weighing the same (for the single photons, every
energy alike, as the benchmark paper's average over 10-100 GeV), with its
standard error from theirs, and the speed-up of that on Geant4's over the same
inputs.

This needs only the standard library, so the results can be copied anywhere
and tabulated there.
"""

import csv
import glob
import json
import math
import os
import sys

from .events import events_needed
from .plan import FAILED, FORMAT, RESULT

COLUMNS = ("device", "scenario", "simulator", "input", "events", "warmup", "mean_s", "sd_s",
           "sem_s", "rel_sem_pct", "median_s", "cpu_mean_s", "showers_per_event",
           "inference_per_event_s", "model_calls_per_event", "events_for_10pct", "speedup",
           "speedup_error", "speedup_reference", "inference_on", "host", "cpu_model", "gpu",
           "drift_z")
# The reference for the speed-ups
GEANT4 = "geant4"


def say(message):
    print("ddml-timing: " + message)


def gather(root):
    """(results, failures): the results of every task under root (the timing
    folder), and the failure records of those that failed last time."""
    results, failures = [], []
    pattern = os.path.join(glob.escape(root), "*", "*", "*", "*.json")
    for path in sorted(glob.glob(pattern)):
        name = os.path.basename(path)
        if name.startswith(".") or os.path.relpath(path, root).split(os.sep)[0] == "slurm":
            continue
        try:
            with open(path) as stream:
                data = json.load(stream)
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict) or data.get("format") != FORMAT:
            continue
        if name.endswith(FAILED):
            if not os.path.exists(path[:-len(FAILED)] + RESULT):
                failures.append(data)
        elif name.endswith(RESULT):
            data["_path"] = path
            results.append(data)
    return results, failures


def _input_name(result):
    return os.path.basename(result["_path"])[:-len(RESULT)]


def _short_names(names):
    """names without the words (separated by _) that all of them start and end
    with, where something is left of each."""
    split = [name.split("_") for name in names]
    if len(split) < 2:
        return dict((name, name) for name in names)
    start = 0
    while all(len(words) > start + 1 for words in split) \
            and len({words[start] for words in split}) == 1:
        start += 1
    end = 0
    while all(len(words) > start + end + 1 for words in split) \
            and len({words[-1 - end] for words in split}) == 1:
        end += 1
    return dict((name, "_".join(words[start:len(words) - end]) or name)
                for name, words in zip(names, split))


def _gpu(result):
    gpus = (result.get("node") or {}).get("gpus") or []
    return "; ".join(gpu.get("name", "?") for gpu in gpus)


def _row(result):
    wall, cpu = result.get("wall") or {}, result.get("cpu") or {}
    inference = result.get("inference_time") or {}
    calls = result.get("model_calls") or {}
    showers = result.get("barrel_showers") or {}
    # Calls are seen only of a model's embedded-Python inference, which the run timed
    timed_calls = bool((result.get("inference") or {}).get("models"))
    node = result.get("node") or {}
    return {
        "device": result["device"], "scenario": result["scenario"],
        "simulator": result["simulator"], "input": _input_name(result),
        "events": wall.get("n"), "warmup": (result.get("events") or {}).get("warmup"),
        "mean_s": wall.get("mean"), "sd_s": wall.get("sd"), "sem_s": wall.get("sem"),
        "rel_sem_pct": None if wall.get("rel_sem") is None else 100 * wall["rel_sem"],
        "median_s": wall.get("median"), "cpu_mean_s": cpu.get("mean"),
        "showers_per_event": showers.get("mean") if result["simulator"] != GEANT4 else None,
        "inference_per_event_s": inference.get("mean"),
        "model_calls_per_event": calls.get("mean") if timed_calls else None,
        "events_for_10pct": events_needed(wall.get("rel_sem"), wall.get("n")),
        "speedup": None, "speedup_error": None, "speedup_reference": None,
        "inference_on": (result.get("inference") or {}).get("on"),
        "host": node.get("host"), "cpu_model": node.get("cpu_model"), "gpu": _gpu(result),
        "drift_z": wall.get("drift_z"),
    }


def _ratio(numerator, numerator_sem, denominator, denominator_sem):
    if not numerator or not denominator:
        return None, None
    ratio = numerator / denominator
    parts = [s / v for s, v in ((numerator_sem, numerator), (denominator_sem, denominator))
             if s is not None]
    return ratio, ratio * math.sqrt(sum(p * p for p in parts)) if parts else None


def _reference(rows, row):
    """Geant4's row of the same input (or of all the inputs) on the same kind
    of node, or else on the cpu nodes, or any; None if there is none."""
    candidates = [r for r in rows if r["simulator"] == GEANT4 and r["scenario"] == row["scenario"]
                  and r["input"] == row["input"]]
    for wanted in (row["device"], "cpu"):
        for candidate in candidates:
            if candidate["device"] == wanted:
                return candidate
    return candidates[0] if candidates else None


def summarise(results):
    """The rows of the table: one per result, then one per kind of node,
    scenario and simulator over all its inputs."""
    rows = sorted((_row(result) for result in results),
                  key=lambda r: (r["device"], r["scenario"], r["simulator"] != GEANT4,
                                 r["simulator"], r["input"]))
    totals = []
    groups = {}
    for row in rows:
        groups.setdefault((row["device"], row["scenario"], row["simulator"]), []).append(row)
    for (device, scenario, simulator), mine in sorted(
            groups.items(), key=lambda item: (item[0][0], item[0][1], item[0][2] != GEANT4,
                                              item[0][2])):
        means = [r["mean_s"] for r in mine if r["mean_s"] is not None]
        if not means:
            continue
        sems = [r["sem_s"] for r in mine if r["mean_s"] is not None]
        mean = sum(means) / len(means)
        sem = math.sqrt(sum(s * s for s in sems)) / len(sems) if all(
            s is not None for s in sems) else None
        totals.append({
            "device": device, "scenario": scenario, "simulator": simulator,
            "input": "all", "inputs": [r["input"] for r in mine if r["mean_s"] is not None],
            "events": sum(r["events"] or 0 for r in mine), "warmup": mine[0]["warmup"],
            "mean_s": mean, "sd_s": None, "sem_s": sem,
            "rel_sem_pct": None if sem is None or not mean else 100 * sem / mean,
            "median_s": None,
            "cpu_mean_s": _mean([r["cpu_mean_s"] for r in mine]),
            "showers_per_event": _mean([r["showers_per_event"] for r in mine]),
            "inference_per_event_s": _mean([r["inference_per_event_s"] for r in mine]),
            "model_calls_per_event": _mean([r["model_calls_per_event"] for r in mine]),
            "events_for_10pct": None, "speedup": None, "speedup_error": None,
            "speedup_reference": None,
            "inference_on": "; ".join(sorted({str(r["inference_on"]) for r in mine})),
            "host": "; ".join(sorted({str(r["host"]) for r in mine})),
            "cpu_model": "; ".join(sorted({str(r["cpu_model"]) for r in mine})),
            "gpu": "; ".join(sorted({r["gpu"] for r in mine if r["gpu"]})),
            "drift_z": None,
        })
    for row in rows:
        reference = _reference(rows, row) if row["simulator"] != GEANT4 else None
        if reference is not None:
            row["speedup"], row["speedup_error"] = _ratio(
                reference["mean_s"], reference["sem_s"], row["mean_s"], row["sem_s"])
            row["speedup_reference"] = "geant4 on %s" % reference["device"]
    for total in totals:
        if total["simulator"] == GEANT4:
            continue
        # Geant4 over the same inputs, on the same kind of node or else the cpu nodes
        for wanted in (total["device"], "cpu"):
            same = [r for r in rows if r["simulator"] == GEANT4 and r["device"] == wanted
                    and r["scenario"] == total["scenario"] and r["input"] in total["inputs"]
                    and r["mean_s"] is not None]
            if len(same) == len(total["inputs"]):
                mean = sum(r["mean_s"] for r in same) / len(same)
                sem = math.sqrt(sum((r["sem_s"] or 0) ** 2 for r in same)) / len(same)
                total["speedup"], total["speedup_error"] = _ratio(mean, sem, total["mean_s"],
                                                                  total["sem_s"])
                total["speedup_reference"] = "geant4 on %s" % wanted
                break
    return rows, totals


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _cell(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return "%.6g" % value
    return value


def write_csv(path, rows, totals):
    partial = "%s.%d.partial" % (path, os.getpid())
    with open(partial, "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(COLUMNS)
        for row in rows + totals:
            writer.writerow([_cell(row.get(column)) for column in COLUMNS])
    os.replace(partial, path)


def _time(value):
    if value is None:
        return "-"
    if value >= 100:
        return "%.0f s" % value
    if value >= 1:
        return "%.3g s" % value
    return "%.3g ms" % (1000 * value)


def print_table(rows, totals, failures, out=print):
    """The table, as text: each kind of node and scenario in turn."""
    if not rows:
        out("ddml-timing: no timing results yet")
    blocks = {}
    for row in rows + totals:
        blocks.setdefault((row["device"], row["scenario"]), []).append(row)
    for (device, scenario), mine in sorted(blocks.items()):
        short = _short_names(sorted({r["input"] for r in mine if r["input"] != "all"}))
        names = dict((id(r), "all" if r["input"] == "all" else short.get(r["input"], r["input"]))
                     for r in mine)
        width = max([len(name) for name in names.values()] + [5])
        simulators = max([len(r["simulator"]) for r in mine] + [9])
        out("")
        out("%s nodes, %s: time per event (mean of the events timed, +- its standard error)"
            % (device, scenario))
        out("  %-*s  %-*s %7s %10s %7s %18s %8s  %s" % (simulators, "simulator", width, "input",
                                                       "events", "per event", "+-%", "speed-up",
                                                       "showers", "inference"))
        for row in sorted(mine, key=lambda r: (r["input"] == "all", r["simulator"] != GEANT4,
                                                r["simulator"], r["input"])):
            speedup = "-" if row["speedup"] is None else "x%.3g" % row["speedup"]
            if row["speedup_error"] is not None and row["speedup"] is not None:
                speedup += " +- %.2g" % row["speedup_error"]
            out("  %-*s  %-*s %7s %10s %7s %18s %8s  %s%s" % (
                simulators, row["simulator"], width, names[id(row)],
                row["events"] if row["events"] is not None else "-", _time(row["mean_s"]),
                "-" if row["rel_sem_pct"] is None else "%.1f" % row["rel_sem_pct"],
                speedup, "-" if row["showers_per_event"] is None
                else "%.2f" % row["showers_per_event"], row["inference_on"] or "-",
                "  (drifts: z = %.1f)" % row["drift_z"] if row["drift_z"] is not None
                and abs(row["drift_z"]) > 3 else ""))
        elsewhere = sorted({r["speedup_reference"] for r in mine if r["speedup_reference"]
                            and r["speedup_reference"] != "geant4 on %s" % device})
        if elsewhere:
            out("  (the speed-ups are on %s)" % " and ".join(elsewhere))
    wide = [r for r in rows if r["rel_sem_pct"] is not None and r["rel_sem_pct"] > 10]
    if wide:
        out("")
        out("ddml-timing: %d result(s) have a standard error above 10%% of their mean; "
            "events_for_10pct in the table says how many events would do" % len(wide))
    if failures:
        out("")
        out("ddml-timing: %d task(s) failed the last time they ran, with no results:"
            % len(failures))
        for failure in failures:
            out("  %s: %s" % (failure.get("key"), str(failure.get("reason")).split("\n")[0]))


def main(root, output=None, events=None, out=print):
    """Tabulate the results under root; returns the exit status."""
    results, failures = gather(root)
    if events is not None:
        results = [r for r in results if (r.get("events") or {}).get("requested") == events]
    rows, totals = summarise(results)
    print_table(rows, totals, failures, out)
    if rows:
        path = output or os.path.join(root, "summary.csv")
        folder = os.path.dirname(os.path.abspath(path))
        os.makedirs(folder, exist_ok=True)
        write_csv(path, rows, totals)
        out("")
        out("ddml-timing: the table is in %s" % path)
    sys.stdout.flush()
    return 0
