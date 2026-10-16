#!/usr/bin/env python3
"""Plot the observables that observables_from_reconstruction.sh measured.

Every profile found in the .npz files (one per simulator, in
<paths.data_dir>/observables) is drawn with every simulator that has it on the
same axes, as in the shower-profile figures of the benchmark paper: the mean
energy per event in each bin, Geant4 as a grey filled histogram and each model
as a line in its own colour and line style, each with its statistical
uncertainty as a band, and below, each model's relative deviation from Geant4.
The legend gives each model's distance from Geant4, by the metric of the
observable in the configuration (jensen_shannon: the Jensen-Shannon divergence,
natural logarithm, of the normalised profiles).

  plot_observables.py [--dir DIR] [--out DIR] [--events N] [--format pdf,png]
                      [--reference NAME] [SIMULATOR ...]

The figures go to <dir>/plots (or --out), one per observable, profile and
scenario, named <observable>_<profile>__<scenario>.<format>, with the metrics in
metrics.csv. It needs numpy and matplotlib, and PyYAML only to read the
configuration for the default folder and the metrics.
"""

import argparse
import csv
import glob
import json
import os
import re
import sys

SEP = "__"
REFERENCE = "geant4"
REFERENCE_FILL = "#d9d9d9"
REFERENCE_EDGE = "#8c8c8c"
INK = "#1f1f1f"
MUTED = "#6b6b6b"
GRID = "#e6e6e6"
# How each kind of observable is drawn: axis labels, and a logarithmic y axis
STYLES = {
    "radial_profile": ("Distance to the shower axis [mm]", True),
    "longitudinal_profile": ("ECAL layer", False),
}


def say(message):
    print("plot_observables: " + message, file=sys.stderr)


def load_settings():
    """(observables folder, {observable: metric}) from the configuration, if it can be read."""
    try:
        sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
        from fasinate_config import load_config
        config = load_config()
    except Exception as error:  # no PyYAML, or no configuration
        say("can't read the configuration (%s), so give --dir" % error)
        return None, {}
    metrics = {name: (settings or {}).get("metric") or "jensen_shannon"
               for name, settings in (config.get("observables") or {}).items()}
    return os.path.join(config["paths"]["data_dir"], "observables"), metrics


class Simulator:
    def __init__(self, path):
        import numpy as np
        with np.load(path, allow_pickle=False) as data:
            self.data = {key: data[key] for key in data.files}
        self.path = path
        self.name = str(self.data.get("meta__simulator", os.path.basename(path)[:-4]))
        self.label = str(self.data.get("meta__label", self.name))
        self.color = str(self.data.get("meta__color", "#1f6fb4"))
        style = json.loads(str(self.data.get("meta__linestyle", '"-"')))
        # matplotlib wants a dash pattern as (offset, (on, off, ...))
        self.linestyle = (style[0], tuple(style[1])) if isinstance(style, list) else style

    def profiles(self):
        found = []
        for key in self.data:
            parts = key.split(SEP)
            if len(parts) == 4 and parts[3] == "sum":
                found.append(tuple(parts[:3]))
        return found

    def profile(self, observable, scenario, profile):
        """Bin edges, mean energy per event per bin, its uncertainty, and the events."""
        import numpy as np
        key = SEP.join((observable, scenario, profile))
        n = float(self.data[SEP.join((scenario, "n_events"))].sum())
        total = self.data[key + SEP + "sum"].sum(axis=0)
        total_sq = self.data[key + SEP + "sum_sq"].sum(axis=0)
        if n == 0:
            return self.data[key + SEP + "edges"], None, None, 0
        mean = total / n
        variance = np.clip(total_sq / n - mean ** 2, 0, None)
        return self.data[key + SEP + "edges"], mean, np.sqrt(variance / n), int(n)


def jensen_shannon(p, q):
    """Jensen-Shannon divergence of two non-negative profiles, normalised (natural log)."""
    import numpy as np
    p = np.clip(np.asarray(p, dtype=float), 0, None)
    q = np.clip(np.asarray(q, dtype=float), 0, None)
    if p.sum() <= 0 or q.sum() <= 0:
        return float("nan")
    p, q = p / p.sum(), q / q.sum()
    m = (p + q) / 2

    def kl(a):
        nonzero = a > 0
        return float(np.sum(a[nonzero] * np.log(a[nonzero] / m[nonzero])))

    return 0.5 * kl(p) + 0.5 * kl(q)


METRICS = {"jensen_shannon": ("JS", jensen_shannon)}


def drawable(values, log_y):
    """The values as drawn: on a log axis, empty bins become gaps rather than
    lines down to the bottom of the axes."""
    import numpy as np
    values = np.asarray(values, dtype=float)
    return np.where(values > 0, values, np.nan) if log_y else values


def stairs_band(ax, edges, low, high, color, alpha):
    import numpy as np
    ax.fill_between(edges, np.append(low, low[-1]), np.append(high, high[-1]),
                    step="post", color=color, alpha=alpha, linewidth=0)


def plot_profile(observable, scenario, profile, curves, reference, metric_name, out_base,
                 formats):
    """One figure; returns [(simulator, metric, value)]."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    xlabel, log_y = STYLES.get(observable, (profile, False))
    short, metric = METRICS.get(metric_name, (metric_name, None))
    if metric is None:
        say("no metric '%s' is known (known: %s), so none is shown"
            % (metric_name, ", ".join(METRICS)))

    # Leave out the bins that are empty for every simulator (e.g. layer numbers
    # the detector doesn't have)
    edges = curves[0][1]
    filled = np.zeros(len(edges) - 1, dtype=bool)
    for _, _, mean, _, _ in curves:
        filled |= mean > 0
    if not filled.any():
        say("%s: every bin is empty, so no plot" % out_base)
        return []
    first, last = np.flatnonzero(filled)[[0, -1]]
    keep = slice(first, last + 1)
    edges = edges[first:last + 2]

    ref = None
    if reference is not None:
        ref = next((c for c in curves if c[0] is reference), None)
    with_ratio = ref is not None and len(curves) > 1
    fig, axes = plt.subplots(2 if with_ratio else 1, 1, sharex=True, figsize=(5.2, 4.8),
                             gridspec_kw={"height_ratios": [3, 1.2]} if with_ratio else None,
                             squeeze=False)
    top = axes[0][0]
    results = []

    if ref is not None:
        _, _, mean, err, n = ref
        mean, err = mean[keep], err[keep]
        top.stairs(drawable(mean, log_y), edges, fill=True, color=REFERENCE_FILL,
                   label=ref[0].label, zorder=1)
        top.stairs(drawable(mean, log_y), edges, color=REFERENCE_EDGE, linewidth=1,
                   baseline=None, zorder=2)
        stairs_band(top, edges, mean - err, mean + err, REFERENCE_EDGE, 0.25)
    for simulator, _, mean, err, n in curves:
        if simulator is reference:
            continue
        mean, err = mean[keep], err[keep]
        label = simulator.label
        if ref is not None and metric is not None:
            value = metric(ref[2][keep], mean)
            results.append((simulator, short, value))
            label = "%s (%s %.2e)" % (label, short, value)
        top.stairs(drawable(mean, log_y), edges, color=simulator.color,
                   linestyle=simulator.linestyle, linewidth=1.6, label=label,
                   baseline=None, zorder=3)
        stairs_band(top, edges, mean - err, mean + err, simulator.color, 0.2)

    if log_y:
        top.set_yscale("log")
        positive = np.concatenate([c[2][keep][c[2][keep] > 0] for c in curves])
        if positive.size:
            top.set_ylim(positive.min() / 3, positive.max() * 3)
    top.set_ylabel("Mean energy per event [GeV]", color=INK)
    n_ref = ref[4] if ref is not None else curves[0][4]
    top.text(0.02, 0.03 if not log_y else 0.03, "%s, %d events%s" % (
        scenario, n_ref, " (Geant4)" if ref is not None else ""),
        transform=top.transAxes, fontsize=8, color=MUTED, va="bottom")
    top.legend(frameon=False, fontsize=8, loc="upper right", labelcolor=INK)

    if with_ratio:
        bottom = axes[1][0]
        r_mean, r_err = ref[2][keep], ref[3][keep]
        valid = r_mean > 0
        safe = np.where(valid, r_mean, np.nan)
        stairs_band(bottom, edges, -r_err / safe * 100, r_err / safe * 100, REFERENCE_EDGE, 0.3)
        bottom.axhline(0, color=REFERENCE_EDGE, linewidth=1)
        deviations = []
        for simulator, _, mean, err, _ in curves:
            if simulator is reference:
                continue
            dev = (mean[keep] / safe - 1) * 100
            band = err[keep] / safe * 100
            bottom.stairs(dev, edges, color=simulator.color, baseline=None,
                          linestyle=simulator.linestyle, linewidth=1.4)
            stairs_band(bottom, edges, dev - band, dev + band, simulator.color, 0.2)
            deviations.append(np.abs(dev[np.isfinite(dev)]))
        spread = np.concatenate(deviations) if deviations else np.array([])
        limit = np.percentile(spread, 95) * 1.3 if spread.size else 10
        limit = float(np.clip(limit, 5, 100))
        bottom.set_ylim(-limit, limit)
        bottom.set_ylabel("Rel. dev. [%]", color=INK)
        bottom.set_xlabel(xlabel, color=INK)
    else:
        top.set_xlabel(xlabel, color=INK)

    for row in axes:
        ax = row[0]
        ax.grid(True, color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
    top.set_xlim(edges[0], edges[-1])
    fig.align_ylabels([row[0] for row in axes])
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.08)
    for fmt in formats:
        fig.savefig("%s.%s" % (out_base, fmt), dpi=200)
    plt.close(fig)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Plot the observables measured by observables_from_reconstruction.sh.")
    parser.add_argument("simulators", nargs="*", metavar="SIMULATOR",
                        help="only these simulators (default: every .npz file found)")
    parser.add_argument("--dir", help="the folder with the .npz files "
                                      "(default: observables in paths.data_dir)")
    parser.add_argument("--out", help="where the figures go (default: <dir>/plots)")
    parser.add_argument("--events", type=int, metavar="N",
                        help="plot the measurements of the first N events (<simulator>_n<N>.npz)")
    parser.add_argument("--format", default="pdf,png",
                        help="figure formats, separated by commas (default: pdf,png)")
    parser.add_argument("--reference", default=REFERENCE,
                        help="the simulator the others are compared to (default: geant4)")
    args = parser.parse_args(argv)

    folder, metrics = load_settings() if not args.dir else (args.dir, load_settings()[1])
    if not folder:
        return 1
    suffix = "" if args.events is None else "_n%d" % args.events
    paths = []
    for path in sorted(glob.glob(os.path.join(folder, "*%s.npz" % suffix))):
        name = os.path.basename(path)[:-len(suffix + ".npz")]
        if name.startswith(".") or (suffix == "" and re.search(r"_n\d+$", name)):
            continue
        if args.simulators and name not in args.simulators and name != args.reference:
            continue
        paths.append(path)
    if not paths:
        say("no measurements (*%s.npz) in %s; make them with "
            "./observables_from_reconstruction.sh" % (suffix, folder))
        return 1
    simulators = [Simulator(p) for p in paths]
    reference = next((s for s in simulators if s.name == args.reference), None)
    if reference is None:
        say("there's no %s measurement, so there's nothing to compare to; plotting "
            "without deviations" % args.reference)
    # The reference first, then the models in the order of their files
    if reference is not None:
        simulators.remove(reference)
        simulators.insert(0, reference)

    out = args.out or os.path.join(folder, "plots")
    os.makedirs(out, exist_ok=True)
    formats = [f.strip() for f in args.format.split(",") if f.strip()]
    groups = []
    for simulator in simulators:
        for key in simulator.profiles():
            if key not in groups:
                groups.append(key)

    rows = []
    for observable, scenario, profile in groups:
        curves = []
        edges = None
        for simulator in simulators:
            if (observable, scenario, profile) not in simulator.profiles():
                continue
            e, mean, err, n = simulator.profile(observable, scenario, profile)
            if mean is None:
                say("%s has no events of %s, so it's left out" % (simulator.name, scenario))
                continue
            if edges is None:
                edges = e
            elif len(e) != len(edges) or not (abs(e - edges) < 1e-9).all():
                say("%s measured %s/%s with other bins, so it's left out; measure it again"
                    % (simulator.name, observable, profile))
                continue
            curves.append((simulator, e, mean, err, n))
        if not curves:
            continue
        base = os.path.join(out, "%s_%s%s%s%s" % (observable, profile, SEP, scenario, suffix))
        results = plot_profile(observable, scenario, profile, curves,
                               reference if any(c[0] is reference for c in curves) else None,
                               metrics.get(observable, "jensen_shannon"), base, formats)
        say("%s: %s" % (", ".join(c[0].label for c in curves),
                        ", ".join("%s.%s" % (os.path.basename(base), f) for f in formats)))
        for simulator, short, value in results:
            rows.append((observable, scenario, profile, simulator.name, short, value))

    with open(os.path.join(out, "metrics%s.csv" % suffix), "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["observable", "scenario", "profile", "simulator", "metric", "value"])
        for row in rows:
            writer.writerow(row[:5] + ("%.6g" % row[5],))
    for observable, scenario, profile, name, short, value in rows:
        print("%-22s %-14s %-7s %-14s %s %.3e" % (observable, scenario, profile, name, short,
                                                 value))
    say("figures and metrics%s.csv in %s" % (suffix, out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
