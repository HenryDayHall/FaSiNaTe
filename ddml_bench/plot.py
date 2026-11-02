"""Plot the observables that the observables stage measured.

Every profile found in the .npz files (one per simulator, in
<paths.data_dir>/observables) is drawn with every simulator that has it on the
same axes, as in the shower-profile figures of the benchmark paper: the mean
energy per event in each bin (for a radial profile, per unit area: divided by
the area of the bin's ring around the shower axis; see DENSITIES), Geant4 as a
grey filled histogram and each model as a line in its own colour and line
style, each with its statistical uncertainty as a band, and below, each model's
relative deviation from Geant4. The legend gives each model's distance from
Geant4, by the metric of the observable in the configuration (jensen_shannon:
the Jensen-Shannon divergence, natural logarithm, of the fractions of the
energy in the bins; l1: the mean absolute difference).

The reconstructed energy of each event (the observable energy) gives two
figures, the energy linearity and resolution of the benchmark paper. The events
at each true energy E give mu90 and sigma90, the mean and the standard
deviation of the central 90% of their reconstructed energies (core90(): sorted,
without the lowest and the highest 5%), each with its uncertainty from
resampling the events (bootstrap_errors()). The linearity is mu90 against E,
fitted with a straight line, slope E + intercept; the resolution is
sigma90/mu90 against E, fitted with a + b/sqrt(E) + c/E. Both fits are
weighted least squares, and the legend gives the fitted function. Each
simulator is a set of points with its fit, Geant4 in grey, and below, each
model's relative deviation from Geant4; the metric (by default l1) compares
the points at the same energies.

The intrinsic angles (the observable intrinsic_angle) give a figure for each
angle, theta and phi, and each PCA of the showers' hits (all of them, and the
most energetic 4%), as in the benchmark paper: the distribution of the
residuals, reconstructed - true, as the fraction of the showers per degree,
drawn as the profiles are (Geant4 filled, the models as lines, the relative
deviations from Geant4 below), and compared by the Jensen-Shannon divergence
of the fractions in the bins. The bins are from -max_residual to max_residual
degrees, n_bins of them, in the observable's settings in the configuration
(only drawing them, so changing them needs no measuring again); showers
outside them still count among all the showers, and those without an axis
(fewer than two hits) don't. The text in the corner says which PCA, and the
side of the bounding box its hits were taken from, if any, as the .npz file
records the settings it was measured with.

The di-photon separation (the observable photon_separation) gives a figure
for each energy of the photons, as in the benchmark paper: the mean number of
reconstructed photons per event in bins of the separation of the two photons
on the ECAL face, drawn as the profiles are, with a line at two photons. Each
simulator's curve is the mean over the ddsim seeds of each seed's mean, and
its band the spread over the seeds (separation_points()). The metric compares
the number of reconstructed photons in each bin, seed by seed, and gives the
mean over the seeds with their spread as its uncertainty (seed_metric()).


A model that is the closure test of another (closure in models/<model>.yaml:
the same model, e.g. the same weights through another of DDML's inference
backends) is also drawn against that one alone, as the reference, in
<out>/closure/<model>/, with their metrics there; and closure.csv says, for
each scenario, whether all the two measured is the same, number for number,
as it should be where both simulated the same events with the same seeds and
closure.identical says so (closures()).

The pi0s of the tau decays (the observable pi0) give two figures, as the
paper's figure 11 does, of the good pi0 candidates (both photons those of one
MC pi0): the invariant mass of their two photons, and their reconstructed
energy less the MC pi0's, each as the fraction of the good pi0s per GeV. Each
simulator's curve is the mean over the ddsim seeds of each seed's, and its band
the spread over the seeds; the metric compares the numbers of good pi0s in the
bins, seed by seed, as the separation's does (seed_histograms(),
seed_metric()).

Every figure is also drawn as SVG for the dashboard, dashboard.html next to
the figures (ddml_bench/dashboard.py), with each model's parts marked with its
name (mark()), so that its menu can hide them; --format svg keeps the SVG
files, and --no-dashboard leaves the dashboard out.


  python3 -m ddml_bench.plot [--dir DIR] [--out DIR] [--events N]
                             [--format pdf,png] [--reference NAME]
                             [--no-closure] [--no-dashboard] [SIMULATOR ...]

(plot_observables.sh runs it.) The figures go to <dir>/plots (or --out), one per
observable, profile and scenario, named <observable>_<profile>__<scenario>.<format>
(energy_linearity__<scenario> and energy_resolution__<scenario> for the
energy, intrinsic_angle_<angle>_<hits>__<scenario> for the intrinsic angles,
e.g. intrinsic_angle_theta_top4__single_photon,
photon_separation_<E>GeV__<scenario> for the separation, and pi0_mass__<scenario>
and pi0_energy__<scenario> for the pi0s), with the metrics in
metrics.csv (and their uncertainties, for those that have one) and the fitted
parameters of the energy figures, with their uncertainties, chi^2 and degrees
of freedom, in fits.csv. It needs numpy and matplotlib, and PyYAML only to
read the configuration for the default folder, the metrics and the bins of the
residuals and separations, so it also works on .npz files copied to a laptop.
"""

import argparse
import contextlib
import csv
import json
import os
import sys
from urllib.parse import quote

from . import dashboard, measure
from .dashboard import DATA, KEY, MARK, TARGET
from .layout import GEANT4, OBSERVABLES_DIR, PLOTS_DIR, SEP, figure_base, measurement_files

REFERENCE = GEANT4
REFERENCE_FILL = "#d9d9d9"
REFERENCE_EDGE = "#8c8c8c"
INK = "#1f1f1f"
MUTED = "#6b6b6b"
GRID = "#e6e6e6"
# A line across the axes at a value the figure is about (e.g. the two photons
# a di-photon event should be reconstructed as), red as in the benchmark paper
GUIDE = "#c4302b"
# How each kind of observable is drawn: axis labels, and a logarithmic y axis
STYLES = {
    "radial_profile": ("Distance to the shower axis [mm]", True),
    "longitudinal_profile": ("ECAL layer", False),
    "intrinsic_angle": ("Reconstructed \u2212 true [\u00b0]", False),
    "photon_separation": ("Separation on the ECAL face [mm]", False),
}
PER_BIN = "Mean energy per event [GeV]"


def ring_areas(edges):
    """The area of each radial bin's ring around the shower axis, pi (r_out^2 -
    r_in^2): the area element of polar coordinates about the axis, r dr dphi
    (Jacobian r), integrated over the bin and all the way around."""
    import numpy as np
    edges = np.asarray(edges, dtype=float)
    return np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)


def bin_widths(edges):
    import numpy as np
    return np.diff(np.asarray(edges, dtype=float))


# The kinds of observable drawn as densities: what each bin's energy is divided
# by (from the bin edges), and the y axis label then. The rings of a radial
# profile grow with the radius (as 2 pi r dr), so the energy in each is divided
# by its area, giving the energy per unit area transverse to the shower axis.
# The residuals of the intrinsic angles, fractions of the showers in each bin,
# are divided by the bins' widths. The other kinds are drawn per bin.
DENSITIES = {
    "radial_profile": (ring_areas, "Mean energy density per event [GeV/mm$^2$]"),
    "intrinsic_angle": (bin_widths, "Fraction of showers per degree"),
}


def bin_scale(observable, edges):
    """What each bin of the observable is divided by in the figures (ones if
    it is drawn per bin)."""
    import numpy as np
    if observable in DENSITIES:
        return DENSITIES[observable][0](edges)
    return np.ones(len(edges) - 1)


def say(message):
    print("plot_observables: " + message, file=sys.stderr)


def load_settings():
    """(observables folder, {observable: its settings}) from the configuration,
    if it can be read."""
    try:
        from .config import load_config
        config = load_config()
    except Exception as error:  # no PyYAML, or no configuration
        say("can't read the configuration (%s), so give --dir" % error)
        return None, {}
    observables = {name: settings if isinstance(settings, dict) else {}
                   for name, settings in (config.get("observables") or {}).items()}
    return os.path.join(config["paths"]["data_dir"], OBSERVABLES_DIR), observables


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

    def per_event(self):
        """[(observable, scenario)] of the observables stored per event: those
        with settings of their own, <observable>__<scenario>__settings (the
        profiles' have a profile too)."""
        found = []
        for key in self.data:
            parts = key.split(SEP)
            if len(parts) == 3 and parts[2] == "settings":
                found.append(tuple(parts[:2]))
        return found

    def events(self, observable, scenario):
        """The true and the reconstructed energy of each event, GeV."""
        key = SEP.join((observable, scenario))
        return self.data[key + SEP + "true"], self.data[key + SEP + "reconstructed"]

    def stored(self, observable, scenario):
        """What is stored of each event of an observable stored per event:
        {what: [event]}, e.g. {"dtheta_all": ...} for the intrinsic angle."""
        prefix = SEP.join((observable, scenario)) + SEP
        return dict((key[len(prefix):], value) for key, value in self.data.items()
                    if key.startswith(prefix) and SEP not in key[len(prefix):]
                    and key[len(prefix):] != "settings")


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


def l1_distance(p, q):
    """The mean absolute difference of two sets of values (bins or points)."""
    import numpy as np
    return float(np.mean(np.abs(np.asarray(q, dtype=float) - np.asarray(p, dtype=float))))


METRICS = {"jensen_shannon": ("JS", jensen_shannon), "l1": ("L1", l1_distance)}
# The metric of an observable whose settings don't give one
DEFAULT_METRICS = {"energy": "l1"}


def metric_of(observable, settings):
    """The metric of an observable: its settings' metric, or its default."""
    return (settings or {}).get("metric") or DEFAULT_METRICS.get(observable, "jensen_shannon")


def drawable(values, log_y):
    """The values as drawn: on a log axis, empty bins become gaps rather than
    lines down to the bottom of the axes."""
    import numpy as np
    values = np.asarray(values, dtype=float)
    return np.where(values > 0, values, np.nan) if log_y else values


def stairs_band(ax, edges, low, high, color, alpha):
    import numpy as np
    return ax.fill_between(edges, np.append(low, low[-1]), np.append(high, high[-1]),
                           step="post", color=color, alpha=alpha, linewidth=0)


def plot_profile(observable, scenario, profile, curves, reference, metric_name, out_base,
                 formats, xlabel=None, note=None, trim=True, room=False, ylabel=None,
                 guide=None, metric_texts=None, every_deviation=False):
    """One figure; returns [(simulator, metric, value)]. xlabel, ylabel and note,
    the text in the corner, replace the observable's own; with trim, the bins
    at either end that are empty for every simulator are left out; with room,
    the top of the axes is left to the legend and the note, which goes at the
    top left (for distributions peaked in the middle); guide is a value to
    draw a line at, across the axes. With metric_name None, the metric is the
    caller's: metric_texts, [(simulator, text)], gives what each simulator's
    label says of it. The axis of the deviations from Geant4 shows the 95% of
    them in the bins it measures well, or with every_deviation all of them
    (for a few bins, all measured well)."""
    import numpy as np

    default_xlabel, log_y = STYLES.get(observable, (profile, False))
    xlabel = xlabel or default_xlabel
    if ylabel is None:
        ylabel = DENSITIES[observable][1] if observable in DENSITIES else PER_BIN
    short, metric = METRICS.get(metric_name, (metric_name, None))
    if metric is None and metric_name is not None:
        say("no metric '%s' is known (known: %s), so none is shown"
            % (metric_name, ", ".join(METRICS)))
    metric_texts = list(metric_texts or ())

    # Leave out the bins that are empty for every simulator (e.g. layer numbers
    # the detector doesn't have)
    edges = curves[0][1]
    filled = np.zeros(len(edges) - 1, dtype=bool)
    for _, _, mean, _, _ in curves:
        filled |= mean > 0
    if not filled.any():
        say("%s: every bin is empty, so no plot" % out_base)
        return []
    first, last = np.flatnonzero(filled)[[0, -1]] if trim else (0, len(filled) - 1)
    keep = slice(first, last + 1)
    edges = edges[first:last + 2]
    # The profiles are drawn divided by this (e.g. a radial profile's ring
    # areas). The metric compares the fractions of the energy in the bins, so
    # it takes the energy per bin, and the deviations from Geant4 are the same
    # either way.
    scale = bin_scale(observable, edges)

    ref = None
    if reference is not None:
        ref = next((c for c in curves if c[0] is reference), None)
    with_ratio = ref is not None and len(curves) > 1
    fig, axes = new_figure(with_ratio)
    top = axes[0][0]
    results = []
    # The legend's entries: (handle, label, simulator; None for the reference)
    entries = []

    if ref is not None:
        _, _, mean, err, n = ref
        mean, err = mean[keep] / scale, err[keep] / scale
        filled = top.stairs(drawable(mean, log_y), edges, fill=True, color=REFERENCE_FILL,
                            label=ref[0].label, zorder=1)
        top.stairs(drawable(mean, log_y), edges, color=REFERENCE_EDGE, linewidth=1,
                   baseline=None, zorder=2)
        stairs_band(top, edges, mean - err, mean + err, REFERENCE_EDGE, 0.25)
        entries.append((filled, ref[0].label, None))
    for simulator, _, mean, err, n in curves:
        if simulator is reference:
            continue
        mean, err = mean[keep], err[keep]
        label = simulator.label
        if ref is not None and metric is not None:
            value = metric(ref[2][keep], mean)
            results.append((simulator, short, value))
            label = "%s (%s %.2e)" % (label, short, value)
        else:
            given = next((t for s, t in metric_texts if s is simulator), None)
            if given:
                label = "%s (%s)" % (label, given)
        mean, err = mean / scale, err / scale
        line = top.stairs(drawable(mean, log_y), edges, color=simulator.color,
                          linestyle=simulator.linestyle, linewidth=1.6, label=label,
                          baseline=None, zorder=3)
        mark(simulator, DATA, line,
             stairs_band(top, edges, mean - err, mean + err, simulator.color, 0.2))
        entries.append((line, label, simulator))

    if log_y:
        top.set_yscale("log")
        drawn = np.concatenate([c[2][keep] / scale for c in curves])
        positive = drawn[drawn > 0]
        if positive.size:
            top.set_ylim(positive.min() / 3, positive.max() * 3)
    elif room:
        drawn = np.concatenate([(c[2][keep] + c[3][keep]) / scale for c in curves])
        drawn = drawn[np.isfinite(drawn)]
        highest = max(float(drawn.max()) if drawn.size else 0.0, guide or 0.0) or 1.0
        # (the legend has a line for each simulator)
        top.set_ylim(0, highest * (1.15 + 0.08 * len(curves)))
    if guide is not None:
        top.axhline(guide, color=GUIDE, linewidth=1.0, linestyle=(0, (4, 2)), zorder=2.5)
    top.set_ylabel(ylabel, color=INK)
    if note is None:
        n_ref = ref[4] if ref is not None else curves[0][4]
        note = "%s, %d events%s" % (scenario, n_ref, " (Geant4)" if ref is not None else "")
    top.text(0.02, 0.97 if room else 0.03, note, transform=top.transAxes, fontsize=8,
             color=MUTED, va="top" if room else "bottom")
    legend(top, entries, fontsize=8, loc="upper right")

    if with_ratio:
        bottom = axes[1][0]
        r_mean, r_err = ref[2][keep], ref[3][keep]
        valid = r_mean > 0
        safe = np.where(valid, r_mean, np.nan)
        stairs_band(bottom, edges, -r_err / safe * 100, r_err / safe * 100, REFERENCE_EDGE, 0.3)
        bottom.axhline(0, color=REFERENCE_EDGE, linewidth=1)
        # The deviations' axis is set by the bins Geant4 measures to 10% or
        # better, if there are any, so that sparse tails don't set it
        measured = valid & (r_err < 0.1 * np.where(valid, r_mean, np.inf))
        measured = measured if measured.any() else valid
        deviations = []
        for simulator, _, mean, err, _ in curves:
            if simulator is reference:
                continue
            dev = (mean[keep] / safe - 1) * 100
            band = err[keep] / safe * 100
            mark(simulator, DATA,
                 bottom.stairs(dev, edges, color=simulator.color, baseline=None,
                               linestyle=simulator.linestyle, linewidth=1.4),
                 stairs_band(bottom, edges, dev - band, dev + band, simulator.color, 0.2))
            deviations.append(np.abs(dev[measured & np.isfinite(dev)]))
        spread = np.concatenate(deviations) if deviations else np.array([])
        limit = np.percentile(spread, 100 if every_deviation else 95) * 1.3 if spread.size else 10
        limit = float(np.clip(limit, 5, 100))
        bottom.set_ylim(-limit, limit)
        bottom.set_ylabel("Rel. dev. [%]", color=INK)
        bottom.set_xlabel(xlabel, color=INK)
    else:
        top.set_xlabel(xlabel, color=INK)

    top.set_xlim(edges[0], edges[-1])
    finish(fig, axes, out_base, formats)
    return results


def new_figure(with_ratio):
    """A figure of the values, with the relative deviations from Geant4 below
    them if with_ratio: (figure, [[axes]])."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt.subplots(2 if with_ratio else 1, 1, sharex=True, figsize=(5.2, 4.8),
                        gridspec_kw={"height_ratios": [3, 1.2]} if with_ratio else None,
                        squeeze=False)


def finish(fig, axes, out_base, formats):
    """Style the axes as every figure is, and save the figure in each format."""
    import matplotlib.pyplot as plt
    for row in axes:
        ax = row[0]
        ax.grid(True, color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
    fig.align_ylabels([row[0] for row in axes])
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.08)
    for fmt in formats:
        if fmt != "svg":
            fig.savefig("%s.%s" % (out_base, fmt), dpi=200)
    # The SVG last, as it adds the legend's click targets (invisible)
    if "svg" in formats or collector is not None:
        svg = svg_of(fig)
        if "svg" in formats:
            with open(out_base + ".svg", "wb") as stream:
                stream.write(svg)
        if collector is not None:
            collector.add(out_base, svg)
    plt.close(fig)


# ---------------------------------------------------------------------------
# The marks the dashboard (ddml_bench/dashboard.py) goes by: each model's parts
# of a figure are marked with its name, so that its menu can hide them

# Gathers the figures for the dashboard while main() draws them (collecting())
collector = None


@contextlib.contextmanager
def collecting(into):
    """Give every figure drawn meanwhile, and its simulators, to into (a
    dashboard.Collector, or None for nothing)."""
    global collector
    before, collector = collector, into
    try:
        yield into
    finally:
        collector = before


def mark(simulator, role, *artists):
    """Mark the artists (and those in containers, e.g. of errorbar() or bar())
    as drawing a simulator, in a role: DATA, its curves, bands, points and fits,
    which the dashboard hides with it, or KEY, its legend entry, which it
    dims. In the SVG, each is a group with the id <MARK>.<role>.<simulator>.<n>
    (numbered by svg_of())."""
    from matplotlib.artist import Artist
    gid = "%s.%s.%s" % (MARK, role, quote(simulator.name, safe=""))

    def walk(item):
        if isinstance(item, Artist):
            item.set_gid(gid)
        elif isinstance(item, (tuple, list)):
            for child in item:
                walk(child)
        # a bar()'s error bars aren't among its patches
        if getattr(item, "errorbar", None) is not None:
            walk(item.errorbar)

    walk(artists)
    if collector is not None:
        collector.simulator(simulator)


def legend(ax, entries, **options):
    """The legend of the axes, from entries, [(handle, label, simulator, None
    for the reference)], each entry marked as its simulator's KEY."""
    drawn = ax.legend([handle for handle, _, _ in entries], [label for _, label, _ in entries],
                      frameon=False, labelcolor=INK, **options)
    for (_, _, simulator), text, handle in zip(entries, drawn.get_texts(), legend_handles(drawn)):
        if simulator is not None:
            mark(simulator, KEY, text, handle)
    return drawn


def legend_handles(drawn):
    """The legend's handles, in the order of its entries (legendHandles before
    matplotlib 3.7)."""
    handles = getattr(drawn, "legend_handles", None)
    return handles if handles is not None else drawn.legendHandles


def legend_targets(fig):
    """An invisible rectangle over each marked legend entry, its handle and
    text, marked as its simulator's TARGET: what the dashboard's clicks land on
    (the outlines of the text's letters alone are hard to hit). Added only for
    the SVG, after the other formats are saved."""
    from matplotlib.patches import Rectangle
    from matplotlib.transforms import Bbox
    prefix = "%s.%s." % (MARK, KEY)
    entries = [(text, handle) for ax in fig.axes if ax.get_legend() is not None
               for text, handle in zip(ax.get_legend().get_texts(),
                                       legend_handles(ax.get_legend()))
               if (text.get_gid() or "").startswith(prefix)]
    if not entries:
        return
    try:
        fig.draw_without_rendering()  # where the legends go (matplotlib 3.6)
    except AttributeError:
        fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    to_figure = fig.transFigure.inverted()
    for text, handle in entries:
        box = Bbox.union([text.get_window_extent(renderer),
                          handle.get_window_extent(renderer)]).padded(1.5 * fig.dpi / 72)
        (x0, y0), (x1, y1) = to_figure.transform(box.get_points())
        fig.add_artist(Rectangle((x0, y0), x1 - x0, y1 - y0, transform=fig.transFigure,
                                 facecolor="white", alpha=0, linewidth=0, zorder=10,
                                 gid="%s.%s.%s" % (MARK, TARGET, text.get_gid()[len(prefix):])))


def svg_of(fig):
    """The figure as SVG (bytes), with the legend's click targets and each mark
    numbered, so that ids are unique; text as outlines (so that it looks the
    same everywhere), and the ids matplotlib makes up for what the figure's
    parts share (clip paths, markers, letters) the same for the same thing in
    every figure, so the dashboard keeps one of each."""
    import io
    import matplotlib
    legend_targets(fig)
    count = 0
    for artist in fig.findobj():
        gid = artist.get_gid()
        if gid and gid.startswith(MARK + "."):
            count += 1
            artist.set_gid("%s.%d" % (gid, count))
    stream = io.BytesIO()
    with matplotlib.rc_context({"svg.fonttype": "path", "svg.hashsalt": MARK}):
        fig.savefig(stream, format="svg", metadata={"Date": None})
    return stream.getvalue()


# ---------------------------------------------------------------------------
# The energy linearity and resolution, from the reconstructed energy of each
# event (the observable energy, ddml_bench/measure.py)

# Events whose true energies differ by less than this fraction of them are at
# the same energy
SAME_ENERGY = 1e-2
# Resamplings of the events at each energy, for the uncertainties of mu90 and
# sigma90; the same every time, so the figures are too
BOOTSTRAP = 200
BOOTSTRAP_SEED = 0
# The models' markers, in the order of their files (Geant4's is a circle)
MARKERS = ("s", "^", "D", "v", "P", "X")
ENERGY_XLABEL = "True energy $E$ [GeV]"
# Each figure of the energy: its name, the value drawn against the true energy
# (a key of energy_points()), its axis label, the names of the parameters of
# the function fitted to it, and how each term of that function is written,
# in the order of the parameters (fit_terms() has the terms themselves)
ENERGY_FIGURES = (
    ("linearity", "mu90", r"$\mu_{90}$ of $E_\mathrm{reco}$ [GeV]",
     ("slope", "intercept"), ("%s $E$", "%s GeV")),
    ("resolution", "resolution", r"$\sigma_{90}/\mu_{90}$ of $E_\mathrm{reco}$",
     ("a", "b", "c"), ("%s", r"%s/$\sqrt{E}$", "%s/$E$")),
)


def core90(data):
    """The central 90% of the values: sorted, without the lowest and the
    highest 5% of them."""
    import numpy as np
    sorted_data = np.sort(data)
    n = len(sorted_data)
    lower_idx = int(n * 0.05)
    upper_idx = int(n * 0.95)
    return sorted_data[lower_idx:upper_idx]


def mu90_sigma90(data):
    """mu90 and sigma90: the mean and the standard deviation of the central 90%
    of the values."""
    import numpy as np
    core_90 = core90(data)
    return float(np.mean(core_90)), float(np.std(core_90))


def bootstrap_errors(data, resamplings=BOOTSTRAP, seed=BOOTSTRAP_SEED):
    """The uncertainties of mu90, sigma90 and sigma90/mu90 of the values: their
    standard deviations over resamplings of the values (as many, drawn with
    replacement)."""
    import numpy as np
    data = np.asarray(data, dtype=float)
    n = len(data)
    rng = np.random.default_rng(seed)
    # The central 90% of each resampling, as core90() takes it
    lower_idx, upper_idx = int(n * 0.05), int(n * 0.95)
    batch = max(1, min(resamplings, 2000000 // n))
    mu, sigma = [], []
    for first in range(0, resamplings, batch):
        size = min(batch, resamplings - first)
        cores = np.sort(data[rng.integers(0, n, size=(size, n))], axis=1)[:, lower_idx:upper_idx]
        mu.append(cores.mean(axis=1))
        sigma.append(cores.std(axis=1))
    mu, sigma = np.concatenate(mu), np.concatenate(sigma)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = sigma / mu
    return float(np.std(mu, ddof=1)), float(np.std(sigma, ddof=1)), float(np.std(ratio, ddof=1))


def same_energies(true):
    """The indices of the events at each true energy, from the lowest: the
    events in order of true energy, split wherever it jumps by more than
    SAME_ENERGY of itself."""
    import numpy as np
    order = np.argsort(true, kind="stable")
    ordered = np.asarray(true, dtype=float)[order]
    jumps = np.flatnonzero(np.diff(ordered) > SAME_ENERGY * np.abs(ordered[1:]))
    return np.split(order, jumps + 1)


def _some(values, most=3):
    """A few of the values, for a message."""
    return ", ".join(values[:most]) + (", ..." if len(values) > most else "")


def energy_points(true, reconstructed):
    """The points of the energy figures, one per true energy, from the lowest:
    dicts of the energy (its events' mean true energy, GeV), the number of
    events n, and mu90, sigma90 and their ratio, the resolution, of the events'
    reconstructed energies, each with its uncertainty (<name>_err); and why
    events were left out, if any were. If most were, none are kept, as the
    events then aren't at fixed energies (e.g. an input of a spectrum)."""
    import numpy as np
    true = np.asarray(true, dtype=float)
    reconstructed = np.asarray(reconstructed, dtype=float)
    if not true.size:
        return [], ["there are no events"]
    points, spread, few, unmeasured = [], [], [], []
    left_out = 0
    for index in same_energies(true):
        energies = true[index]
        energy = float(energies.mean())
        values = reconstructed[index]
        if energies.max() - energies.min() > SAME_ENERGY * abs(energy):
            spread.append("%.4g to %.4g" % (energies.min(), energies.max()))
        elif len(core90(values)) < 2:
            few.append("%.4g" % energy)
        else:
            mu90, sigma90 = mu90_sigma90(values)
            if mu90 > 0:
                mu90_err, sigma90_err, resolution_err = bootstrap_errors(values)
                points.append({"energy": energy, "n": len(index), "mu90": mu90,
                               "mu90_err": mu90_err, "sigma90": sigma90,
                               "sigma90_err": sigma90_err, "resolution": sigma90 / mu90,
                               "resolution_err": resolution_err})
                continue
            unmeasured.append("%.4g" % energy)
        left_out += len(index)
    skipped = []
    if spread:
        skipped.append("%d set(s) of events whose true energies differ by more than %g%% (%s GeV) "
                       "are left out: the energy linearity and resolution need inputs at fixed "
                       "energies" % (len(spread), 100 * SAME_ENERGY, _some(spread)))
    if few:
        skipped.append("%d energ%s with fewer than 3 events (%s GeV) %s left out, as sigma90 "
                       "needs more" % (len(few), "y" if len(few) == 1 else "ies", _some(few),
                                       "is" if len(few) == 1 else "are"))
    if unmeasured:
        skipped.append("%d energ%s whose mu90 isn't above 0 (%s GeV) %s left out"
                       % (len(unmeasured), "y" if len(unmeasured) == 1 else "ies",
                          _some(unmeasured), "is" if len(unmeasured) == 1 else "are"))
    if points and left_out > true.size / 2:
        skipped.append("so few of the events are at fixed energies (%d of %d) that none are "
                       "plotted" % (true.size - left_out, true.size))
        points = []
    return points, skipped


def fit_terms(figure, energy):
    """[point, term] of the function fitted in a figure of the energy, which
    is the sum of its parameters times these terms of the energy E: E and 1
    for the linearity (slope E + intercept), 1, 1/sqrt(E) and 1/E for the
    resolution (a + b/sqrt(E) + c/E)."""
    import numpy as np
    energy = np.asarray(energy, dtype=float)
    if figure == "linearity":
        return np.stack([energy, np.ones_like(energy)], axis=1)
    return np.stack([np.ones_like(energy), 1 / np.sqrt(energy), 1 / energy], axis=1)


def least_squares(design, values, errors):
    """The parameters p that minimise sum(((design p - values) / errors)^2),
    their uncertainties (from the errors as they are), chi^2 and the degrees
    of freedom; None if the points can't fix every parameter. Without positive
    errors for every point, the points weigh the same, and the scatter about
    the fit gives the uncertainties (chi^2 is then nan)."""
    import numpy as np
    values = np.asarray(values, dtype=float)
    errors = np.asarray(errors, dtype=float)
    weighted = bool(np.all(np.isfinite(errors) & (errors > 0)))
    weights = 1 / errors if weighted else np.ones_like(values)
    a = design * weights[:, None]
    b = values * weights
    if np.linalg.matrix_rank(a) < design.shape[1]:
        return None
    params = np.linalg.lstsq(a, b, rcond=None)[0]
    covariance = np.linalg.inv(a.T @ a)
    ndf = len(values) - design.shape[1]
    residuals = a @ params - b
    # (with no degrees of freedom the fit goes through every point)
    chi2 = float(residuals @ residuals) if ndf > 0 else 0.0
    if not weighted:
        covariance = covariance * (chi2 / ndf if ndf > 0 else float("nan"))
        chi2 = float("nan")
    return params, np.sqrt(np.diag(covariance)), chi2, ndf


def fit_text(terms, params):
    """A fitted function written out, to 4 significant figures (a slope near
    1 needs them), e.g. 0.9832 $E$ - 0.1234 GeV (with a minus sign, U+2212)."""
    text = ""
    for i, (term, value) in enumerate(zip(terms, params)):
        number = term % ("%.4g" % abs(value))
        if i == 0:
            text = ("\u2212" if value < 0 else "") + number
        else:
            text += (" \u2212 " if value < 0 else " + ") + number
    return text


def aligned(reference, points):
    """[(reference point, point)] of the points at the reference's energies."""
    pairs = []
    for point in points:
        match = next((r for r in reference
                      if abs(r["energy"] - point["energy"]) <= SAME_ENERGY * abs(r["energy"])),
                     None)
        if match is not None:
            pairs.append((match, point))
    return pairs


def plot_energy(observable, scenario, measured, reference, metric_name, out, suffix, formats):
    """The energy linearity and resolution figures of a scenario, from
    measured, [(simulator, true energies, reconstructed energies)]; returns
    ([(simulator, figure, metric, value)], [(simulator, figure, parameter,
    value, uncertainty, chi^2, degrees of freedom)])."""
    entries = []
    for simulator, true, reconstructed in measured:
        points, skipped = energy_points(true, reconstructed)
        for why in skipped:
            say("%s, %s: %s" % (simulator.name, scenario, why))
        if not points:
            say("%s has no energy of %s to plot, so it's left out" % (simulator.name, scenario))
            continue
        entries.append((simulator, points))
    if not any(simulator is reference for simulator, _ in entries):
        reference = None
    results, fits = [], []
    for spec in ENERGY_FIGURES if entries else ():
        base = figure_base(out, observable, spec[0], scenario, suffix)
        figure_results, figure_fits = plot_energy_figure(spec, scenario, entries, reference,
                                                         metric_name, base, formats)
        results += [(simulator, spec[0], short, value)
                    for simulator, short, value in figure_results]
        fits += [(simulator, spec[0]) + fit for simulator, fit in figure_fits]
        say("%s: %s" % (", ".join(simulator.label for simulator, _ in entries),
                        ", ".join("%s.%s" % (os.path.basename(base), f) for f in formats)))
    return results, fits


def plot_energy_figure(spec, scenario, entries, reference, metric_name, out_base, formats):
    """One figure of the energy (a spec of ENERGY_FIGURES): every simulator's
    points against the true energy, each with its uncertainty and the function
    fitted to them, and below, each model's relative deviation from Geant4.
    Returns ([(simulator, metric, value)], [(simulator, (parameter, value,
    uncertainty, chi^2, degrees of freedom))])."""
    from matplotlib.lines import Line2D
    import numpy as np

    figure, value, ylabel, parameters, terms = spec
    short, metric = METRICS.get(metric_name, (metric_name, None))
    if metric is None:
        say("no metric '%s' is known (known: %s), so none is shown"
            % (metric_name, ", ".join(METRICS)))

    def column(points, key):
        return np.array([point[key] for point in points], dtype=float)

    ref = next((points for simulator, points in entries if simulator is reference), None)
    with_ratio = ref is not None and len(entries) > 1
    fig, axes = new_figure(with_ratio)
    top = axes[0][0]
    # Each simulator's colour, line style, marker, its size and layer: Geant4
    # in grey under the models, its markers big enough to show around theirs
    styles, models = [], 0
    for simulator, _ in entries:
        if simulator is reference:
            styles.append((REFERENCE_EDGE, "-", "o", 8.5, 3))
        else:
            styles.append((simulator.color, simulator.linestyle,
                           MARKERS[models % len(MARKERS)], 5.5, 4 + models))
            models += 1

    keyed, results, fits = [], [], []
    for (simulator, points), (color, linestyle, marker, size, layer) in zip(entries, styles):
        model = simulator is not reference
        x, y, err = column(points, "energy"), column(points, value), column(points, value + "_err")
        drawn = [top.errorbar(x, y, yerr=err, fmt=marker, color=color, markersize=size,
                              markeredgecolor="white", markeredgewidth=0.8, elinewidth=1.0,
                              capsize=0, zorder=layer)]
        label = simulator.label
        if ref is not None and simulator is not reference and metric is not None:
            pairs = aligned(ref, points)
            if pairs:
                result = metric([r[value] for r, _ in pairs], [p[value] for _, p in pairs])
                results.append((simulator, short, result))
                label = "%s (%s %.2e)" % (label, short, result)
        fit = (least_squares(fit_terms(figure, x), y, err)
               if len(x) >= len(parameters) else None)
        if fit is None:
            say("%s, %s: %d energ%s can't fix the %d parameters of the %s fit, so it isn't "
                "fitted" % (simulator.name, scenario, len(x), "y" if len(x) == 1 else "ies",
                            len(parameters), figure))
        else:
            params, uncertainties, chi2, ndf = fit
            line = np.linspace(x.min(), x.max(), 200)
            drawn += top.plot(line, fit_terms(figure, line) @ params, color=color,
                              linestyle=linestyle, linewidth=1.2 if simulator is reference else 1.4,
                              zorder=layer - 0.5)
            label += ": " + fit_text(terms, params)
            fits += [(simulator, (name, float(p), float(u), chi2, ndf))
                     for name, p, u in zip(parameters, params, uncertainties)]
        if model:
            mark(simulator, DATA, drawn)
        keyed.append((Line2D([], [], color=color, linestyle=linestyle if fit else "none",
                             linewidth=1.4, marker=marker, markersize=size,
                             markeredgecolor="white", markeredgewidth=0.8),
                      label, simulator if model else None))

    highest = max(float(np.max(column(points, value) + column(points, value + "_err")))
                  for _, points in entries)
    top.set_ylim(0, 1.25 * highest if highest > 0 else 1)
    top.set_xlim(0, 1.05 * max(points[-1]["energy"] for _, points in entries))
    top.set_ylabel(ylabel, color=INK)
    counts = [point["n"] for point in (ref or entries[0][1])]
    left = figure == "resolution"  # where the figure is empty, below the points
    top.text(0.02 if left else 0.98, 0.03, "%s, %s events per energy%s" % (
        scenario, min(counts) if min(counts) == max(counts) else "%d\u2013%d" % (
            min(counts), max(counts)), " (Geant4)" if ref is not None else ""),
        transform=top.transAxes, fontsize=8, color=MUTED, va="bottom",
        ha="left" if left else "right")
    legend(top, keyed, fontsize=7.5, loc="upper right" if left else "upper left")

    if with_ratio:
        bottom = axes[1][0]
        r_x, r_y, r_err = column(ref, "energy"), column(ref, value), column(ref, value + "_err")
        r_band = np.where(r_y > 0, r_err / np.where(r_y > 0, r_y, 1) * 100, np.nan)
        if len(r_x) > 1:
            bottom.fill_between(r_x, -r_band, r_band, color=REFERENCE_EDGE, alpha=0.3,
                                linewidth=0)
        else:
            bottom.errorbar(r_x, [0], yerr=r_band, fmt="none", ecolor=REFERENCE_EDGE)
        bottom.axhline(0, color=REFERENCE_EDGE, linewidth=1)
        spread = [r_band]
        for (simulator, points), (color, linestyle, marker, size, layer) in zip(entries, styles):
            pairs = aligned(ref, points) if simulator is not reference else []
            if not pairs:
                continue
            base = np.array([r[value] for r, _ in pairs])
            dev = (np.array([p[value] for _, p in pairs]) / base - 1) * 100
            band = np.array([p[value + "_err"] for _, p in pairs]) / base * 100
            energy = [p["energy"] for _, p in pairs]
            # The line apart from the points, so the error bars stay solid
            mark(simulator, DATA,
                 bottom.plot(energy, dev, color=color, linestyle=linestyle, linewidth=1.0,
                             zorder=layer - 0.5),
                 bottom.errorbar(energy, dev, yerr=band, fmt=marker, color=color,
                                 markersize=size - 1, markeredgecolor="white",
                                 markeredgewidth=0.8, elinewidth=1.0, capsize=0, zorder=layer))
            spread.append(np.abs(dev) + band)
        spread = np.concatenate(spread)
        spread = spread[np.isfinite(spread)]
        limit = float(np.clip(1.25 * spread.max(), 2, 100)) if spread.size else 10.0
        bottom.set_ylim(-limit, limit)
        bottom.set_ylabel("Rel. dev. [%]", color=INK)
        bottom.set_xlabel(ENERGY_XLABEL, color=INK)
    else:
        top.set_xlabel(ENERGY_XLABEL, color=INK)

    finish(fig, axes, out_base, formats)
    return results, fits


def energy_figures(observable, scenario, simulators, reference, settings, out, suffix, formats):
    """The energy linearity and resolution figures (plot_energy()) of the
    simulators that measured the energy."""
    measured = [(simulator,) + simulator.events(observable, scenario) for simulator in simulators]
    return plot_energy(observable, scenario, measured, reference, metric_of(observable, settings),
                       out, suffix, formats)


# ---------------------------------------------------------------------------
# The intrinsic angles: the residuals, reconstructed - true, of theta and phi
# of each shower's axis (the observable intrinsic_angle, ddml_bench/measure.py)

ANGLE_LABELS = {
    "theta": r"Intrinsic angle $\theta_\mathrm{reco} - \theta_\mathrm{true}$" + " [\u00b0]",
    "phi": r"Intrinsic angle $\phi_\mathrm{reco} - \phi_\mathrm{true}$" + " [\u00b0]",
}


def hits_text(name):
    """The hits of a PCA, from its name: all hits, or e.g. the top 4% of hits for top4."""
    return "all hits" if name == "all" else "the top %s%% of hits" % name[len("top"):]


def measured_box(simulator, observable, scenario):
    """The side (mm) of the bounding box a simulator's intrinsic angles were
    measured in, by the settings its .npz file records (measure.bounding_box());
    None for every hit, or if the file doesn't say."""
    data = getattr(simulator, "data", None) or {}
    stored = data.get(SEP.join((observable, scenario, "settings")))
    try:
        side = json.loads(str(stored)).get("bounding_box") if stored is not None else None
    except (ValueError, AttributeError):
        return None
    return float(side) if isinstance(side, (int, float)) and not isinstance(side, bool) else None


def residual_histogram(residuals, edges):
    """The fraction of the showers with an axis (a finite residual) in each
    bin and its statistical uncertainty, the number of them, and the fraction
    of them outside the bins; None if there are none."""
    import numpy as np
    residuals = np.asarray(residuals, dtype=float)
    residuals = residuals[np.isfinite(residuals)]
    if not residuals.size:
        return None
    counts = np.histogram(residuals, bins=edges)[0].astype(float)
    n = residuals.size
    return counts / n, np.sqrt(counts) / n, n, 1 - counts.sum() / n


def angle_figures(observable, scenario, simulators, reference, settings, out, suffix, formats):
    """The figures of the intrinsic angles of a scenario, of the simulators that
    measured them: for each PCA and each angle, the distribution of the
    residuals, in the bins of the settings (measure.residual_binning(); its
    defaults without settings). Returns ([(simulator, figure, metric, value)],
    []), each figure named <angle>_<PCA>, e.g. theta_top4."""
    import numpy as np
    default = (-measure.MAX_RESIDUAL, measure.MAX_RESIDUAL, measure.RESIDUAL_BINS)
    binning = {}
    if settings:
        try:
            binning = measure.residual_binning(observable, settings)
        except measure.ObservableError as error:
            say("%s, so the residuals are drawn from %g to %g degrees in %d bins"
                % ((error,) + default))
    stored = [(simulator, simulator.stored(observable, scenario)) for simulator in simulators]
    # The bounding box each simulator's hits were taken from
    boxes = dict((simulator.name, measured_box(simulator, observable, scenario))
                 for simulator, _ in stored)
    if len(set(boxes.values())) > 1:
        say("%s: the intrinsic angles weren't all measured in the same bounding box (%s), so "
            "the figures compare showers cut differently" % (scenario, ", ".join(
                "%s: %s" % (name, "none" if side is None else "%g mm" % side)
                for name, side in sorted(boxes.items()))))
    # The PCAs, in the order they were measured in
    selections = []
    for _, values in stored:
        for what in values:
            name = what[len("dtheta_"):]
            if what.startswith("dtheta_") and name not in selections:
                selections.append(name)
    metric_name = metric_of(observable, settings)
    results = []
    for selection in selections:
        first, last, n_bins = binning.get(selection, default)
        edges = np.linspace(first, last, n_bins + 1)
        for angle in measure.ANGLES:
            what = "d%s_%s" % (angle, selection)
            curves, outside = [], []
            for simulator, values in stored:
                if what not in values:
                    continue
                without = int(np.sum(~np.isfinite(values[what])))
                if without:
                    say("%s, %s: %d of %d showers have no axis from the PCA of %s (fewer "
                        "than two hits%s), so they're left out" % (
                            simulator.name, scenario, without, len(values[what]),
                            hits_text(selection),
                            "" if boxes[simulator.name] is None else " in the bounding box"))
                histogram = residual_histogram(values[what], edges)
                if histogram is None:
                    continue
                fractions, errors, n, beyond = histogram
                curves.append((simulator, edges, fractions, errors, n))
                outside.append(beyond)
            if not curves:
                say("no shower of %s has an axis from the PCA of %s, so there's no plot"
                    % (scenario, hits_text(selection)))
                continue
            profile = "%s_%s" % (angle, selection)
            base = figure_base(out, observable, profile, scenario, suffix)
            # The text in the corner is of Geant4, if it is there
            ref = reference if any(c[0] is reference for c in curves) else None
            shown = next((i for i, c in enumerate(curves) if c[0] is ref), 0)
            side = boxes[curves[shown][0].name]
            note = "%s, PCA of %s%s\n%d showers%s" % (
                scenario, hits_text(selection),
                "" if side is None else "\nin a box of side %g mm" % side, curves[shown][4],
                " (%s)" % ref.label if ref is not None else "")
            if outside[shown] > 0:
                note += "\n%.2g%% beyond \u00b1%g\u00b0" % (100 * outside[shown], last)
            figure_results = plot_profile(observable, scenario, profile, curves, ref, metric_name,
                                          base, formats, xlabel=ANGLE_LABELS[angle], note=note,
                                          trim=False, room=True)
            results += [(simulator, profile, short, value)
                        for simulator, short, value in figure_results]
            say("%s: %s" % (", ".join(c[0].label for c in curves),
                            ", ".join("%s.%s" % (os.path.basename(base), f) for f in formats)))
    return results, []


# ---------------------------------------------------------------------------
# The di-photon separation: the mean number of reconstructed photons against
# the separation of the two incident photons on the ECAL face (the observable
# photon_separation, ddml_bench/measure.py)

SEPARATION_XLABEL = r"Separation on the ECAL face $d_{\gamma\gamma}$ [mm]"
# The number of photons an event of two photons should be reconstructed as,
# drawn as a line across the figures
PHOTONS_PER_EVENT = 2


def event_seeds(simulator, scenario, n_events):
    """The ddsim seed of each of a scenario's events (as the observables stored
    per event have them), from the seed of each file (<scenario>__seeds, -1
    where ddsim chose it) and its number of events (<scenario>__n_events); all
    -1 if the measurements don't record them."""
    import numpy as np
    data = getattr(simulator, "data", None) or {}
    counts = data.get(SEP.join((scenario, "n_events")))
    seeds = data.get(SEP.join((scenario, "seeds")))
    if counts is None or seeds is None or len(seeds) != len(counts) \
            or int(np.sum(counts)) != n_events:
        return np.full(n_events, -1, dtype=np.int64)
    return np.repeat(np.asarray(seeds, dtype=np.int64), np.asarray(counts, dtype=np.int64))


def separation_points(separation, photons, seeds, edges):
    """The mean number of reconstructed photons per event in each bin of the
    separation, from the events of each ddsim seed: a dict of "mean", the mean
    over the seeds of each seed's mean; "error", the standard deviation of
    those over the seeds (the sample's, ddof 1), or with one seed the
    statistical uncertainty of its mean; "spread", whether the error is the
    spread over the seeds; "seeds", the seeds in order; "photons", {seed: the
    photons in each bin}, which the metric compares; "events", {seed: its
    events}; and "outside", the fraction of the events outside the bins.
    Events without a separation (NaN) are left out."""
    import numpy as np
    separation = np.asarray(separation, dtype=float)
    photons = np.asarray(photons, dtype=float)
    seeds = np.asarray(seeds)
    found = np.isfinite(separation)
    order = sorted(set(int(s) for s in seeds[found]))
    means, sums, sums_sq, counts = [], [], [], []
    points = {"seeds": order, "photons": {}, "events": {}}
    for seed in order:
        chosen = found & (seeds == seed)
        n = np.histogram(separation[chosen], edges)[0].astype(float)
        total = np.histogram(separation[chosen], edges, weights=photons[chosen])[0]
        total_sq = np.histogram(separation[chosen], edges, weights=photons[chosen] ** 2)[0]
        with np.errstate(divide="ignore", invalid="ignore"):
            means.append(np.where(n > 0, total / n, np.nan))
        sums.append(total)
        sums_sq.append(total_sq)
        counts.append(n)
        points["photons"][seed] = total
        points["events"][seed] = int(chosen.sum())
    n_bins = len(edges) - 1
    means = np.array(means).reshape(len(order), n_bins)
    have = np.isfinite(means)
    k = have.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = np.where(k > 0, np.where(have, means, 0).sum(axis=0) / k, np.nan)
        if len(order) > 1:
            deviation = np.where(have, means - mean, 0)
            error = np.where(k > 1, np.sqrt((deviation ** 2).sum(axis=0) / (k - 1)), np.nan)
        elif order:
            n, total, total_sq = counts[0], sums[0], sums_sq[0]
            variance = np.clip(total_sq / n - (total / n) ** 2, 0, None)
            error = np.where(n > 0, np.sqrt(variance / n), np.nan)
        else:
            error = np.full(n_bins, np.nan)
    events = sum(points["events"].values())
    beyond = events - sum(float(c.sum()) for c in counts)
    points.update(mean=mean, error=error, spread=len(order) > 1,
                  outside=beyond / events if events else 0.0)
    return points


def seed_metric(metric, reference, points, key="photons"):
    """The metric of what is in each bin (key: the photons of a simulator's
    events, or the counts of its pi0s) from the reference's, seed by seed (the
    seeds of each in order, paired): the mean of its values, their standard
    deviation (the sample's, ddof 1; nan with fewer than two pairs), and the
    number of pairs."""
    import numpy as np
    values = [metric(reference[key][r], points[key][s])
              for r, s in zip(reference["seeds"], points["seeds"])]
    values = [value for value in values if np.isfinite(value)]
    if not values:
        return float("nan"), float("nan"), 0
    spread = float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
    return float(np.mean(values)), spread, len(values)


def energy_name(energy):
    """A figure's name for photons of an energy, GeV: 20GeV, 12.5GeV, 100GeV."""
    return "%gGeV" % float("%.3g" % energy)


def plot_separation(observable, scenario, measured, reference, settings, out, suffix, formats):
    """The figures of the di-photon separation, from measured, [(simulator, and
    of each event its separation, number of reconstructed photons, photon
    energy and ddsim seed)]: one per energy of the photons,
    photon_separation_<E>GeV__<scenario>, with each simulator's mean number of
    reconstructed photons per event in each bin of the separation and the
    spread over the seeds as its band (separation_points()), drawn as the
    profiles are, with a line at two photons and each model's relative
    deviation from Geant4 below. The legend gives each model's metric (by
    default jensen_shannon) of the photons in the bins from Geant4's, the mean
    over the seeds and its spread (seed_metric()). Returns ([(simulator,
    figure, metric, value, uncertainty)], [])."""
    import numpy as np
    settings = settings or {}
    try:
        first, last, n_bins = measure.separation_binning(observable, settings)
    except measure.ObservableError as error:
        first, last, n_bins = 0.0, measure.MAX_SEPARATION, measure.SEPARATION_BINS
        say("%s, so the separation is drawn from 0 to %g mm in %d bins" % (error, last, n_bins))
    edges = np.linspace(first, last, n_bins + 1)
    metric_name = metric_of(observable, settings)
    short, metric = METRICS.get(metric_name, (metric_name, None))
    if metric is None:
        say("no metric '%s' is known (known: %s), so none is shown"
            % (metric_name, ", ".join(METRICS)))
    codes = sorted({abs(int(code)) for code in settings.get("pdg") or []})
    counted = "particles" if "pdg" in settings and codes != [22] else "photons"

    # The events of each simulator that have a separation, by the energy of
    # their photons
    groups = []  # [[energy, [(simulator, its events' separation, photons, seeds)]]]
    for simulator, separation, photons, energy, seeds in measured:
        separation = np.asarray(separation, dtype=float)
        energy = np.asarray(energy, dtype=float)
        found = np.flatnonzero(np.isfinite(separation) & np.isfinite(energy))
        if len(found) < len(separation):
            say("%s, %s: %d of %d events haven't two incident photons heading into the ECAL "
                "face, so they're left out" % (simulator.name, scenario,
                                              len(separation) - len(found), len(separation)))
        spread = []
        for index in same_energies(energy[found]) if len(found) else []:
            chosen = found[index]
            value = float(energy[chosen].mean())
            if energy[chosen].max() - energy[chosen].min() > SAME_ENERGY * abs(value):
                spread.append("%.4g to %.4g" % (energy[chosen].min(), energy[chosen].max()))
                continue
            group = next((g for g in groups if abs(g[0] - value) <= SAME_ENERGY * abs(g[0])),
                         None)
            if group is None:
                group = [value, []]
                groups.append(group)
            group[1].append((simulator, separation[chosen], np.asarray(photons)[chosen],
                             np.asarray(seeds)[chosen]))
        if spread:
            say("%s, %s: %d set(s) of events whose photons' energies differ by more than %g%% "
                "(%s GeV) are left out: the separation figures need inputs at fixed energies"
                % (simulator.name, scenario, len(spread), 100 * SAME_ENERGY, _some(spread)))

    results = []
    for energy, members in sorted(groups, key=lambda group: group[0]):
        profile = energy_name(energy)
        entries = [(simulator, separation_points(separation, photons, seeds, edges))
                   for simulator, separation, photons, seeds in members]
        entries = [(simulator, points) for simulator, points in entries if points["seeds"]]
        if not entries:
            continue
        ref = next((points for simulator, points in entries if simulator is reference), None)
        curves = [(simulator, edges, points["mean"], points["error"],
                   sum(points["events"].values())) for simulator, points in entries]
        texts = []
        for simulator, points in entries:
            if ref is None or simulator is reference or metric is None:
                continue
            value, spread, pairs = seed_metric(metric, ref, points)
            if not pairs:
                continue
            results.append((simulator, profile, short, value, spread))
            texts.append((simulator, "%s %.2e \u00b1 %.1e" % (short, value, spread)
                          if np.isfinite(spread) else "%s %.2e" % (short, value)))
        # The text in the corner is of Geant4, if it is there
        shown = ref if ref is not None else entries[0][1]
        events = sorted(set(shown["events"].values()))
        note = "%s, %s GeV photons\n%s events \u00d7 %d seed%s%s\nbands: %s" % (
            scenario, profile[:-len("GeV")],
            events[0] if len(events) == 1 else "%d\u2013%d" % (events[0], events[-1]),
            len(shown["seeds"]), "" if len(shown["seeds"]) == 1 else "s",
            " (%s)" % reference.label if ref is not None else "",
            "the spread over the seeds" if shown["spread"] else "statistical")
        if shown["outside"] > 0:
            note += "\n%.2g%% beyond %g mm" % (100 * shown["outside"], last)
        base = figure_base(out, observable, profile, scenario, suffix)
        plot_profile(observable, scenario, profile, curves,
                     reference if ref is not None else None, None, base, formats,
                     xlabel=SEPARATION_XLABEL, ylabel="Mean reconstructed %s per event" % counted,
                     note=note, trim=False, room=True, guide=PHOTONS_PER_EVENT,
                     metric_texts=texts, every_deviation=True)
        say("%s: %s" % (", ".join(c[0].label for c in curves),
                        ", ".join("%s.%s" % (os.path.basename(base), f) for f in formats)))
    return results, []


def separation_figures(observable, scenario, simulators, reference, settings, out, suffix,
                       formats):
    """The figures of the di-photon separation (plot_separation()) of the
    simulators that measured it."""
    measured = []
    for simulator in simulators:
        stored = simulator.stored(observable, scenario)
        separation = stored["separation"]
        measured.append((simulator, separation, stored["n_photons"], stored["energy"],
                         event_seeds(simulator, scenario, len(separation))))
    return plot_separation(observable, scenario, measured, reference, settings, out, suffix,
                           formats)


# ---------------------------------------------------------------------------
# The pi0s of the tau decays: the invariant mass of the good pi0 candidates'
# two photons, and their reconstructed energy less their MC pi0s' (the
# observable pi0, ddml_bench/measure.py), as in the paper's figure 11

PI0_LABELS = {
    "mass": r"Reconstructed $\pi^0$ mass $M_{\gamma\gamma}$ [GeV]",
    "energy": r"$E_{\pi^0}^\mathrm{reco} - E_{\pi^0}^\mathrm{MC}$ [GeV]",
}
PI0_YLABEL = r"Fraction of good $\pi^0$s per GeV"
# The categories of the pi0 reconstruction (the paper's figure 10), in the
# order drawn: the key pi0_categories() gives each, and its label
PI0_CATEGORIES = (("n_reco", "nRecoPi0"), ("n_fake", "nFake"), ("n_good", "nGood"),
                  ("n_confused", "nConfused"), ("n_missed", "nMissed"))
PI0_CATEGORY_FIGURE = "categories"
PI0_CATEGORY_YLABEL = r"$\pi^0$s per seed"


def seed_histograms(values, seeds, edges):
    """The distribution of the values, as their fraction in each bin per unit
    of the values, from the values of each ddsim seed: a dict of "mean", the
    mean over the seeds of each seed's fractions; "error", their standard
    deviation over the seeds (the sample's, ddof 1), or with one seed the
    statistical uncertainty; "spread", whether the error is the spread over
    the seeds; "seeds", the seeds in order; "counts", {seed: the values in each
    bin}, which the metric compares; "values", {seed: how many values it has};
    and "outside", the fraction of all of them outside the bins. A seed's
    fractions are of all its values, those outside the bins too; NaN values
    are left out."""
    import numpy as np
    values = np.asarray(values, dtype=float)
    seeds = np.asarray(seeds)
    found = np.isfinite(values)
    order = sorted(set(int(s) for s in seeds[found]))
    widths = bin_widths(edges)
    n_bins = len(edges) - 1
    fractions, counts, sizes = [], {}, {}
    for seed in order:
        chosen = values[found & (seeds == seed)]
        counts[seed] = np.histogram(chosen, edges)[0].astype(float)
        sizes[seed] = int(chosen.size)
        fractions.append(counts[seed] / chosen.size / widths)
    fractions = np.array(fractions).reshape(len(order), n_bins)
    if len(order) > 1:
        error = fractions.std(axis=0, ddof=1)
    elif order:
        error = np.sqrt(counts[order[0]]) / sizes[order[0]] / widths
    else:
        error = np.full(n_bins, np.nan)
    total = sum(sizes.values())
    inside = sum(float(c.sum()) for c in counts.values())
    return {"mean": fractions.mean(axis=0) if order else np.full(n_bins, np.nan),
            "error": error, "spread": len(order) > 1, "seeds": order, "counts": counts,
            "values": sizes, "outside": (total - inside) / total if total else 0.0}


def good_pi0s(simulator, observable, scenario, energy):
    """{figure: [good pi0]} of what the figures draw of a simulator's good pi0
    candidates (measure.GOOD): the invariant mass of their two photons, and
    their reconstructed energy (energy: the stored item, the fitted energy or
    the sum of the photons') less the MC pi0's; and the ddsim seed of each."""
    import numpy as np
    stored = simulator.stored(observable, scenario)
    counts = np.asarray(stored["n_candidates"], dtype=np.int64)
    seeds = np.repeat(event_seeds(simulator, scenario, len(counts)), counts)
    good = np.asarray(stored["match"]) == measure.GOOD
    return ({"mass": np.asarray(stored["mass"])[good],
             "energy": (np.asarray(stored[energy]) - np.asarray(stored["true_energy"]))[good]},
            seeds[good])

def pi0_categories(simulator, observable, scenario):
    """({category: [event]}, [event's ddsim seed]) of the reconstruction of a
    simulator's pi0s, as the paper's figure 10: of each event its candidates
    (n_reco), those good (measure.GOOD), confused (measure.CONFUSED) and fake
    (the rest, so n_fake = n_reco - n_good - n_confused), and its MC pi0s
    that no good or confused candidate is of (n_missed). Candidates can share
    photons, so a confused candidate can be of an MC pi0 that another is
    good of: an MC pi0 is counted found once, however many are of it."""
    import numpy as np
    stored = simulator.stored(observable, scenario)
    n_reco = np.asarray(stored["n_candidates"], dtype=np.int64)
    n_true = np.asarray(stored["n_true"], dtype=np.int64)
    n_events = len(n_reco)
    event = np.repeat(np.arange(n_events), n_reco)
    match = np.asarray(stored["match"])
    n_good = np.bincount(event[match == measure.GOOD], minlength=n_events)
    n_confused = np.bincount(event[match == measure.CONFUSED], minlength=n_events)
    # The MC pi0s found: each (event, true_index) of a good or confused candidate once
    found = (match == measure.GOOD) | (match == measure.CONFUSED)
    pairs = np.unique(np.stack([event[found],
                                np.asarray(stored["true_index"], dtype=np.int64)[found]]),
                      axis=1) if found.any() else np.zeros((2, 0), dtype=np.int64)
    n_found = np.bincount(pairs[0], minlength=n_events)
    return ({"n_reco": n_reco, "n_fake": n_reco - n_good - n_confused, "n_good": n_good,
             "n_confused": n_confused, "n_missed": n_true - n_found},
            event_seeds(simulator, scenario, n_events))


def category_totals(categories, seeds):
    """Of each category, its total over the events of each ddsim seed: a dict
    of "seeds", in order; "totals", {category: [seed's total]}; "mean" and
    "error", {category: the mean over the seeds, and their standard deviation
    (the sample's, ddof 1; with one seed the Poisson uncertainty)}; "spread",
    whether the error is the spread over the seeds; and "events", {seed: its
    events}."""
    import numpy as np
    seeds = np.asarray(seeds)
    order = sorted(set(int(s) for s in seeds))
    totals = {key: np.array([float(np.sum(values[seeds == seed])) for seed in order])
              for key, values in categories.items()}
    mean, error = {}, {}
    for key, values in totals.items():
        mean[key] = float(values.mean()) if order else float("nan")
        error[key] = (float(values.std(ddof=1)) if len(order) > 1 else
                      float(np.sqrt(values[0])) if order else float("nan"))
    return {"seeds": order, "totals": totals, "mean": mean, "error": error,
            "spread": len(order) > 1,
            "events": {seed: int(np.sum(seeds == seed)) for seed in order}}


def plot_pi0_categories(observable, scenario, entries, reference, out, suffix, formats):
    """The figure of the categories of the pi0 reconstruction,
    pi0_categories__<scenario>, from entries, [(simulator, category_totals())]:
    of each category (PI0_CATEGORIES) a bar of each simulator, its mean over
    the seeds with their spread as the error, Geant4 first; below, each
    model's relative deviation from Geant4 in each category, its error the
    model's spread, with Geant4's as the band. Returns [(simulator,
    figure, metric, value, uncertainty)]: of each model and category the
    relative deviation in %, figure categories_<category>."""
    import numpy as np
    ref = next((points for simulator, points in entries if simulator is reference), None)
    with_ratio = ref is not None and len(entries) > 1
    fig, axes = new_figure(with_ratio)
    top = axes[0][0]
    keys = [key for key, _ in PI0_CATEGORIES]
    x = np.arange(len(keys))
    # Geant4 first, then the models in their order
    ordered = sorted(entries, key=lambda entry: entry[0] is not reference)
    width = 0.8 / len(ordered)
    results, keyed = [], []
    for place, (simulator, points) in enumerate(ordered):
        shift = (place - (len(ordered) - 1) / 2) * width
        is_ref = simulator is reference
        color = REFERENCE_EDGE if is_ref else simulator.color
        mean = np.array([points["mean"][key] for key in keys])
        error = np.array([points["error"][key] for key in keys])
        bars = top.bar(x + shift, mean, width, color=REFERENCE_FILL if is_ref else color,
                       edgecolor=color, linewidth=1, alpha=1 if is_ref else 0.85,
                       label=simulator.label, yerr=error,
                       error_kw={"ecolor": INK, "elinewidth": 1, "capsize": 2})
        keyed.append((bars, simulator.label, None if is_ref else simulator))
        if not is_ref:
            mark(simulator, DATA, bars)
        if not with_ratio:
            continue
        if is_ref:
            r_mean = np.array([ref["mean"][key] for key in keys])
            r_err = np.array([ref["error"][key] for key in keys])
            safe = np.where(r_mean > 0, r_mean, np.nan)
            for i in x:
                band = r_err[i] / safe[i] * 100
                if np.isfinite(band):
                    axes[1][0].fill_between([i - 0.4, i + 0.4], -band, band,
                                            color=REFERENCE_EDGE, alpha=0.3, linewidth=0)
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            dev = (mean / safe - 1) * 100
            band = error / safe * 100
        mark(simulator, DATA,
             axes[1][0].errorbar(x + shift, dev, yerr=band, fmt="o", color=color, markersize=4,
                                 markeredgecolor="white", markeredgewidth=0.6, elinewidth=1,
                                 capsize=0))
        for key, value, uncertainty in zip(keys, dev, band):
            if np.isfinite(value):
                results.append((simulator, "%s_%s" % (PI0_CATEGORY_FIGURE, key), "dev%",
                                float(value), float(uncertainty)))
    top.set_ylabel(PI0_CATEGORY_YLABEL, color=INK)
    highest = max((points["mean"][key] + (points["error"][key]
                                          if np.isfinite(points["error"][key]) else 0))
                  for _, points in entries for key in keys)
    top.set_ylim(0, 1.3 * highest if highest > 0 else 1)
    legend(top, keyed, fontsize=7.5, loc="upper right")
    shown = ref if ref is not None else entries[0][1]
    events = sorted(set(shown["events"].values()))
    top.text(0.02, 0.97, "%s\n%s events × %d seed%s%s\nerrors: %s" % (
        scenario, events[0] if len(events) == 1 else "%d–%d" % (events[0], events[-1]),
        len(shown["seeds"]), "" if len(shown["seeds"]) == 1 else "s",
        " (%s)" % reference.label if ref is not None else "",
        "seed spread" if shown["spread"] else "statistical"),
        transform=top.transAxes, fontsize=8, color=MUTED, va="top", ha="left")
    bottom = axes[-1][0]
    bottom.set_xticks(x)
    bottom.set_xticklabels([label for _, label in PI0_CATEGORIES], color=INK)
    bottom.set_xlim(-0.5, len(keys) - 0.5)
    if with_ratio:
        bottom.axhline(0, color=REFERENCE_EDGE, linewidth=1)
        spread = [abs(r[3]) + r[4] for r in results if np.isfinite(r[4])]
        limit = float(np.clip(1.25 * max(spread), 2, 100)) if spread else 10.0
        bottom.set_ylim(-limit, limit)
        bottom.set_ylabel("Rel. dev. [%]", color=INK)
    base = figure_base(out, observable, PI0_CATEGORY_FIGURE, scenario, suffix)
    finish(fig, axes, base, formats)
    say("%s: %s" % (", ".join(simulator.label for simulator, _ in ordered),
                    ", ".join("%s.%s" % (os.path.basename(base), f) for f in formats)))
    return results


def pi0_figures(observable, scenario, simulators, reference, settings, out, suffix, formats):
    """The figures of the pi0s of a scenario, of the simulators that measured
    them, as the paper's figure 11: of the good candidates, the invariant mass
    of their two photons, pi0_mass__<scenario>, and their reconstructed energy
    (reconstructed_energy: fitted, or photons) less the MC pi0s',
    pi0_energy__<scenario>; each as the fraction of the good pi0s per GeV,
    the mean over the ddsim seeds with the spread over them as the band
    (seed_histograms()), drawn as the profiles are, with each model's relative
    deviation from Geant4 below. The legend gives each model's metric (by
    default jensen_shannon) of the numbers of good pi0s in the bins from
    Geant4's, seed by seed: the mean, with the spread over the seeds
    (seed_metric()). Then, as the paper's figure 10, the categories of the
    reconstruction (pi0_categories()) of every simulator,
    pi0_categories__<scenario> (plot_pi0_categories()). Returns ([(simulator,
    figure, metric, value, uncertainty)], [])."""
    import numpy as np
    settings = settings or {}
    try:
        binning = measure.pi0_binning(observable, settings)
    except measure.ObservableError as error:
        binning = dict(measure.PI0_BINNING)
        say("%s, so the pi0s are drawn in the bins of %s" % (error, binning))
    which = settings.get("reconstructed_energy") or "fitted"
    if which not in measure.RECONSTRUCTED_ENERGIES:
        say("observables.%s.reconstructed_energy is %r, so the fitted energy is drawn"
            % (observable, which))
        which = "fitted"
    metric_name = metric_of(observable, settings)
    short, metric = METRICS.get(metric_name, (metric_name, None))
    if metric is None:
        say("no metric '%s' is known (known: %s), so none is shown"
            % (metric_name, ", ".join(METRICS)))
    measured = []
    for simulator in simulators:
        values, seeds = good_pi0s(simulator, observable, scenario,
                                  measure.RECONSTRUCTED_ENERGIES[which])
        if not len(seeds):
            say("%s has no good pi0 of %s, so it's left out" % (simulator.name, scenario))
            continue
        measured.append((simulator, values, seeds))

    results = []
    for figure in ("mass", "energy"):
        first, last, n_bins = binning[figure]
        edges = np.linspace(first, last, n_bins + 1)
        entries = [(simulator, seed_histograms(values[figure], seeds, edges))
                   for simulator, values, seeds in measured]
        entries = [(simulator, points) for simulator, points in entries if points["seeds"]]
        if not entries:
            continue
        ref = next((points for simulator, points in entries if simulator is reference), None)
        curves = [(simulator, edges, points["mean"], points["error"],
                   sum(points["values"].values())) for simulator, points in entries]
        texts = []
        for simulator, points in entries:
            if ref is None or simulator is reference or metric is None:
                continue
            value, spread, pairs = seed_metric(metric, ref, points, key="counts")
            if not pairs:
                continue
            results.append((simulator, figure, short, value, spread))
            texts.append((simulator, "%s %.2e ± %.1e" % (short, value, spread)
                          if np.isfinite(spread) else "%s %.2e" % (short, value)))
        # The text in the corner is of Geant4, if it is there
        shown = ref if ref is not None else entries[0][1]
        sizes = sorted(set(shown["values"].values()))
        note = "%s, good $\\pi^0$s\n%s × %d seed%s%s%s\nbands: %s" % (
            scenario, sizes[0] if len(sizes) == 1 else "%d–%d" % (sizes[0], sizes[-1]),
            len(shown["seeds"]), "" if len(shown["seeds"]) == 1 else "s",
            " (%s)" % reference.label if ref is not None else "",
            "\n%s energy" % which if figure == "energy" else "",
            "seed spread" if shown["spread"] else "statistical")
        if shown["outside"] > 0:
            note += "\n%.2g%% outside the bins" % (100 * shown["outside"])
        base = figure_base(out, observable, figure, scenario, suffix)
        plot_profile(observable, scenario, figure, curves, reference if ref is not None else None,
                     None, base, formats, xlabel=PI0_LABELS[figure], ylabel=PI0_YLABEL,
                     note=note, trim=False, room=True, metric_texts=texts)
        say("%s: %s" % (", ".join(c[0].label for c in curves),
                        ", ".join("%s.%s" % (os.path.basename(base), f) for f in formats)))

    # The categories of the reconstruction, of every simulator that measured
    # the pi0s (one without a good pi0 too)
    entries = [(simulator, category_totals(*pi0_categories(simulator, observable, scenario)))
               for simulator in simulators]
    entries = [(simulator, points) for simulator, points in entries if points["seeds"]]
    if entries:
        ref = next((points for simulator, points in entries if simulator is reference), None)
        for simulator, points in entries:
            if ref is not None and simulator is not reference \
                    and sorted(points["events"].values()) != sorted(ref["events"].values()):
                say("%s, %s: its seeds have %s events and %s's %s, so its totals of the pi0 "
                    "categories aren't of the same events" % (
                        simulator.name, scenario, sorted(points["events"].values()),
                        reference.name, sorted(ref["events"].values())))
        results += plot_pi0_categories(observable, scenario, entries, reference, out, suffix,
                                       formats)
    return results, []


# How the observables stored per event are drawn: each is called with the
# observable, the scenario, the simulators that measured it, the reference,
# the observable's settings in the configuration ({} without one), the folder
# of the figures, the sample's suffix and the formats, and gives ([(simulator,
# figure, metric, value)], [(simulator, figure, parameter, value, uncertainty,
# chi^2, degrees of freedom)]); a metric may have its uncertainty after its
# value
EVENT_FIGURES = {"energy": energy_figures, "intrinsic_angle": angle_figures,
                 "photon_separation": separation_figures, "pi0": pi0_figures}


def uncertainty_text(row, form="%.6g"):
    """A metric's uncertainty, written out, if it has one: the seventh item of
    its row; empty without one, or if it couldn't be estimated (nan)."""
    import math
    error = row[6] if len(row) > 6 else None
    return form % error if error is not None and math.isfinite(error) else ""


def draw(folder, out, suffix, formats, observables, wanted=(), reference_name=REFERENCE):
    """Draw every figure of the measurements in folder (<simulator><suffix>.npz;
    of the simulators wanted, and the reference, or of all of them) into out,
    with metrics<suffix>.csv and fits<suffix>.csv there, comparing each to the
    reference. Returns the simulators drawn ([] if there are no measurements)."""
    paths = [path for name, path in measurement_files(folder, suffix)
             if not wanted or name in wanted or name == reference_name]
    if not paths:
        return []
    simulators = [Simulator(p) for p in paths]
    reference = next((s for s in simulators if s.name == reference_name), None)
    if reference is None:
        say("there's no %s measurement, so there's nothing to compare to; plotting "
            "without deviations" % reference_name)
    # The reference first, then the models in the order of their files
    if reference is not None:
        simulators.remove(reference)
        simulators.insert(0, reference)

    os.makedirs(out, exist_ok=True)
    if collector is not None:
        collector.section(out, reference)
    groups, event_groups = [], []
    for simulator in simulators:
        for key in simulator.profiles():
            if key not in groups:
                groups.append(key)
        for key in simulator.per_event():
            if key not in event_groups:
                event_groups.append(key)

    rows, fit_rows = [], []
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
        if collector is not None:
            collector.context(observable, scenario)
        base = figure_base(out, observable, profile, scenario, suffix)
        results = plot_profile(observable, scenario, profile, curves,
                               reference if any(c[0] is reference for c in curves) else None,
                               metric_of(observable, observables.get(observable)), base, formats)
        say("%s: %s" % (", ".join(c[0].label for c in curves),
                        ", ".join("%s.%s" % (os.path.basename(base), f) for f in formats)))
        for simulator, short, value in results:
            rows.append((observable, scenario, profile, simulator.name, short, value))

    # The observables stored per event: the energy linearity and resolution,
    # and the intrinsic angles
    for observable, scenario in event_groups:
        if observable not in EVENT_FIGURES:
            say("there are no figures of %s, so it isn't plotted" % observable)
            continue
        measured = [simulator for simulator in simulators
                    if (observable, scenario) in simulator.per_event()]
        if collector is not None:
            collector.context(observable, scenario)
        results, fits = EVENT_FIGURES[observable](observable, scenario, measured, reference,
                                                  observables.get(observable) or {}, out,
                                                  suffix, formats)
        for result in results:
            simulator, figure, short, value = result[:4]
            rows.append((observable, scenario, figure, simulator.name, short, value)
                        + tuple(result[4:5]))
        for simulator, figure, parameter, value, error, chi2, ndf in fits:
            fit_rows.append((observable, scenario, figure, simulator.name, parameter, value,
                             error, chi2, ndf))

    with open(os.path.join(out, "metrics%s.csv" % suffix), "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["observable", "scenario", "profile", "simulator", "metric", "value",
                         "uncertainty"])
        for row in rows:
            writer.writerow(row[:5] + ("%.6g" % row[5], uncertainty_text(row)))
    # The functions fitted in the energy figures, a row per parameter
    with open(os.path.join(out, "fits%s.csv" % suffix), "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["observable", "scenario", "profile", "simulator", "parameter", "value",
                         "error", "chi2", "ndf"])
        for row in fit_rows:
            writer.writerow(row[:5] + tuple("%.6g" % v for v in row[5:8]) + (row[8],))
    for row in rows:
        observable, scenario, profile, name, short, value = row[:6]
        error = uncertainty_text(row, "%.1e")
        print("%-22s %-14s %-10s %-14s %s %.3e%s" % (observable, scenario, profile, name, short,
                                                    value, " +- " + error if error else ""))
    for observable, scenario, profile, name, parameter, value, error, chi2, ndf in fit_rows:
        print("%-22s %-14s %-10s %-14s %-9s %.4g +- %.2g" % (
            observable, scenario, profile, name, parameter, value, error))
    say("figures, metrics%s.csv and fits%s.csv in %s" % (suffix, suffix, out))
    return simulators


# ---------------------------------------------------------------------------
# Closure tests: a model that is another's (the same weights through another
# inference backend, say), measured on the same events, compared with it


CLOSURE_DIR = "closure"
# How a model's and its twin's measurements of a scenario were seeded
SAME_SEEDS, DDSIM_SEEDS, OTHER_SEEDS = "same", "ddsim's own", "different"


def closure_pairs():
    """{model: (the model it is a closure test of, whether their events should
    be identical with the same seeds)}, from models/ (ddml_bench/models.py);
    {} if the models' descriptions can't be read."""
    try:
        from .models import load_models
        models = load_models()
    except Exception as error:  # no PyYAML, or a description that is wrong
        say("can't read the models' descriptions (%s), so no closure tests are plotted" % error)
        return {}
    return dict((m["name"], (m["closure"]["of"], m["closure"]["identical"]))
                for m in models if m.get("closure"))


def compare_scenario(first, second, scenario):
    """(seeds, identical) of two simulators' measurements of a scenario: how
    they were seeded (SAME_SEEDS: every file with the same ddsim seed, given;
    DDSIM_SEEDS: any chosen by ddsim; OTHER_SEEDS), and whether everything
    measured on it is the same, number for number (but the reconstructed
    files' names, which name the simulator)."""
    import numpy as np

    def keys(simulator):
        found = set()
        for key in simulator.data:
            parts = key.split(SEP)
            if (parts[0] == scenario or (len(parts) > 2 and parts[1] == scenario)) \
                    and key != SEP.join((scenario, "files")):
                found.add(key)
        return found

    seed_key = SEP.join((scenario, "seeds"))
    seeds = [s.data.get(seed_key) for s in (first, second)]
    if any(s is None or (np.asarray(s) < 0).any() for s in seeds):
        how = DDSIM_SEEDS
    elif np.array_equal(seeds[0], seeds[1]):
        how = SAME_SEEDS
    else:
        how = OTHER_SEEDS
    if keys(first) != keys(second):
        return how, False
    for key in keys(first):
        a, b = np.asarray(first.data[key]), np.asarray(second.data[key])
        nan = a.dtype.kind == "f" and b.dtype.kind == "f"
        if a.shape != b.shape or not np.array_equal(a, b, equal_nan=nan):
            return how, False
    return how, True


def closures(folder, out, suffix, formats, observables, wanted=(), pairs=None):
    """For each model that is a closure test of another (closure in
    models/<model>.yaml, or pairs, as closure_pairs() gives them), with both
    measured (and the model among those wanted, if any are): every figure of
    the two, the other as the reference, in <out>/closure/<model>/ with their
    metrics, and a row for each scenario of whether their measurements are the
    same, in <out>/closure<suffix>.csv. Their events should be the same where
    they are simulated with the same seeds, if closure.identical says so.
    Returns the rows."""
    measured = dict(measurement_files(folder, suffix))
    rows = []
    for model, (twin, identical) in sorted((closure_pairs() if pairs is None
                                            else pairs).items()):
        if (wanted and model not in wanted) or model not in measured or twin not in measured:
            continue
        say("closure test: %s against %s" % (model, twin))
        drawn = dict((s.name, s) for s in draw(folder, os.path.join(out, CLOSURE_DIR, model),
                                                suffix, formats, observables, [model], twin))
        if model not in drawn or twin not in drawn:
            say("the measurements of %s or %s name another simulator, so there's no closure "
                "test of them" % (model, twin))
            continue
        first, second = drawn[model], drawn[twin]
        scenarios = sorted({key.split(SEP)[0] for s in (first, second) for key in s.data
                            if key.endswith(SEP + "n_events")})
        for scenario in scenarios:
            seeds, same = compare_scenario(first, second, scenario)
            expected = identical and seeds == SAME_SEEDS
            rows.append((model, twin, scenario, seeds, "yes" if same else "no",
                         "identical" if expected else "statistical",
                         ("yes" if same else "no") if expected else ""))
            if expected and not same:
                say("closure test FAILED: %s and %s were simulated with the same seeds on %s, "
                    "but their measurements differ" % (model, twin, scenario))
            else:
                say("closure test: %s on %s: %s%s" % (
                    model, scenario, "the same as %s, number for number" % twin if same
                    else "not the same as %s, number for number" % twin,
                    "" if expected else " (seeds: %s; compare the figures in %s)" % (
                        seeds, os.path.join(out, CLOSURE_DIR, model))))
    if rows:
        path = os.path.join(out, "closure%s.csv" % suffix)
        with open(path, "w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["model", "reference", "scenario", "seeds", "identical", "expected",
                             "passed"])
            writer.writerows(rows)
        say("closure tests in %s and %s" % (path, os.path.join(out, CLOSURE_DIR)))
    return rows


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
    parser.add_argument("--no-closure", action="store_true",
                        help="leave out the closure tests (closure in models/*.yaml)")
    parser.add_argument("--no-dashboard", action="store_true",
                        help="don't write %s.html, every figure in one page with a menu to "
                             "show or hide each model" % dashboard.NAME)
    args = parser.parse_args(argv)

    folder, observables = load_settings() if not args.dir else (args.dir, load_settings()[1])
    if not folder:
        return 1
    suffix = "" if args.events is None else "_n%d" % args.events
    out = args.out or os.path.join(folder, PLOTS_DIR)
    formats = [f.strip() for f in args.format.split(",") if f.strip()]
    gathered = None if args.no_dashboard else dashboard.Collector(out)
    with collecting(gathered):
        if not draw(folder, out, suffix, formats, observables, args.simulators, args.reference):
            say("no measurements (*%s.npz) in %s; make them with "
                "./observables_from_reconstruction.sh" % (suffix, folder))
            return 1
        if not args.no_closure:
            closures(folder, out, suffix, formats, observables, args.simulators)
    if gathered is not None and gathered.figures:
        path = dashboard.write(gathered, dashboard.path(out, suffix), folder=folder,
                               events=args.events, reference_colors=(REFERENCE_FILL,
                                                                     REFERENCE_EDGE))
        say("every figure, with a menu to show or hide each model: %s" % path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
