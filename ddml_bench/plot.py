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

  python3 -m ddml_bench.plot [--dir DIR] [--out DIR] [--events N]
                             [--format pdf,png] [--reference NAME] [SIMULATOR ...]

(plot_observables.sh runs it.) The figures go to <dir>/plots (or --out), one per
observable, profile and scenario, named <observable>_<profile>__<scenario>.<format>
(energy_linearity__<scenario> and energy_resolution__<scenario> for the
energy), with the metrics in metrics.csv and the fitted parameters of the
energy figures, with their uncertainties, chi^2 and degrees of freedom, in
fits.csv. It needs numpy and matplotlib, and PyYAML only to read the
configuration for the default folder and the metrics, so it also works on .npz
files copied to a laptop.
"""

import argparse
import csv
import json
import os
import sys

from .layout import GEANT4, OBSERVABLES_DIR, PLOTS_DIR, SEP, figure_base, measurement_files

REFERENCE = GEANT4
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
PER_BIN = "Mean energy per event [GeV]"


def ring_areas(edges):
    """The area of each radial bin's ring around the shower axis, pi (r_out^2 -
    r_in^2): the area element of polar coordinates about the axis, r dr dphi
    (Jacobian r), integrated over the bin and all the way around."""
    import numpy as np
    edges = np.asarray(edges, dtype=float)
    return np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)


# The kinds of observable drawn as densities: what each bin's energy is divided
# by (from the bin edges), and the y axis label then. The rings of a radial
# profile grow with the radius (as 2 pi r dr), so the energy in each is divided
# by its area, giving the energy per unit area transverse to the shower axis.
# The other kinds are drawn per bin.
DENSITIES = {
    "radial_profile": (ring_areas, "Mean energy density per event [GeV/mm$^2$]"),
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
    """(observables folder, {observable: metric}) from the configuration, if it can be read."""
    try:
        from .config import load_config
        config = load_config()
    except Exception as error:  # no PyYAML, or no configuration
        say("can't read the configuration (%s), so give --dir" % error)
        return None, {}
    metrics = {name: (settings or {}).get("metric")
               for name, settings in (config.get("observables") or {}).items()}
    return os.path.join(config["paths"]["data_dir"], OBSERVABLES_DIR), metrics


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
        """[(observable, scenario)] of the observables stored per event."""
        found = []
        for key in self.data:
            parts = key.split(SEP)
            if len(parts) == 3 and parts[2] == "reconstructed" \
                    and SEP.join(parts[:2] + ["true"]) in self.data:
                found.append(tuple(parts[:2]))
        return found

    def events(self, observable, scenario):
        """The true and the reconstructed energy of each event, GeV."""
        key = SEP.join((observable, scenario))
        return self.data[key + SEP + "true"], self.data[key + SEP + "reconstructed"]


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


def metric_of(metrics, observable):
    return metrics.get(observable) or DEFAULT_METRICS.get(observable, "jensen_shannon")


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
    import numpy as np

    xlabel, log_y = STYLES.get(observable, (profile, False))
    ylabel = DENSITIES[observable][1] if observable in DENSITIES else PER_BIN
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

    if ref is not None:
        _, _, mean, err, n = ref
        mean, err = mean[keep] / scale, err[keep] / scale
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
        mean, err = mean / scale, err / scale
        top.stairs(drawable(mean, log_y), edges, color=simulator.color,
                   linestyle=simulator.linestyle, linewidth=1.6, label=label,
                   baseline=None, zorder=3)
        stairs_band(top, edges, mean - err, mean + err, simulator.color, 0.2)

    if log_y:
        top.set_yscale("log")
        drawn = np.concatenate([c[2][keep] / scale for c in curves])
        positive = drawn[drawn > 0]
        if positive.size:
            top.set_ylim(positive.min() / 3, positive.max() * 3)
    top.set_ylabel(ylabel, color=INK)
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
        fig.savefig("%s.%s" % (out_base, fmt), dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# The energy linearity and resolution, from the reconstructed energy of each
# event (the observable energy, ddml_bench/measure.py)

# Events whose true energies differ by less than this fraction of them are at
# the same energy
SAME_ENERGY = 1e-3
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

    handles, labels, results, fits = [], [], [], []
    for (simulator, points), (color, linestyle, marker, size, layer) in zip(entries, styles):
        x, y, err = column(points, "energy"), column(points, value), column(points, value + "_err")
        top.errorbar(x, y, yerr=err, fmt=marker, color=color, markersize=size,
                     markeredgecolor="white", markeredgewidth=0.8, elinewidth=1.0, capsize=0,
                     zorder=layer)
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
            top.plot(line, fit_terms(figure, line) @ params, color=color, linestyle=linestyle,
                     linewidth=1.2 if simulator is reference else 1.4, zorder=layer - 0.5)
            label += ": " + fit_text(terms, params)
            fits += [(simulator, (name, float(p), float(u), chi2, ndf))
                     for name, p, u in zip(parameters, params, uncertainties)]
        handles.append(Line2D([], [], color=color, linestyle=linestyle if fit else "none",
                              linewidth=1.4, marker=marker, markersize=size,
                              markeredgecolor="white", markeredgewidth=0.8))
        labels.append(label)

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
    top.legend(handles, labels, frameon=False, fontsize=7.5, labelcolor=INK,
               loc="upper right" if left else "upper left")

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
            bottom.plot(energy, dev, color=color, linestyle=linestyle, linewidth=1.0,
                        zorder=layer - 0.5)
            bottom.errorbar(energy, dev, yerr=band, fmt=marker, color=color,
                            markersize=size - 1, markeredgecolor="white",
                            markeredgewidth=0.8, elinewidth=1.0, capsize=0, zorder=layer)
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


# How the observables stored per event are drawn
EVENT_FIGURES = {"energy": plot_energy}


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
    paths = [path for name, path in measurement_files(folder, suffix)
             if not args.simulators or name in args.simulators or name == args.reference]
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

    out = args.out or os.path.join(folder, PLOTS_DIR)
    os.makedirs(out, exist_ok=True)
    formats = [f.strip() for f in args.format.split(",") if f.strip()]
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
        base = figure_base(out, observable, profile, scenario, suffix)
        results = plot_profile(observable, scenario, profile, curves,
                               reference if any(c[0] is reference for c in curves) else None,
                               metric_of(metrics, observable), base, formats)
        say("%s: %s" % (", ".join(c[0].label for c in curves),
                        ", ".join("%s.%s" % (os.path.basename(base), f) for f in formats)))
        for simulator, short, value in results:
            rows.append((observable, scenario, profile, simulator.name, short, value))

    # The observables stored per event: the energy linearity and resolution
    for observable, scenario in event_groups:
        if observable not in EVENT_FIGURES:
            say("there are no figures of %s, so it isn't plotted" % observable)
            continue
        measured = [(simulator,) + simulator.events(observable, scenario)
                    for simulator in simulators if (observable, scenario) in simulator.per_event()]
        results, fits = EVENT_FIGURES[observable](observable, scenario, measured, reference,
                                                  metric_of(metrics, observable), out, suffix,
                                                  formats)
        for simulator, figure, short, value in results:
            rows.append((observable, scenario, figure, simulator.name, short, value))
        for simulator, figure, parameter, value, error, chi2, ndf in fits:
            fit_rows.append((observable, scenario, figure, simulator.name, parameter, value,
                             error, chi2, ndf))

    with open(os.path.join(out, "metrics%s.csv" % suffix), "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["observable", "scenario", "profile", "simulator", "metric", "value"])
        for row in rows:
            writer.writerow(row[:5] + ("%.6g" % row[5],))
    # The functions fitted in the energy figures, a row per parameter
    with open(os.path.join(out, "fits%s.csv" % suffix), "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["observable", "scenario", "profile", "simulator", "parameter", "value",
                         "error", "chi2", "ndf"])
        for row in fit_rows:
            writer.writerow(row[:5] + tuple("%.6g" % v for v in row[5:8]) + (row[8],))
    for observable, scenario, profile, name, short, value in rows:
        print("%-22s %-14s %-10s %-14s %s %.3e" % (observable, scenario, profile, name, short,
                                                  value))
    for observable, scenario, profile, name, parameter, value, error, chi2, ndf in fit_rows:
        print("%-22s %-14s %-10s %-14s %-9s %.4g +- %.2g" % (
            observable, scenario, profile, name, parameter, value, error))
    say("figures, metrics%s.csv and fits%s.csv in %s" % (suffix, suffix, out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
