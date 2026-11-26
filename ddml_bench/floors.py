"""The statistical floors of the metrics of a sample smaller than the
reference's, such as a model's low-cost sample (ddml_bench/low_cost.py),
which plot.py compares with Geant4's full sample.

A metric of a model measured on few events is the model's distance from
Geant4 plus the noise of its few events. Its floor at that size is what the
metric would be if the model were Geant4 itself, with as many events of each
file as the sample has: the noise alone. A model's metric within its floor
and the floor's spread is as close to Geant4 as that many events can tell.

The model and Geant4 simulate the same input events, so the noise of their
comparison is that of the showers (and of which events a sample has), not of
drawing other events: a pseudo-sample of the model's size is drawn
RESAMPLINGS times, compared with the reference as the figures compare them,
and the floor is the mean of those values, their standard deviation its
spread:

  observables stored per event   with Geant4's second sample (geant4_2: the
  (energy, intrinsic_angle,      same events, simulated with other seeds), as
  photon_separation, pi0)        many of its events of each file as the sample
                                 has, drawn without replacement (a pi0's
                                 candidates with their event), against
                                 Geant4's full sample as it is; without it,
                                 as many of Geant4's own events, drawn with
                                 replacement, against Geant4's drawn again
                                 likewise (which takes the spread of the
                                 events for noise too, so is higher where the
                                 inputs drive the metric, as the separation)
  profiles (radial_profile,      stored as each file's sums of the energy in
  longitudinal_profile)          each bin and of its square: each bin's sum
                                 over a file's n events drawn as a normal
                                 number about n times its mean, with n times
                                 its variance (the files are of one energy
                                 each, so the spread over the energies isn't
                                 taken for noise), the bins apart (their
                                 correlations within an event aren't stored),
                                 of the sample's size and of the reference's

The metrics are computed as plot.py computes them, with its functions: the
energy figures' L1 of mu90 and of the resolution at the same energies, the
residuals' and the good pi0s' distributions, the separation's photons per
bin, seed by seed. The floors are of the metrics that measure a distance,
the figures' own (JS, L1) and the Wasserstein distance (W1), not of the
chi^2, whose uncertainties already take both samples' sizes (it is about its
degrees of freedom whatever their sizes), nor of the pi0 categories'
deviations.

At the reference's own sample size, the floor is what Geant4's second sample
measures (plot.py takes it from its metrics), or without it the floors here
of the reference's own sizes. project() takes a metric of a smaller sample to
the reference's sample size: a divergence (JS, which grows as the square of
the differences) less its floor at the sample's size plus that at the
reference's; a distance (W1, L1) likewise in quadrature; and a chi^2's excess
over its degrees of freedom scaled by the sizes, as its uncertainties shrink
with them. These are first order: exact for the Jensen-Shannon divergence's
expected value, close for the rest.

A sample's files are matched to the reference's, and the second sample's, by
their input and ddsim seed (file_key()); one with no match leaves its
scenario without floors.
"""

import math
import os
import re

from . import measure, plot
from .layout import REC_SUFFIX, REC_SUFFIX_LCIO_ONLY, SECOND_SEED_OFFSET, SEED_TAG, SEP

# The samples drawn of each size, and the seed of the draws, the same every
# time so that the figures are too
RESAMPLINGS = 100
SEED = 20261120
# The metrics that have floors: the divergences, which grow as the square of
# the differences, and the distances, which grow as the differences do
DIVERGENCES = ("JS",)
DISTANCES = (plot.WASSERSTEIN, "L1")
# What a reconstruction's name has between its input's and its ddsim seed's,
# for a sample of the first N events of each input
_SAMPLE_TAG = re.compile(r"_n\d+(?=(?:%s\d+)?$)" % SEED_TAG)
_SEED = re.compile(r"%s(\d+)$" % SEED_TAG)


def file_key(path, second=False):
    """What a reconstructed file is of, whatever its simulator and sample:
    its input and ddsim seed, from its name (without its folder, its ending
    and its sample's _n<N>); with second, a file of Geant4's second sample,
    whose seeds are the first's plus SECOND_SEED_OFFSET (or that alone for
    ddsim's own), by the first's seed."""
    name = os.path.basename(str(path))
    for ending in (REC_SUFFIX, REC_SUFFIX_LCIO_ONLY):
        if name.endswith(ending):
            name = name[:-len(ending)]
            break
    name = _SAMPLE_TAG.sub("", name)
    if second:
        found = _SEED.search(name)
        if found and int(found.group(1)) >= SECOND_SEED_OFFSET:
            seed = int(found.group(1)) - SECOND_SEED_OFFSET
            name = name[:found.start()] + ("%s%d" % (SEED_TAG, seed) if seed else "")
    return name


def counts_in(reference, sample, scenario, second=False):
    """(the sample's events in each of the reference's files of the scenario,
    in the reference's order, why not or None): its files matched to the
    reference's by their inputs and seeds (file_key(); second for Geant4's
    second sample); None, and why, if one of its files has no match."""
    data, theirs = reference.data, sample.data
    files_key, events_key = SEP.join((scenario, "files")), SEP.join((scenario, "n_events"))
    if files_key not in data or events_key not in data:
        return None, "%s has no record of the files of %s" % (reference.name, scenario)
    if files_key not in theirs or events_key not in theirs:
        return None, "%s has no record of the files of %s" % (sample.name, scenario)
    index = dict((file_key(path), number) for number, path in enumerate(data[files_key]))
    counts = [0] * len(index)
    for path, events in zip(theirs[files_key], theirs[events_key]):
        number = index.get(file_key(path, second))
        if number is None:
            return None, ("%s's %s isn't of an input and seed that %s measured"
                          % (sample.name, os.path.basename(str(path)), reference.name))
        counts[number] += int(events)
    return counts, None


def second_counts(reference, second, scenario, counts):
    """The events of each of the second sample's files of the scenario, in
    its order, that a sample with counts of the reference's files has; None
    if its files and the reference's don't match one for one."""
    files = second.data.get(SEP.join((scenario, "files")))
    theirs = reference.data.get(SEP.join((scenario, "files")))
    if files is None or theirs is None or len(files) != len(theirs):
        return None
    index = dict((file_key(path), number) for number, path in enumerate(theirs))
    found = []
    for path in files:
        number = index.get(file_key(path, second=True))
        if number is None:
            return None
        found.append(counts[number])
    return found


class Resampled(object):
    """A simulator's measurements of a scenario stored per event, with each
    file's events drawn again, as many as counts says, with replacement or
    (replace False) without, as many as the file has at most (a pi0
    candidate with its event): what plot.py reads of them (data, stored(),
    events())."""

    def __init__(self, source, scenario, counts, rng, replace=True):
        import numpy as np
        self.name, self.label = source.name, source.label
        sizes = np.asarray(source.data[SEP.join((scenario, "n_events"))], dtype=np.int64)
        counts = np.where(sizes > 0, np.asarray(counts, dtype=np.int64), 0)
        if not replace:
            counts = np.minimum(counts, sizes)
        starts = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)
        picks = [start + (rng.integers(0, size, count) if replace
                          else np.sort(rng.choice(size, count, replace=False)))
                 for start, size, count in zip(starts, sizes, counts) if count > 0]
        events = np.concatenate(picks) if picks else np.zeros(0, dtype=np.int64)
        self.data = {SEP.join((scenario, "n_events")): counts,
                     SEP.join((scenario, "seeds")): np.asarray(
                         source.data.get(SEP.join((scenario, "seeds")),
                                         np.full(len(sizes), -1)))}
        candidates = {}  # observable: the candidates of the events drawn
        for key, value in source.data.items():
            parts = key.split(SEP)
            if len(parts) != 3 or parts[1] != scenario or parts[0] not in measure.PER_EVENT:
                continue
            observable, _, item = parts
            if item == "settings":
                self.data[key] = value
                continue
            per_object = measure.PER_CANDIDATE.get(observable)
            if per_object and item in per_object[1]:
                if observable not in candidates:
                    number = np.asarray(source.data[SEP.join((observable, scenario,
                                                              per_object[0]))], dtype=np.int64)
                    first = np.concatenate([[0], np.cumsum(number)[:-1]]).astype(np.int64)
                    lengths = number[events]
                    before = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
                    candidates[observable] = (np.repeat(first[events] - before, lengths)
                                              + np.arange(int(lengths.sum())))
                self.data[key] = np.asarray(value)[candidates[observable]]
            else:
                self.data[key] = np.asarray(value)[events]

    def stored(self, observable, scenario):
        prefix = SEP.join((observable, scenario)) + SEP
        return dict((key[len(prefix):], value) for key, value in self.data.items()
                    if key.startswith(prefix) and key[len(prefix):] != "settings")

    def events(self, observable, scenario):
        stored = self.stored(observable, scenario)
        return stored["true"], stored["reconstructed"]


# ---------------------------------------------------------------------------
# The metrics of the figures of what is stored per event, as plot.py computes
# them, without drawing: each gives {(figure, metric): value} of a simulator
# compared with the reference


def _metric(observable, settings):
    return plot.METRICS.get(plot.metric_of(observable, settings), (None, None))


def energy_metrics(observable, scenario, reference, simulator, settings):
    """The energy figures' metric (by default L1) of mu90 and of the
    resolution at the reference's energies (plot.plot_energy_figure())."""
    short, metric = _metric(observable, settings)
    if metric is None:
        return {}
    theirs = plot.energy_points(*reference.events(observable, scenario), errors=False)[0]
    mine = plot.energy_points(*simulator.events(observable, scenario), errors=False)[0]
    pairs = plot.aligned(theirs, mine)
    if not pairs:
        return {}
    return dict(((figure, short), metric([r[value] for r, _ in pairs],
                                         [p[value] for _, p in pairs]))
                for figure, value, _, _, _ in plot.ENERGY_FIGURES)


def angle_metrics(observable, scenario, reference, simulator, settings):
    """The residuals' metric and W1 of each angle and PCA
    (plot.angle_figures())."""
    import numpy as np
    short, metric = _metric(observable, settings)
    default = (-measure.MAX_RESIDUAL, measure.MAX_RESIDUAL, measure.RESIDUAL_BINS)
    try:
        binning = measure.residual_binning(observable, settings) if settings else {}
    except measure.ObservableError:
        binning = {}
    theirs = reference.stored(observable, scenario)
    mine = simulator.stored(observable, scenario)
    found = {}
    for what in theirs:
        if not what.startswith("dtheta_"):
            continue
        selection = what[len("dtheta_"):]
        first, last, n_bins = binning.get(selection, default)
        edges = np.linspace(first, last, n_bins + 1)
        for angle in measure.ANGLES:
            name = "d%s_%s" % (angle, selection)
            if name not in theirs or name not in mine:
                continue
            a = plot.residual_histogram(theirs[name], edges)
            b = plot.residual_histogram(mine[name], edges)
            if a is None or b is None:
                continue
            profile = "%s_%s" % (angle, selection)
            if metric is not None:
                found[(profile, short)] = metric(a[0], b[0])
            found[(profile, plot.WASSERSTEIN)] = plot.wasserstein(a[0], b[0], edges)
    return found


def separation_metrics(observable, scenario, reference, simulator, settings):
    """The metric and W1 of the photons in each bin of the separation, seed by
    seed, for each energy of the photons (plot.plot_separation())."""
    import numpy as np
    settings = settings or {}
    try:
        first, last, n_bins = measure.separation_binning(observable, settings)
    except measure.ObservableError:
        first, last, n_bins = 0.0, measure.MAX_SEPARATION, measure.SEPARATION_BINS
    edges = np.linspace(first, last, n_bins + 1)
    short, metric = _metric(observable, settings)
    measured = []
    for each in (reference, simulator):
        stored = each.stored(observable, scenario)
        measured.append((each, stored["separation"], stored["n_photons"], stored["energy"],
                         plot.event_seeds(each, scenario, len(stored["separation"]))))
    found = {}
    for energy, members in plot.separation_groups(measured)[0]:
        points = dict((each.name, plot.separation_points(separation, photons, seeds, edges))
                      for each, separation, photons, seeds in members)
        if reference.name not in points or simulator.name not in points \
                or reference is simulator:
            continue
        theirs, mine = points[reference.name], points[simulator.name]
        if not theirs["seeds"] or not mine["seeds"]:
            continue
        profile = plot.energy_name(energy)
        if metric is not None:
            value, _, pairs = plot.seed_metric(metric, theirs, mine)
            if pairs:
                found[(profile, short)] = value
        value, _, pairs = plot.seed_metric(lambda p, q: plot.wasserstein(p, q, edges), theirs,
                                           mine)
        if pairs:
            found[(profile, plot.WASSERSTEIN)] = value
    return found


def pi0_metrics(observable, scenario, reference, simulator, settings):
    """The metric and W1 of the good pi0s in the bins of each figure, seed by
    seed (plot.pi0_figures())."""
    import numpy as np
    settings = settings or {}
    try:
        binning = measure.pi0_binning(observable, settings)
    except measure.ObservableError:
        binning = dict(measure.PI0_BINNING)
    which = settings.get("reconstructed_energy") or "fitted"
    if which not in measure.RECONSTRUCTED_ENERGIES:
        which = "fitted"
    short, metric = _metric(observable, settings)
    good = [plot.good_pi0s(each, observable, scenario, measure.RECONSTRUCTED_ENERGIES[which])
            for each in (reference, simulator)]
    found = {}
    if not len(good[0][1]) or not len(good[1][1]):
        return found
    for figure, _, _ in plot.PI0_FIGURES:
        first, last, n_bins = binning[figure]
        edges = np.linspace(first, last, n_bins + 1)
        theirs, mine = [plot.seed_histograms(values[figure], seeds, edges)
                        for values, seeds in good]
        if not theirs["seeds"] or not mine["seeds"]:
            continue
        if metric is not None:
            value, _, pairs = plot.seed_metric(metric, theirs, mine, key="counts")
            if pairs:
                found[(figure, short)] = value
        value, _, pairs = plot.seed_metric(lambda p, q: plot.wasserstein(p, q, edges), theirs,
                                           mine, key="counts")
        if pairs:
            found[(figure, plot.WASSERSTEIN)] = value
    return found


# The metrics of the figures of each observable stored per event
EVENT_METRICS = {"energy": energy_metrics, "intrinsic_angle": angle_metrics,
                 "photon_separation": separation_metrics, "pi0": pi0_metrics}


# ---------------------------------------------------------------------------
# The floors


def profile_values(reference, observable, scenario, profile, sizes, counts, metric, resamplings,
                   rng):
    """{metric: [value]} of resamplings of one profile (see the module
    docstring): the reference's sizes against counts, each file's sums of
    each bin drawn as normal numbers."""
    import numpy as np
    key = SEP.join((observable, scenario, profile))
    sums = np.asarray(reference.data[key + SEP + "sum"], dtype=float)
    squares = np.asarray(reference.data[key + SEP + "sum_sq"], dtype=float)
    edges = reference.data[key + SEP + "edges"]
    sizes = np.asarray(sizes, dtype=float)[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = np.where(sizes > 0, sums / sizes, 0.0)
        variance = np.where(sizes > 0, np.clip(squares / sizes - mean ** 2, 0, None), 0.0)
    counts = np.where(sizes > 0, np.asarray(counts, dtype=float)[:, None], 0.0)

    def draw(number):
        noise = rng.standard_normal((resamplings,) + sums.shape)
        return (number * mean + np.sqrt(number * variance) * noise).sum(axis=1)

    theirs, mine = draw(sizes), draw(counts)
    short, function = metric
    found = {plot.WASSERSTEIN: [plot.wasserstein(a, b, edges) for a, b in zip(theirs, mine)]}
    if function is not None:
        found[short] = [function(a, b) for a, b in zip(theirs, mine)]
    return found


def floors(reference, counts, observables, second=None, resamplings=RESAMPLINGS, seed=SEED,
           per_event=True):
    """{(observable, scenario, figure, metric): (floor, spread)} of a sample
    with counts[scenario] events in each of the reference's files of each
    scenario (counts_in(); the reference's own sizes give the floors of its
    own sample size), observables the observables' settings, second
    Geant4's second sample if it is measured: of each figure the reference
    has (of its profiles alone, without per_event), the mean of its metrics
    (and W1) over resamplings, and their standard deviation (see the module
    docstring)."""
    import numpy as np
    rng = np.random.default_rng(seed)
    values = {}

    def keep(observable, scenario, found):
        for (figure, short), value in found.items():
            if value is not None and math.isfinite(value):
                values.setdefault((observable, scenario, figure, short), []).append(value)

    for scenario, wanted in counts.items():
        sizes = reference.data.get(SEP.join((scenario, "n_events")))
        if sizes is None or wanted is None or len(wanted) != len(sizes):
            continue
        for observable, where, profile in reference.profiles():
            if where != scenario:
                continue
            settings = observables.get(observable)
            found = profile_values(reference, observable, scenario, profile, sizes, wanted,
                                   _metric(observable, settings), resamplings, rng)
            for short, numbers in found.items():
                for value in numbers:
                    keep(observable, scenario, {(profile, short): value})
        stored = [observable for observable, where in reference.per_event()
                  if where == scenario and observable in EVENT_METRICS]
        if not stored or not per_event:
            continue
        # The second sample's events of the same files, where it measured
        # every observable of the scenario
        alike = None
        if second is not None and all((observable, scenario) in second.per_event()
                                      for observable in stored):
            alike = second_counts(reference, second, scenario, wanted)
        for _ in range(resamplings):
            if alike is not None:
                theirs = reference
                mine = Resampled(second, scenario, alike, rng, replace=False)
            else:
                theirs = Resampled(reference, scenario, sizes, rng)
                mine = Resampled(reference, scenario, wanted, rng)
            mine.name = mine.name + " (resampled)"
            for observable in stored:
                keep(observable, scenario, EVENT_METRICS[observable](
                    observable, scenario, theirs, mine, observables.get(observable) or {}))
    return dict((key, summarise(numbers)) for key, numbers in values.items())


def summarise(values):
    """(the mean, the standard deviation, ddof 1; nan with one value)."""
    import numpy as np
    values = np.asarray(values, dtype=float)
    if not values.size:
        return float("nan"), float("nan")
    return float(values.mean()), float(values.std(ddof=1)) if values.size > 1 else float("nan")


def project(metric, value, floor, full_floor, ratio=None, ndf=None):
    """A metric of a sample taken to the reference's sample size (see the
    module docstring): value, measured at the sample's size, where its floor
    is floor, and full_floor at the reference's; for a chi^2, ratio, the
    reference's events over the sample's, and ndf, its degrees of freedom
    (its floor at any size): its excess over them, scaled, on full_floor (the
    second sample's chi^2), or else on ndf. None where it can't be."""
    finite = [x for x in (value, floor, full_floor) if x is not None and math.isfinite(x)]
    if metric in DIVERGENCES:
        if len(finite) < 3:
            return None
        return max(value - floor, 0.0) + full_floor
    if metric in DISTANCES:
        if len(finite) < 3:
            return None
        return math.sqrt(max(value ** 2 - floor ** 2, 0.0) + full_floor ** 2)
    if metric == plot.CHI2:
        if value is None or not math.isfinite(value) or not ndf or ratio is None:
            return None
        base = full_floor if full_floor is not None and math.isfinite(full_floor) else ndf
        return base + max(value - ndf, 0.0) * chi2_scale(ratio)
    return None


def chi2_scale(ratio):
    """What a chi^2's excess over its degrees of freedom is multiplied by,
    from a sample's size to the reference's (ratio, the reference's events
    over the sample's): each point's uncertainties, squared and added, go as
    1/N_sample + 1/N_reference, which is 2/N_reference at the reference's
    size, so the excess grows by (ratio + 1) / 2."""
    return (ratio + 1.0) / 2.0
