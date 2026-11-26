"""The statistical floors of a smaller sample's metrics (ddml_bench/floors.py):
how its files are matched to the reference's, the draws, the floors of a
known distribution, the metrics computed as the figures compute them, and
the projection to the full sample's size."""

import json
import math
import types

import numpy as np
import pytest

from ddml_bench import floors, plot
from test_plot import (angle_simulator, di_photon_sample, energy_sample, pi0_simulator,
                       residual_sample, simulators)

SCENARIO = "single_photon"


@pytest.fixture
def figures(monkeypatch):
    """The figures drawn, kept open (as tests/test_plot.py keeps them)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    close = plt.close
    kept = []
    monkeypatch.setattr(plt, "close", kept.append)
    yield kept
    for figure in kept:
        close(figure)


def measured(name, data, label=None):
    """A simulator's measurements, as plot.Simulator reads them from a file."""
    simulator = plot.Simulator.__new__(plot.Simulator)
    simulator.name, simulator.data, simulator.path = name, data, None
    simulator.label = simulator.base_label = label or name
    simulator.color, simulator.linestyle = "#000000", "-"
    simulator.sample, simulator.floors = None, {}
    return simulator


def angles(files, events, rng, folder="geant4", tag="", seed=None):
    """The intrinsic angles' residuals of each event of files of events
    each, uniform from -5 to 5 degrees (the figures' default bins: 100 of
    them), and the files' record."""
    total = files * events
    data = {"%s__n_events" % SCENARIO: np.full(files, events),
            "%s__seeds" % SCENARIO: np.full(files, -1 if seed is None else seed),
            "%s__files" % SCENARIO: np.array([
                "rec_output/%s/%s/input%d%s%s_REC.edm4hep.root" % (
                    SCENARIO, folder, number, tag,
                    "" if seed is None else "_simseed%d" % seed) for number in range(files)]),
            "intrinsic_angle__%s__settings" % SCENARIO: np.array("{}")}
    for angle in ("theta", "phi"):
        data["intrinsic_angle__%s__d%s_all" % (SCENARIO, angle)] = rng.uniform(-5, 5, total)
        data["intrinsic_angle__%s__%s_true" % (SCENARIO, angle)] = rng.uniform(40, 140, total)
    return data


def test_a_samples_files_are_matched_to_the_references_by_input_and_seed():
    key = floors.file_key
    assert key("rec_output/dp/geant4/di_seg1of3_simseed2_REC.edm4hep.root") == \
        key("rec_output/dp/mine/di_seg1of3_n1000_simseed2_REC.edm4hep.root") == \
        "di_seg1of3_simseed2"
    assert key("rec_output/sp/geant4/p22_en20_REC.edm4hep.root") == \
        key("rec_output/sp/mine/p22_en20_n1000_REC.edm4hep.root") == "p22_en20"
    # Geant4's second sample, by the first's seeds (ddsim's own for none)
    assert key("rec_output/dp/geant4_2/di_simseed1000002_REC.edm4hep.root", second=True) == \
        "di_simseed2"
    assert key("rec_output/sp/geant4_2/p22_en20_simseed1000000_REC.edm4hep.root",
               second=True) == "p22_en20"

    rng = np.random.default_rng(1)
    reference = measured("geant4", angles(3, 50, rng))
    sample = measured("mine", angles(3, 5, rng, "mine", "_n5"))
    # (in another order)
    for what in ("files", "n_events"):
        key_ = "%s__%s" % (SCENARIO, what)
        sample.data[key_] = sample.data[key_][::-1]
    sample.data["%s__n_events" % SCENARIO] = np.array([4, 5, 3])
    assert floors.counts_in(reference, sample, SCENARIO) == ([3, 5, 4], None)
    sample.data["%s__files" % SCENARIO][0] = "rec_output/x/mine/other_n5_REC.edm4hep.root"
    counts, why = floors.counts_in(reference, sample, SCENARIO)
    assert counts is None and "other_n5_REC.edm4hep.root isn't of an input and seed" in why


def test_a_pi0s_candidates_are_drawn_with_their_event():
    n_candidates = np.array([2, 0, 1, 3, 1])
    data = {"tau_pairs__n_events": np.array([3, 2]), "tau_pairs__seeds": np.array([42, 43]),
            # each event's number, and each candidate's its event's
            "pi0__tau_pairs__n_true": np.arange(5),
            "pi0__tau_pairs__n_candidates": n_candidates,
            "pi0__tau_pairs__mass": np.repeat(np.arange(5), n_candidates).astype(float),
            "pi0__tau_pairs__settings": np.array("{}")}
    source = measured("geant4", data)
    rng = np.random.default_rng(3)
    for replace in (True, False):
        drawn = floors.Resampled(source, "tau_pairs", [4, 1], rng, replace=replace)
        events = drawn.data["pi0__tau_pairs__n_true"]
        # Of each file, from that file's events
        assert drawn.data["tau_pairs__n_events"].tolist() == ([4, 1] if replace else [3, 1])
        assert set(events[:-1]) <= {0, 1, 2} and events[-1] in (3, 4)
        np.testing.assert_array_equal(drawn.stored("pi0", "tau_pairs")["mass"],
                                      np.repeat(events, n_candidates[events]))
    # Without replacement, all of a file's events at most, each once
    assert sorted(drawn.data["pi0__tau_pairs__n_true"][:3]) == [0, 1, 2]


def test_the_floor_of_a_distribution_is_that_of_its_bins():
    """Two samples of N and M events of one distribution over k bins have a
    Jensen-Shannon divergence of about (k - 1) / 8 (1/N + 1/M) (the README's
    (k - 1)/4N for N = M)."""
    rng = np.random.default_rng(5)
    reference = measured("geant4", angles(4, 5000, rng))
    found = floors.floors(reference, {SCENARIO: [500] * 4}, {"intrinsic_angle": {}})
    floor, spread = found[("intrinsic_angle", SCENARIO, "theta_all", "JS")]
    assert floor == pytest.approx(99 / 8.0 * (1 / 2000.0 + 1 / 20000.0), rel=0.12)
    # Its spread is the scatter of the metric, from statistics alone
    assert 0 < spread < floor / 2
    w1, _ = found[("intrinsic_angle", SCENARIO, "theta_all", plot.WASSERSTEIN)]
    assert w1 > 0
    # Of the reference's own size, half as much
    own, _ = floors.floors(reference, {SCENARIO: [5000] * 4}, {"intrinsic_angle": {}})[
        ("intrinsic_angle", SCENARIO, "theta_all", "JS")]
    assert own == pytest.approx(floor * (2 / 20000.0) / (1 / 2000.0 + 1 / 20000.0), rel=0.15)


def test_with_a_second_sample_its_events_of_the_same_inputs_are_drawn():
    rng = np.random.default_rng(7)
    reference = measured("geant4", angles(2, 300, rng))
    second = measured("geant4_2", angles(2, 300, rng, "geant4_2", seed=1000000))
    observables = {"intrinsic_angle": {}}
    # All of its events: what it measures itself, every time
    found = floors.floors(reference, {SCENARIO: [300, 300]}, observables, second=second)
    measured_ = floors.angle_metrics("intrinsic_angle", SCENARIO, reference, second, {})
    for (figure, metric), value in measured_.items():
        floor, spread = found[("intrinsic_angle", SCENARIO, figure, metric)]
        assert floor == pytest.approx(value) and spread == pytest.approx(0, abs=1e-12)
    # Fewer, of the same inputs: a floor above it
    fewer = floors.floors(reference, {SCENARIO: [30, 30]}, observables, second=second)
    assert fewer[("intrinsic_angle", SCENARIO, "theta_all", "JS")][0] > \
        found[("intrinsic_angle", SCENARIO, "theta_all", "JS")][0]


def test_a_profiles_floor_is_drawn_from_each_files_sums():
    """Profiles are stored as sums: each file's bins drawn about their means,
    with their variances, so a profile measured on many events has a small
    floor, and a sample of a tenth of them about ten times it."""
    rng = np.random.default_rng(11)
    events = np.array([2000, 2000])
    edges = np.linspace(0, 50, 26)
    shape = np.exp(-np.linspace(0, 5, 25))
    data = {"%s__n_events" % SCENARIO: events}
    sums, squares = [], []
    for n, scale in zip(events, (10.0, 50.0)):
        # each event's energy in each bin, exponential about its mean
        deposits = rng.exponential(scale * shape, (n, len(shape)))
        sums.append(deposits.sum(axis=0))
        squares.append((deposits ** 2).sum(axis=0))
    key = "radial_profile__%s__r50" % SCENARIO
    data.update({key + "__sum": np.array(sums), key + "__sum_sq": np.array(squares),
                 key + "__edges": edges, key + "__settings": np.array("{}")})
    reference = measured("geant4", data)
    observables = {"radial_profile": {"metric": "jensen_shannon"}}
    small, _ = floors.floors(reference, {SCENARIO: [200, 200]}, observables)[
        ("radial_profile", SCENARIO, "r50", "JS")]
    full, _ = floors.floors(reference, {SCENARIO: list(events)}, observables)[
        ("radial_profile", SCENARIO, "r50", "JS")]
    assert 0 < full < small
    assert small / full == pytest.approx((1 / 400.0 + 1 / 4000.0) / (2 / 4000.0), rel=0.25)


def test_the_metrics_are_those_of_the_figures(figures, tmp_path):
    """Each observable's metrics, computed without drawing, as its figures
    give them."""
    geant4, model = simulators()

    def rows(results):
        return dict(((row[1], row[2]), row[3]) for row in results if row[0] is not geant4)

    # The intrinsic angles
    g, m = residual_sample(1.0, 0.0, 1), residual_sample(1.1, 0.1, 2)
    first = angle_simulator("geant4", "Geant4", "#8c8c8c", "-", g)
    second = angle_simulator("model", "Model", "#e66100", "--", m)
    settings = {"max_residual": [4, 2], "n_bins": 40, "hit_fractions": [1.0, 0.04]}
    results, _ = plot.angle_figures("intrinsic_angle", SCENARIO, [first, second], first,
                                    settings, str(tmp_path), "", ["png"])
    found = floors.angle_metrics("intrinsic_angle", SCENARIO, first, second, settings)
    assert found == pytest.approx(dict((key, value) for key, value in rows(results).items()
                                       if key[1] in ("JS", plot.WASSERSTEIN)))

    # The energy figures' L1
    def with_events(simulator, true, reconstructed):
        simulator = types.SimpleNamespace(**vars(simulator))
        simulator.events = lambda observable, scenario: (true, reconstructed)
        return simulator
    first = with_events(geant4, *energy_sample(1.0, 0.15, 1, n=500))
    second = with_events(model, *energy_sample(0.97, 0.17, 2, n=500))
    results, _ = plot.energy_figures("energy", SCENARIO, [first, second], first, {},
                                     str(tmp_path), "", ["png"])
    found = floors.energy_metrics("energy", SCENARIO, first, second, {})
    assert found == pytest.approx(dict((key, value) for key, value in rows(results).items()
                                       if key[1] == "L1"))

    # The separation's, seed by seed
    def separations(simulator, columns, seeds):
        separation, photons, energy, _ = columns
        stored = {"separation": separation, "n_photons": photons, "energy": energy}
        return types.SimpleNamespace(
            name=simulator.name, label=simulator.label, color=simulator.color,
            linestyle=simulator.linestyle, stored=lambda observable, scenario: stored,
            data={"di_photon__n_events": np.full(len(seeds), len(separation) // len(seeds)),
                  "di_photon__seeds": np.array(seeds)})
    first = separations(geant4, di_photon_sample(20, 1, n=400), (1, 2, 3))
    second = separations(model, di_photon_sample(24, 2, n=400), (1, 2, 3))
    results, _ = plot.separation_figures("photon_separation", "di_photon", [first, second],
                                         first, {}, str(tmp_path), "", ["png"])
    found = floors.separation_metrics("photon_separation", "di_photon", first, second, {})
    assert found == pytest.approx(rows(results)) and len(found) == 2

    # The good pi0s', seed by seed
    first = pi0_simulator("geant4", "Geant4", "#8c8c8c", "-", 1.0, 1, n_events=300)
    second = pi0_simulator("model", "Model", "#e66100", "--", 0.98, 2, n_events=300)
    settings = {"reconstructed_energy": "photons"}
    results, _ = plot.pi0_figures("pi0", "tau_pairs", [first, second], first, settings,
                                  str(tmp_path), "", ["png"])
    found = floors.pi0_metrics("pi0", "tau_pairs", first, second, settings)
    assert found == pytest.approx(dict((key, value) for key, value in rows(results).items()
                                       if key[1] in ("JS", plot.WASSERSTEIN)))
    assert len(found) == 8


def test_a_metric_is_projected_to_the_full_samples_size():
    # A divergence: less its floor here, plus the floor there
    assert floors.project("JS", 0.010, 0.004, 0.001) == pytest.approx(0.007)
    # (below its floor, as close as statistics can tell: the floor there)
    assert floors.project("JS", 0.003, 0.004, 0.001) == pytest.approx(0.001)
    # A distance, in quadrature
    assert floors.project("W1", 5.0, 3.0, 1.0) == pytest.approx(math.sqrt(16 + 1))
    assert floors.project("L1", 2.0, 3.0, 1.0) == pytest.approx(1.0)
    # A chi^2: its excess over its degrees of freedom, as its uncertainties
    # shrink (4 times the events: (4 + 1) / 2), on the second sample's chi^2
    assert floors.project("chi2", 30.0, None, 9.0, ratio=4.0, ndf=10) == \
        pytest.approx(9.0 + 20.0 * 2.5)
    assert floors.project("chi2", 30.0, None, None, ratio=4.0, ndf=10) == \
        pytest.approx(10.0 + 50.0)
    assert floors.project("chi2", 8.0, None, 9.0, ratio=4.0, ndf=10) == pytest.approx(9.0)
    # Nothing without what it needs, nor of a deviation
    assert floors.project("JS", 0.01, None, 0.001) is None
    assert floors.project("chi2", 30.0, None, None) is None
    assert floors.project("dev%", 3.0, 1.0, 1.0) is None


def test_a_samples_label_says_its_events_of_each_input():
    data = {"single_photon__n_events": np.array([1000, 1000]),
            "tau_pairs__n_events": np.array([500, 500, 480]),
            "meta__label": np.array("Mine")}
    sample = measured("mine", data, "Mine")
    sample.sample = "_n1000_tau_pairs500"
    assert sample.scenarios() == ["single_photon", "tau_pairs"]
    assert sample.label_for("tau_pairs") == "Mine (n=500)"
    assert sample.events_in("tau_pairs") == 1480 and sample.per_input("tau_pairs") == 500
    assert sample.menu_label() == "Mine (low-cost)"
    assert sample.sample_text() == "n=1000 (500 of tau_pairs)"
    sample.sample = None
    assert sample.label_for("tau_pairs") == "Mine" == sample.menu_label()
    assert json.dumps(plot.parse_samples(["a:1000", "b:_n5_tau_pairs2"])) == \
        '{"a": "_n1000", "b": "_n5_tau_pairs2"}'
    for wrong in (["a:x"], ["a:0"], [":5"], ["a:5", "a:6"]):
        with pytest.raises(ValueError):
            plot.parse_samples(wrong)
