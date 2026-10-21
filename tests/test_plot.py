"""The figures: what each kind of profile is drawn as, and how it is compared;
and the energy linearity and resolution, with their fits."""

import os
import types

import numpy as np
import pytest

from ddml_bench import measure, plot


@pytest.fixture
def figures(monkeypatch):
    """The figures plot_profile draws, kept open to look at."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    close = plt.close
    kept = []
    monkeypatch.setattr(plt, "close", kept.append)
    yield kept
    for figure in kept:
        close(figure)


def draw(observable, edges, means, out):
    """Draw Geant4 (the first mean energies per bin) and a model (the second),
    each with a 5% uncertainty; the metrics."""
    geant4 = types.SimpleNamespace(name="geant4", label="Geant4", color="#8c8c8c",
                                   linestyle="-")
    model = types.SimpleNamespace(name="model", label="Model", color="#e66100",
                                  linestyle="--")
    curves = [(simulator, edges, mean, 0.05 * mean, 100)
              for simulator, mean in zip((geant4, model), means)]
    return plot.plot_profile(observable, "single_photon", "p", curves, geant4,
                             "jensen_shannon", str(out), ["png"])


def drawn_steps(axes):
    """The values of the histograms on the axes, in the order they were drawn."""
    from matplotlib.patches import StepPatch
    return [patch.get_data().values for patch in axes.patches if isinstance(patch, StepPatch)]


def test_a_radial_profile_is_drawn_per_unit_area(figures, tmp_path):
    edges = measure.bin_edges((0.0, 30.0, 30))
    centres = (edges[1:] + edges[:-1]) / 2
    areas = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    # The mean energy in each ring, of showers whose energy density falls off
    # from the axis, the model's more slowly
    geant4 = 0.05 * np.exp(-centres / 5) * areas
    model = 0.05 * np.exp(-centres / 6) * areas
    (_, short, value), = draw("radial_profile", edges, (geant4, model), tmp_path / "radial")

    top = figures[0].axes[0]
    # Geant4's filled histogram and its outline, then the model's line
    steps = drawn_steps(top)
    assert len(steps) == 3
    for values, mean in zip(steps, (geant4, geant4, model)):
        np.testing.assert_allclose(values, mean / areas)
    # and their uncertainty bands with them
    for band, mean in zip(top.collections, (geant4, model)):
        y = band.get_paths()[0].vertices[:, 1]
        assert y.min() == pytest.approx((0.95 * mean / areas).min())
        assert y.max() == pytest.approx((1.05 * mean / areas).max())
    assert top.get_ylabel() == "Mean energy density per event [GeV/mm$^2$]"
    # The metric still compares the fractions of the energy in the rings (which
    # the fractions of the densities would weigh differently)
    assert short == "JS"
    assert value == pytest.approx(plot.jensen_shannon(geant4, model))
    assert value != pytest.approx(plot.jensen_shannon(geant4 / areas, model / areas))


def test_a_longitudinal_profile_is_drawn_per_layer(figures, tmp_path):
    edges = measure.bin_edges((-0.5, 30.5, 31))
    layers = np.arange(31)
    geant4 = 2.0 * np.exp(-((layers - 8) / 6.0) ** 2) + 0.01
    model = 2.1 * np.exp(-((layers - 8.5) / 6.0) ** 2) + 0.01
    (_, _, value), = draw("longitudinal_profile", edges, (geant4, model),
                          tmp_path / "longitudinal")

    top = figures[0].axes[0]
    for values, mean in zip(drawn_steps(top), (geant4, geant4, model)):
        np.testing.assert_allclose(values, mean)
    assert top.get_ylabel() == "Mean energy per event [GeV]"
    assert value == pytest.approx(plot.jensen_shannon(geant4, model))


def test_energy_spread_evenly_over_the_area_gives_a_flat_radial_profile():
    """What dividing by the rings' areas is for: the outer rings get more of
    the energy only for being bigger."""
    n_hits = 90000
    # Hits at evenly spaced fractions of the area of a disc of 30 mm, 1 MeV each
    radius = 30 * np.sqrt((np.arange(n_hits) + 0.5) / n_hits)
    edges = measure.bin_edges((0.0, 30.0, 30))
    energy, _ = measure.accumulate(edges, radius, np.full(n_hits, 0.001),
                                   np.zeros(n_hits, dtype=np.int64), 1, np.array([True]))
    # Per ring, it grows with the radius
    assert energy[-1] == pytest.approx(59 * energy[0], rel=0.02)
    # Per unit area, it is the same everywhere
    density = energy / plot.bin_scale("radial_profile", edges)
    np.testing.assert_allclose(density, n_hits * 0.001 / (np.pi * 30 ** 2), rtol=0.02)
    assert plot.ring_areas(edges).sum() == pytest.approx(np.pi * 30 ** 2)


# ---------------------------------------------------------------------------
# The energy linearity and resolution, from the reconstructed energy of each event


def simulators():
    return (types.SimpleNamespace(name="geant4", label="Geant4", color="#8c8c8c", linestyle="-"),
            types.SimpleNamespace(name="model", label="Model", color="#e66100", linestyle="--"))


def energy_sample(scale, stochastic, seed, energies=(10, 20, 50, 100), n=2000):
    """True and reconstructed energies of n events at each energy: Gaussian,
    with the mean scale E and the width stochastic sqrt(E)."""
    rng = np.random.default_rng(seed)
    true = np.repeat(np.asarray(energies, dtype=float), n)
    return true, rng.normal(scale * true, stochastic * np.sqrt(true))


def test_mu90_and_sigma90_are_of_the_central_90_percent():
    """The mean and standard deviation of the values from int(n * 0.05) to
    int(n * 0.95) (not included) once sorted."""
    rng = np.random.default_rng(1)
    for n in (3, 7, 20, 101, 3000):
        data = rng.normal(50, 3, n)
        core = np.sort(data)[int(n * 0.05):int(n * 0.95)]
        np.testing.assert_array_equal(plot.core90(data), core)
        assert plot.mu90_sigma90(data) == (np.mean(core), np.std(core))
    # Of 20 values, the lowest and the highest are left out
    assert list(plot.core90(np.arange(20.0)[::-1])) == list(range(1, 19))


def test_a_point_per_true_energy():
    rng = np.random.default_rng(2)
    # Two inputs at 20 GeV, and true energies as float32 momenta give them
    energies = np.repeat([80.0, 20.0, 50.0, 20.0], 300)
    true = energies * (1 + rng.normal(0, 1e-7, energies.size))
    reconstructed = rng.normal(0.98 * energies, 0.17 * np.sqrt(energies))
    points, skipped = plot.energy_points(true, reconstructed)
    assert skipped == []
    assert [round(point["energy"]) for point in points] == [20, 50, 80]
    assert [point["n"] for point in points] == [600, 300, 300]
    for point, energy in zip(points, (20, 50, 80)):
        values = reconstructed[energies == energy]
        mu90, sigma90 = plot.mu90_sigma90(values)
        assert (point["mu90"], point["sigma90"]) == (mu90, sigma90)
        assert point["resolution"] == pytest.approx(sigma90 / mu90)
        # The uncertainties, from resampling the events, are about those of a
        # mean and a standard deviation of the central 90%
        assert 0.8 < point["mu90_err"] / (sigma90 / np.sqrt(len(values))) < 2
        assert 0.8 < point["sigma90_err"] / (sigma90 / np.sqrt(2 * len(values))) < 3
    # and the same every time
    assert plot.energy_points(true, reconstructed)[0] == points


def test_events_not_at_fixed_energies_give_no_points():
    true = np.random.default_rng(3).uniform(1, 126, 3000)
    points, skipped = plot.energy_points(true, 0.98 * true)
    assert points == []
    assert any("fixed energies" in why for why in skipped), skipped
    # and an energy with too few events for sigma90 is left out, and said so
    points, skipped = plot.energy_points([20, 20, 80, 80, 80], [19, 20, 78, 79, 80])
    assert [round(point["energy"]) for point in points] == [80]
    assert skipped == ["1 energy with fewer than 3 events (20 GeV) is left out, as sigma90 "
                       "needs more"]


def test_the_fits_find_the_parameters_of_points_on_their_functions():
    energy = np.arange(10, 101, 10.0)
    line = 0.98 * energy - 0.2
    params, _, chi2, ndf = plot.least_squares(plot.fit_terms("linearity", energy), line,
                                              0.01 * line)
    np.testing.assert_allclose(params, [0.98, -0.2], atol=1e-10)
    assert chi2 == pytest.approx(0, abs=1e-12) and ndf == 8
    resolution = 0.01 + 0.17 / np.sqrt(energy) + 0.05 / energy
    params, _, _, ndf = plot.least_squares(plot.fit_terms("resolution", energy), resolution,
                                           0.02 * resolution)
    np.testing.assert_allclose(params, [0.01, 0.17, 0.05], atol=1e-10)
    assert ndf == 7
    # Points that can't fix every parameter give no fit
    assert plot.least_squares(plot.fit_terms("resolution", [20.0, 20.0, 80.0]),
                              [0.04, 0.04, 0.02], [0.001] * 3) is None


def test_a_straight_line_fit_has_the_textbook_uncertainties():
    """With the same uncertainty s on every point: var(slope) = s^2/Sxx,
    var(intercept) = s^2 sum(x^2)/(n Sxx), chi^2 = sum(residual^2)/s^2."""
    x = np.array([10.0, 20, 40, 80])
    y = np.array([9.9, 19.4, 39.5, 78.0])
    s = 0.3
    params, errors, chi2, ndf = plot.least_squares(plot.fit_terms("linearity", x), y,
                                                   np.full(4, s))
    sxx = np.sum((x - x.mean()) ** 2)
    slope = np.sum((x - x.mean()) * (y - y.mean())) / sxx
    np.testing.assert_allclose(params, [slope, y.mean() - slope * x.mean()])
    np.testing.assert_allclose(errors, [s / np.sqrt(sxx), s * np.sqrt(np.sum(x ** 2) / (4 * sxx))])
    assert chi2 == pytest.approx(np.sum((y - params[0] * x - params[1]) ** 2) / s ** 2)
    assert ndf == 2


def test_a_fitted_function_is_written_with_its_signs():
    # to 4 significant figures, as a slope near 1 needs them
    assert plot.fit_text(("%s $E$", "%s GeV"), [0.999518, -0.12345]) \
        == "0.9995 $E$ \u2212 0.1235 GeV"
    assert plot.fit_text(("%s", r"%s/$\sqrt{E}$", "%s/$E$"), [-0.002, 0.165, 0.02]) \
        == "\u22120.002 + 0.165/$\\sqrt{E}$ + 0.02/$E$"


def test_the_energy_figures_draw_points_fits_and_deviations(figures, tmp_path):
    geant4, model = simulators()
    measured = [(geant4,) + energy_sample(0.98, 0.17, 1), (model,) + energy_sample(0.97, 0.19, 2)]
    results, fits = plot.plot_energy("energy", "single_photon", measured, geant4, "l1",
                                     str(tmp_path), "", ["png"])
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "energy_linearity__single_photon.png", "energy_resolution__single_photon.png"]
    points = [plot.energy_points(true, reconstructed)[0] for _, true, reconstructed in measured]

    for figure, (name, value, _, parameters, terms) in zip(figures, plot.ENERGY_FIGURES):
        top, bottom = figure.axes
        assert top.get_xlim()[0] == 0 and bottom.get_xlabel() == "True energy $E$ [GeV]"
        # Each simulator's points, with their uncertainties
        for container, simulator_points in zip(top.containers, points):
            x, y = container.lines[0].get_data()
            np.testing.assert_allclose(x, [p["energy"] for p in simulator_points])
            np.testing.assert_allclose(y, [p[value] for p in simulator_points])
        # and the function fitted to them, written in the legend
        drawn = [line for line in top.lines if len(line.get_xdata()) == 200]
        labels = [text.get_text() for text in top.get_legend().get_texts()]
        for simulator, simulator_points, line, label in zip((geant4, model), points, drawn,
                                                           labels):
            x = np.array([p["energy"] for p in simulator_points])
            params, errors, chi2, ndf = plot.least_squares(
                plot.fit_terms(name, x), [p[value] for p in simulator_points],
                [p[value + "_err"] for p in simulator_points])
            np.testing.assert_allclose(line.get_ydata(),
                                       plot.fit_terms(name, line.get_xdata()) @ params)
            assert label.startswith(simulator.label)
            assert label.endswith(plot.fit_text(terms, params))
            assert [fit[2:] for fit in fits if fit[0] is simulator and fit[1] == name] == [
                (parameter, p, e, chi2, ndf) for parameter, p, e in zip(parameters, params, errors)]
        # The model's distance from Geant4, and its relative deviation below
        reference, mine = ([p[value] for p in simulator_points] for simulator_points in points)
        (metric,) = [result for result in results if result[1] == name]
        assert metric[0] is model and metric[2] == "L1"
        assert metric[3] == pytest.approx(np.mean(np.abs(np.subtract(mine, reference))))
        assert "(L1 %.2e)" % metric[3] in labels[1]
        np.testing.assert_allclose(bottom.containers[0].lines[0].get_ydata(),
                                   (np.divide(mine, reference) - 1) * 100)
    # A wider resolution for the model
    assert all(m["resolution"] > g["resolution"] for g, m in zip(*points))


def test_two_energies_fit_only_the_linearity(figures, tmp_path):
    geant4, _ = simulators()
    measured = [(geant4,) + energy_sample(0.98, 0.17, 1, energies=(20, 80), n=6)]
    results, fits = plot.plot_energy("energy", "single_photon", measured, geant4, "l1",
                                     str(tmp_path), "_n6", ["png"])
    assert results == []
    assert [(fit[1], fit[2], fit[6]) for fit in fits] == [("linearity", "slope", 0),
                                                         ("linearity", "intercept", 0)]
    linearity, resolution = figures
    # Geant4 alone has no deviations to show, and an unfitted figure only its points
    assert len(resolution.axes) == 1
    assert [text.get_text() for text in resolution.axes[0].get_legend().get_texts()] \
        == ["Geant4"]
    assert os.path.isfile(str(tmp_path / "energy_resolution__single_photon_n6.png"))
