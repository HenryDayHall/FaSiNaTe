"""The figures: what each kind of profile is drawn as, and how it is compared;
and the energy linearity and resolution, with their fits."""

import json
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
    (_, short, value), (_, w_short, w_value) = draw("radial_profile", edges, (geant4, model),
                                                    tmp_path / "radial")

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
    # and so does the Wasserstein distance, in mm: the model's energy is
    # farther out
    assert w_short == "W1"
    assert w_value == pytest.approx(plot.wasserstein(geant4, model, edges))
    assert w_value == pytest.approx(np.sum(centres * model) / model.sum()
                                    - np.sum(centres * geant4) / geant4.sum(), rel=0.02)


def test_a_models_deviation_carries_its_own_and_geant4s_uncertainty():
    dev, band = plot.relative_deviation([1.1, 2.0, 2.0, 1.0, 3.0], [0.11, 0.1, 0.1, 0.2, 0.1],
                                        [1.0, 2.5, 2.0, 0.0, -1.0], [0.05, 0.25, 0.2, 0.1, 0.1])
    np.testing.assert_allclose(dev[:3], [10.0, -20.0, 0.0])
    # As for a ratio of independent values: its relative uncertainty is the
    # quadrature sum of theirs
    np.testing.assert_allclose(band[:3], [110 * np.hypot(0.11 / 1.1, 0.05 / 1.0),
                                          80 * np.hypot(0.1 / 2.0, 0.25 / 2.5),
                                          np.hypot(5.0, 10.0)])
    # (where the two agree, the quadrature sum of the two bands drawn before:
    # the model's 5% and Geant4's 10% of Geant4's value); none without a
    # reference above 0
    assert np.isnan(dev[3:]).all() and np.isnan(band[3:]).all()


def test_each_models_deviation_has_one_band_of_both_uncertainties(figures, tmp_path):
    from matplotlib.colors import to_hex
    edges = measure.bin_edges((-0.5, 4.5, 5))
    geant4 = np.array([1.0, 2.0, 4.0, 2.0, 1.0])
    model = np.array([1.1, 1.8, 4.0, 2.2, 1.0])
    draw("longitudinal_profile", edges, (geant4, model), tmp_path / "layers")
    top, bottom = figures[0].axes
    # Geant4's band stays above, but below there is only the model's, in its colour
    assert len(top.collections) == 2
    (band,) = bottom.collections
    assert to_hex(band.get_facecolor()[0], keep_alpha=False) == "#e66100"
    dev, error = plot.relative_deviation(model, 0.05 * model, geant4, 0.05 * geant4)
    # with 5% on each, sqrt(2) times 5% of their ratio
    np.testing.assert_allclose(error, np.sqrt(2) * 5 * model / geant4)
    y = band.get_paths()[0].vertices[:, 1]
    assert y.min() == pytest.approx((dev - error).min())
    assert y.max() == pytest.approx((dev + error).max())


def test_a_longitudinal_profile_is_drawn_per_layer(figures, tmp_path):
    edges = measure.bin_edges((-0.5, 30.5, 31))
    layers = np.arange(31)
    geant4 = 2.0 * np.exp(-((layers - 8) / 6.0) ** 2) + 0.01
    model = 2.1 * np.exp(-((layers - 8.5) / 6.0) ** 2) + 0.01
    (_, _, value), (_, _, w_value) = draw("longitudinal_profile", edges, (geant4, model),
                                          tmp_path / "longitudinal")

    top = figures[0].axes[0]
    for values, mean in zip(drawn_steps(top), (geant4, geant4, model)):
        np.testing.assert_allclose(values, mean)
    assert top.get_ylabel() == "Mean energy per event [GeV]"
    assert value == pytest.approx(plot.jensen_shannon(geant4, model))
    assert w_value == pytest.approx(plot.wasserstein(geant4, model, edges))


def test_the_chi_squared_weighs_each_difference_by_both_uncertainties():
    # (1 - 2)^2 / (0.3^2 + 0.4^2) + (5 - 5)^2 / ... = 4
    assert plot.chi_squared([1.0, 5.0], [0.3, 0.1], [2.0, 5.0], [0.4, 0.2]) == (
        pytest.approx(4.0), 2)
    # Points without an uncertainty, or a value, aren't counted
    assert plot.chi_squared([1.0, 3.0, np.nan], [0.0, 0.5, 0.1], [2.0, 2.0, 1.0],
                            [0.0, 0.0, 0.1]) == (pytest.approx(4.0), 1)
    found = plot.chi_squared([1.0], [0.0], [2.0], [0.0])
    assert np.isnan(found[0]) and found[1] == 0
    # As the legends give it: with the points (nothing is fitted), in full above 100
    assert plot.chi_squared_text(25.3456, 10) == "$\\chi^2$/ndf 25.3/10"
    assert plot.chi_squared_text(9304.41, 10) == "$\\chi^2$/ndf 9304/10"


def test_the_wasserstein_distance_is_how_far_the_distribution_moves():
    edges = np.linspace(0, 20, 21)
    shape = np.exp(-((np.arange(20) - 6.0) / 2) ** 2)
    # The same shape, 3 bins on: 3 units, however much of it there is
    moved = np.roll(shape, 3)
    assert plot.wasserstein(shape, moved, edges) == pytest.approx(3, rel=1e-3)
    assert plot.wasserstein(10 * shape, moved, edges) == pytest.approx(3, rel=1e-3)
    assert plot.wasserstein(moved, shape, edges) == plot.wasserstein(shape, moved, edges)
    assert plot.wasserstein(shape, shape, edges) == 0
    # In the units of the bins
    assert plot.wasserstein(shape, moved, 0.5 * edges) == pytest.approx(1.5, rel=1e-3)
    # Half of it moved 4 bins, in bins of any widths: each bin's content at its centre
    uneven = np.array([0.0, 1.0, 1.5, 3.0, 5.0, 8.0])
    assert plot.wasserstein([1, 1, 0, 0, 0], [1, 0, 0, 0, 1], uneven) == pytest.approx(
        0.5 * ((8.0 + 5.0) / 2 - (1.0 + 1.5) / 2))
    assert np.isnan(plot.wasserstein([0, 0], [1, 1], [0, 1, 2]))
    # as scipy has it, of weighted values at the centres
    stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(3)
    p, q = rng.random(20), rng.random(20)
    centres = (edges[1:] + edges[:-1]) / 2
    assert plot.wasserstein(p, q, edges) == pytest.approx(
        stats.wasserstein_distance(centres, centres, p, q))


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
        (metric, chi2) = [result for result in results if result[1] == name]
        assert metric[0] is model and metric[2] == "L1"
        assert metric[3] == pytest.approx(np.mean(np.abs(np.subtract(mine, reference))))
        # and its chi^2, with the uncertainties of both, both in the legend
        assert chi2[0] is model and chi2[2] == "chi2"
        errors = [np.hypot(m[value + "_err"], g[value + "_err"]) for g, m in zip(*points)]
        assert chi2[3] == pytest.approx(np.sum((np.subtract(mine, reference) / errors) ** 2))
        assert "(L1 %.2e, %s)" % (metric[3], plot.chi_squared_text(chi2[3], len(mine))) in \
            labels[1]
        np.testing.assert_allclose(bottom.containers[0].lines[0].get_ydata(),
                                   (np.divide(mine, reference) - 1) * 100)
        # its error bars the model's and Geant4's uncertainties combined, with
        # no band of Geant4's of its own
        (bars,) = bottom.collections
        _, band = plot.relative_deviation(
            mine, [p[value + "_err"] for p in points[1]], reference,
            [p[value + "_err"] for p in points[0]])
        np.testing.assert_allclose([s[1, 1] - s[0, 1] for s in bars.get_segments()], 2 * band)
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


# ---------------------------------------------------------------------------
# The intrinsic angles: the distributions of the residuals of the showers' axes


def angle_simulator(name, label, color, linestyle, stored):
    """A simulator that measured the intrinsic angle: {what: [event]}."""
    return types.SimpleNamespace(name=name, label=label, color=color, linestyle=linestyle,
                                 stored=lambda observable, scenario: stored)


def residual_sample(width, shift, seed, n=4000):
    """Residuals of n showers for each angle and PCA, Gaussian, the top 4%'s
    half as wide; a few showers without an axis from the top 4%, and a few far
    out."""
    rng = np.random.default_rng(seed)
    stored = {"theta_true": rng.uniform(43, 137, n), "phi_true": rng.uniform(79, 109, n)}
    for hits, scale in (("all", 1.0), ("top4", 0.5)):
        for angle in ("theta", "phi"):
            stored["d%s_%s" % (angle, hits)] = rng.normal(shift, width * scale, n)
    stored["dtheta_top4"][:7] = np.nan
    stored["dtheta_all"][:3] = 40.0
    return stored


def test_the_intrinsic_angles_are_drawn_as_fractions_of_the_showers_per_degree(figures,
                                                                               tmp_path):
    geant4 = angle_simulator("geant4", "Geant4", "#8c8c8c", "-", residual_sample(1.0, 0, 1))
    model = angle_simulator("model", "Model", "#e66100", "--", residual_sample(1.2, 0.2, 2))
    settings = {"hit_fractions": [1.0, 0.04], "max_residual": [5, 2], "n_bins": [50, 40]}
    results, fits = plot.angle_figures("intrinsic_angle", "single_photon", [geant4, model],
                                       geant4, settings, str(tmp_path), "", ["png"])
    names = ["theta_all", "phi_all", "theta_top4", "phi_top4"]
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        "intrinsic_angle_%s__single_photon.png" % name for name in names)
    # Of each figure, its metric, then its Wasserstein distance
    assert fits == [] and [result[1:3] for result in results] == [
        (name, metric) for name in names for metric in ("JS", "W1")]

    for figure, name, (simulator, profile, short, value), (_, _, _, w_value) in zip(
            figures, names, results[::2], results[1::2]):
        angle, hits = name.split("_")
        limit, n_bins = (5, 50) if hits == "all" else (2, 40)
        edges = np.linspace(-limit, limit, n_bins + 1)
        top, bottom = figure.axes
        assert top.get_xlim() == (-limit, limit)
        assert bottom.get_xlabel().startswith("Intrinsic angle $\\%s_\\mathrm{reco}" % angle)
        # Each simulator's fractions of its showers with an axis in each bin,
        # per degree: Geant4's filled, and the model's line
        fractions = []
        for stored in (geant4.stored(None, None), model.stored(None, None)):
            residuals = stored["d" + name]
            residuals = residuals[np.isfinite(residuals)]
            fractions.append(np.histogram(residuals, edges)[0] / residuals.size)
        steps = drawn_steps(top)
        for values, expected in zip(steps, (fractions[0], fractions[0], fractions[1])):
            np.testing.assert_allclose(values, expected / (2 * limit / n_bins))
        # compared by the Jensen-Shannon divergence of the fractions, and by
        # their Wasserstein distance, in degrees
        assert (simulator, profile, short) == (model, name, "JS")
        assert value == pytest.approx(plot.jensen_shannon(*fractions))
        assert w_value == pytest.approx(plot.wasserstein(*fractions, edges))
        # and the text in the corner says which PCA, of how many showers
        (note,) = [text.get_text() for text in top.texts]
        n = int(np.isfinite(geant4.stored(None, None)["d" + name]).sum())
        assert note.startswith("single_photon, PCA of %s\n%d showers (Geant4)" % (
            "all hits" if hits == "all" else "the top 4% of hits", n))
        assert n == (3993 if name == "theta_top4" else 4000)
        # with the share of them beyond the bins
        if name == "theta_all":
            assert note.endswith("\n0.075% beyond \u00b15\u00b0")
        # and room above the curves for it and the legend
        assert top.get_ylim()[1] > 1.3 * max(np.max(s) for s in steps)


def test_without_settings_the_residuals_are_drawn_in_the_default_bins(figures, tmp_path):
    model = angle_simulator("model", "Model", "#e66100", "--", residual_sample(1.2, 0.2, 2))
    results, _ = plot.angle_figures("intrinsic_angle", "single_photon", [model], None, {},
                                    str(tmp_path), "_n4000", ["png"])
    # Nothing to compare to, so no metric and no deviations
    assert results == [] and len(figures) == 4 and all(len(f.axes) == 1 for f in figures)
    assert figures[0].axes[0].get_xlim() == (-measure.MAX_RESIDUAL, measure.MAX_RESIDUAL)
    assert len(drawn_steps(figures[0].axes[0])[0]) == measure.RESIDUAL_BINS
    assert os.path.isfile(str(tmp_path / "intrinsic_angle_phi_top4__single_photon_n4000.png"))


def measured_in_box(simulator, side):
    """The simulator, with the settings of its intrinsic angles as its .npz file
    records them: hits in a bounding box with that side (mm), or every hit."""
    settings = {"hit_collections": ["EcalBarrelCollectionRec"], "hit_fractions": [1.0, 0.04],
                "scenarios": ["single_photon"]}
    if side is not None:
        settings["bounding_box"] = side
    simulator.data = {"intrinsic_angle__single_photon__settings": np.array(
        json.dumps(settings, sort_keys=True))}
    return simulator


def test_the_angle_figures_say_which_bounding_box_the_hits_were_in(figures, tmp_path, capsys):
    geant4 = measured_in_box(angle_simulator("geant4", "Geant4", "#8c8c8c", "-",
                                             residual_sample(1.0, 0, 1)), 800)
    model = measured_in_box(angle_simulator("model", "Model", "#e66100", "--",
                                            residual_sample(1.2, 0.2, 2)), 800)
    plot.angle_figures("intrinsic_angle", "single_photon", [geant4, model], geant4, {},
                       str(tmp_path), "", ["png"])
    assert len(figures) == 4
    for figure in figures:
        (note,) = [text.get_text() for text in figure.axes[0].texts]
        lines = note.split("\n")
        assert lines[0].startswith("single_photon, PCA of ") and lines[1] == \
            "in a box of side 800 mm" and lines[2].endswith(" showers (Geant4)"), note
    said = capsys.readouterr().err
    assert "(fewer than two hits in the bounding box), so they're left out" in said
    assert "same bounding box" not in said
    # Measured in another box (or in none), a model's showers aren't cut as
    # Geant4's are: that's said, and the note is Geant4's
    del figures[:]
    measured_in_box(model, None)
    plot.angle_figures("intrinsic_angle", "single_photon", [geant4, model], geant4, {},
                       str(tmp_path), "", ["png"])
    assert "single_photon: the intrinsic angles weren't all measured in the same bounding box " \
        "(geant4: 800 mm, model: none)" in capsys.readouterr().err
    assert all(f.axes[0].texts[0].get_text().split("\n")[1] == "in a box of side 800 mm"
               for f in figures)
    # and without a box, the note doesn't mention one
    del figures[:]
    plot.angle_figures("intrinsic_angle", "single_photon", [model], None, {}, str(tmp_path),
                       "", ["png"])
    assert all("box" not in f.axes[0].texts[0].get_text() for f in figures) and figures


# ---------------------------------------------------------------------------
# The di-photon separation: the mean number of reconstructed photons in bins
# of the separation, with the spread over the seeds


def di_photon_sample(told_apart, seed, energies=(20.0,), seeds=(1, 2, 3), n=3000):
    """Of each event: its separation, uniform from 0 to 90 mm, its number of
    reconstructed photons (two once the photons are further apart than
    told_apart mm, one before, sometimes a third), its photons' energy, and its
    seed; n events of each energy for each seed."""
    rng = np.random.default_rng(seed)
    columns = []
    for energy in energies:
        for simulation_seed in seeds:
            separation = rng.uniform(0, 90, n)
            two = rng.random(n) < 1 / (1 + np.exp(-(separation - told_apart) / 3))
            photons = 1 + two + (rng.random(n) < 0.05)
            columns.append((separation, photons, np.full(n, energy), np.full(n, simulation_seed)))
    return [np.concatenate(parts) for parts in zip(*columns)]


def test_the_mean_number_of_photons_and_its_spread_over_the_seeds():
    edges = np.array([0.0, 10.0, 20.0, 30.0])
    # Seed 1: 1, 2 photons in the first bin, 2 in the second; seed 2: 1 and 2;
    # a third bin without events, one beyond the bins, and one without a separation
    separation = np.array([1, 2, 15, 3, 15, 50, np.nan])
    photons = np.array([1, 2, 2, 1, 2, 2, 2])
    seeds = np.array([1, 1, 1, 2, 2, 2, 2])
    points = plot.separation_points(separation, photons, seeds, edges)
    assert points["seeds"] == [1, 2] and points["spread"]
    np.testing.assert_allclose(points["mean"][:2], [(1.5 + 1.0) / 2, 2.0])
    np.testing.assert_allclose(points["error"][:2], [np.std([1.5, 1.0], ddof=1), 0.0])
    assert np.isnan(points["mean"][2]) and np.isnan(points["error"][2])
    np.testing.assert_allclose(points["photons"][1], [3, 2, 0])
    np.testing.assert_allclose(points["photons"][2], [1, 2, 0])
    assert points["events"] == {1: 3, 2: 3} and points["outside"] == pytest.approx(1 / 6)

    # With one seed, the uncertainty is the statistical one of the mean
    points = plot.separation_points(separation[:3], photons[:3], seeds[:3], edges)
    assert not points["spread"]
    np.testing.assert_allclose(points["error"][:2], [np.std([1, 2]) / np.sqrt(2), 0.0])


def test_the_metric_compares_the_photons_in_the_bins_seed_by_seed():
    edges = np.linspace(0, 90, 19)
    reference = plot.separation_points(*di_photon_sample(20, 1)[:2],
                                       di_photon_sample(20, 1)[3], edges)
    separation, photons, _, seeds = di_photon_sample(24, 2)
    points = plot.separation_points(separation, photons, seeds, edges)
    value, spread, pairs = plot.seed_metric(plot.jensen_shannon, reference, points)
    values = [plot.jensen_shannon(reference["photons"][s], points["photons"][s]) for s in (1, 2, 3)]
    assert pairs == 3
    assert value == pytest.approx(np.mean(values)) and spread == pytest.approx(np.std(values, ddof=1))
    # (the photons in each bin, not their mean per event)
    assert values[0] == pytest.approx(plot.jensen_shannon(
        np.histogram(separation[seeds == 1], edges, weights=photons[seeds == 1])[0],
        reference["photons"][1]))
    # A model of other seeds is paired with Geant4's in order
    moved = plot.separation_points(separation, photons, seeds + 10, edges)
    assert plot.seed_metric(plot.jensen_shannon, reference, moved) == (value, spread, 3)
    # and with one pair the spread is unknown
    single = plot.separation_points(separation[seeds == 1], photons[seeds == 1],
                                    seeds[seeds == 1], edges)
    value, spread, pairs = plot.seed_metric(plot.jensen_shannon, reference, single)
    assert pairs == 1 and value == pytest.approx(values[0]) and np.isnan(spread)


def test_a_separation_figure_for_each_energy(figures, tmp_path):
    geant4, model = simulators()
    measured = [(geant4,) + tuple(di_photon_sample(20, 1, energies=(20.0, 100.0))),
                (model,) + tuple(di_photon_sample(24, 2, energies=(20.0, 100.0)))]
    # A third simulator measured only at 20 GeV
    other = types.SimpleNamespace(name="other", label="Other", color="#2a9a3a", linestyle=":")
    measured.append((other,) + tuple(di_photon_sample(21, 3)))
    settings = {"pdg": [22], "max_separation": 90, "n_bins": 18}
    results, fits = plot.plot_separation("photon_separation", "di_photon", measured, geant4,
                                         settings, str(tmp_path), "", ["png"])
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "photon_separation_100GeV__di_photon.png", "photon_separation_20GeV__di_photon.png"]
    assert fits == []
    # Of each energy, the models' metrics, then their Wasserstein distances
    assert [(r[0].name, r[1], r[2]) for r in results] == [
        ("model", "20GeV", "JS"), ("other", "20GeV", "JS"), ("model", "20GeV", "W1"),
        ("other", "20GeV", "W1"), ("model", "100GeV", "JS"), ("model", "100GeV", "W1")]

    edges = np.linspace(0, 90, 19)
    for figure, energy in zip(figures, (20.0, 100.0)):
        top, bottom = figure.axes
        assert top.get_ylabel() == "Mean reconstructed photons per event"
        assert bottom.get_xlabel() == plot.SEPARATION_XLABEL
        # Geant4's filled histogram and its outline, then each model's line:
        # the mean over the seeds of each seed's mean number of photons
        expected = []
        for simulator, separation, photons, energies, seeds in measured:
            chosen = energies == energy
            if chosen.any():
                expected.append(plot.separation_points(separation[chosen], photons[chosen],
                                                       seeds[chosen], edges))
        steps = drawn_steps(top)
        assert len(steps) == len(expected) + 1
        for values, points in zip(steps, [expected[0]] + expected):
            np.testing.assert_allclose(values, points["mean"])
        # and two photons, the target, as a line across
        assert [line.get_ydata()[0] for line in top.lines] == [2]
        labels = [text.get_text() for text in top.get_legend().get_texts()]
        value, spread, _ = plot.seed_metric(plot.jensen_shannon, expected[0], expected[1])
        assert labels[0] == "Geant4" and labels[1] == "Model (JS %.2e ± %.1e)" % (value,
                                                                                    spread)
        (result,) = [r for r in results if r[0] is model and r[1] == plot.energy_name(energy)
                     and r[2] == "JS"]
        assert result[3:] == (value, spread)
        # and the Wasserstein distance of the photons in the bins, seed by
        # seed as the metric, in mm
        (distance,) = [r for r in results if r[0] is model
                       and r[1] == plot.energy_name(energy) and r[2] == "W1"]
        assert distance[3:] == plot.seed_metric(
            lambda p, q: plot.wasserstein(p, q, edges), expected[0], expected[1])[:2]
        assert 0 < distance[3] < 90
        (note,) = [text.get_text() for text in top.texts]
        assert note == ("di_photon, %g GeV photons\n3000 events × 3 seeds (Geant4)\n"
                        "bands: the spread over the seeds" % energy)


def test_the_seeds_of_the_events_come_from_their_files():
    stored = {"s__n_events": np.array([2, 3]), "s__seeds": np.array([4, -1])}
    simulator = types.SimpleNamespace(data=stored)
    assert list(plot.event_seeds(simulator, "s", 5)) == [4, 4, -1, -1, -1]
    # Measurements without the seeds, or of other events, have none
    assert list(plot.event_seeds(types.SimpleNamespace(data={"s__n_events": np.array([5])}), "s",
                                 5)) == [-1] * 5
    assert list(plot.event_seeds(simulator, "s", 4)) == [-1] * 4
    assert [plot.energy_name(e) for e in (20.0000004, 99.999, 12.5, 5)] == [
        "20GeV", "100GeV", "12.5GeV", "5GeV"]


# ---------------------------------------------------------------------------
# The pi0s of tau decays: of the good ones, the mass of their two photons and
# their energy less the MC pi0s', over the seeds (the paper's figure 11)


def test_the_distribution_over_the_seeds_and_its_spread():
    edges = np.linspace(0.0, 4.0, 5)
    values = np.array([0.5, 1.5, 1.5, 3.5, 9.0, np.nan,  # seed 1: 5 values, one outside
                       0.5, 0.5, 2.5, 2.5,  # seed 2
                       1.5, 1.5, 1.5, 3.5])  # seed 3
    seeds = np.array([1] * 6 + [2] * 4 + [3] * 4)
    points = plot.seed_histograms(values, seeds, edges)
    fractions = np.array([[1, 2, 0, 1], [2, 0, 2, 0], [0, 3, 0, 1]]) / np.array([[5], [4], [4]])
    np.testing.assert_allclose(points["mean"], fractions.mean(axis=0))
    np.testing.assert_allclose(points["error"], fractions.std(axis=0, ddof=1))
    assert points["seeds"] == [1, 2, 3] and points["spread"]
    assert points["values"] == {1: 5, 2: 4, 3: 4}
    np.testing.assert_array_equal(points["counts"][3], [0, 3, 0, 1])
    assert points["outside"] == pytest.approx(1 / 13)
    # Of one seed, the statistical uncertainty, per unit of the values
    single = plot.seed_histograms(values[seeds == 2], seeds[seeds == 2], edges)
    np.testing.assert_allclose(single["error"], np.sqrt([2, 0, 2, 0]) / 4)
    assert not single["spread"]


def pi0_simulator(name, label, color, linestyle, scale, seed, seeds=(1, 2, 3), n_events=1500,
                  angular=0.05):
    """A simulator that measured the pi0s of tau_pairs: of each of n_events
    for each ddsim seed, its candidates (0 to 2), each good, confused or fake,
    with the mass of its photons, its fitted energy, its photons' energy and
    its MC pi0's, the energies scaled; and theta and phi of its direction and
    its MC pi0's, in degrees, off by angular (the standard deviation), phi all
    the way round."""
    rng = np.random.default_rng(seed)
    counts = rng.integers(0, 3, n_events * len(seeds))
    k = int(counts.sum())
    match = rng.choice([measure.GOOD, measure.CONFUSED, measure.FAKE], k, p=[0.8, 0.15, 0.05])
    true = rng.uniform(10, 80, k)
    slot = np.concatenate([rng.permutation(2)[:c] for c in counts]) if k else np.zeros(0, int)
    stored = {"n_candidates": counts, "match": match, "n_true": np.full(len(counts), 2),
              "true_index": np.where(match != measure.FAKE, slot, -1),
              "mass": rng.normal(0.135 * np.sqrt(scale), 0.011, k),
              "energy": rng.normal(true * scale, 0.8),
              "photon_energy": rng.normal(true * scale - 0.5, 1.0),
              "true_energy": np.where(match != measure.FAKE, true, np.nan)}
    true_theta, true_phi = rng.uniform(40, 140, k), rng.uniform(-180, 180, k)
    stored.update(theta=true_theta + rng.normal(0, angular, k),
                  phi=(true_phi + rng.normal(0, angular, k) + 180) % 360 - 180,
                  true_theta=np.where(match != measure.FAKE, true_theta, np.nan),
                  true_phi=np.where(match != measure.FAKE, true_phi, np.nan))
    data = {"tau_pairs__n_events": np.array([n_events] * len(seeds)),
            "tau_pairs__seeds": np.array(seeds)}
    return types.SimpleNamespace(name=name, label=label, color=color, linestyle=linestyle,
                                 data=data, stored=lambda observable, scenario: stored)


def test_the_pi0_figures_draw_the_good_pi0s_over_the_seeds(figures, tmp_path):
    geant4 = pi0_simulator("geant4", "Geant4", "#8c8c8c", "-", 1.0, 1)
    model = pi0_simulator("model", "Model", "#e66100", "--", 0.98, 2, angular=0.06)
    settings = {"mass_range": [0.09, 0.18], "mass_bins": 18, "energy_range": [-4, 4],
                "energy_bins": 32, "reconstructed_energy": "photons", "theta_range": [-0.2, 0.2],
                "theta_bins": 40, "phi_range": [-0.25, 0.25], "phi_bins": 25}
    results, fits = plot.pi0_figures("pi0", "tau_pairs", [geant4, model], geant4, settings,
                                     str(tmp_path), "", ["png"])
    assert sorted(p.name for p in tmp_path.iterdir()) == ["pi0_categories__tau_pairs.png",
                                                          "pi0_energy__tau_pairs.png",
                                                          "pi0_mass__tau_pairs.png",
                                                          "pi0_phi__tau_pairs.png",
                                                          "pi0_theta__tau_pairs.png"]
    # Of each figure, the metric, then the Wasserstein distance, then the
    # categories' deviations, then their chi^2
    assert fits == [] and [result[:3] for result in results[:8]] == [
        (model, name, metric) for name in ("mass", "energy", "theta", "phi")
        for metric in ("JS", "W1")]
    assert [result[1:3] for result in results[8:]] == [
        ("categories_" + key, "dev%") for key, _ in plot.PI0_CATEGORIES] + [
        ("categories", "chi2")]
    distances = results[1:8:2]
    results = results[:8:2]
    # The residuals, reconstructed less the MC pi0's, phi's the shorter way round
    residuals = {"mass": lambda stored: stored["mass"],
                 "energy": lambda stored: stored["photon_energy"] - stored["true_energy"],
                 "theta": lambda stored: stored["theta"] - stored["true_theta"],
                 "phi": lambda stored: (stored["phi"] - stored["true_phi"] + 180) % 360 - 180}
    for figure, (_, name, _, value, spread), distance, (first, last, n_bins), unit in zip(
            figures, results, distances, ((0.09, 0.18, 18), (-4.0, 4.0, 32), (-0.2, 0.2, 40),
                                          (-0.25, 0.25, 25)), ("GeV", "GeV", "degree", "degree")):
        edges = np.linspace(first, last, n_bins + 1)
        width = (last - first) / n_bins
        expected, counts = [], []
        for simulator in (geant4, model):
            stored = simulator.stored(None, None)
            seeds = np.repeat(np.repeat([1, 2, 3], 1500), stored["n_candidates"])
            good = stored["match"] == measure.GOOD
            values = residuals[name](stored)
            per_seed = [np.histogram(values[good & (seeds == s)], edges)[0] for s in (1, 2, 3)]
            sizes = [np.sum(good & (seeds == s)) for s in (1, 2, 3)]
            expected.append(np.mean([c / n / width for c, n in zip(per_seed, sizes)], axis=0))
            counts.append(per_seed)
        top, bottom = figure.axes
        assert top.get_xlim() == (first, last)
        assert top.get_ylabel() == r"Fraction of good $\pi^0$s per " + unit
        steps = drawn_steps(top)
        for values, wanted in zip(steps, (expected[0], expected[0], expected[1])):
            np.testing.assert_allclose(values, wanted)
        # The metric: of the good pi0s in the bins, seed by seed
        each = [plot.jensen_shannon(g, m) for g, m in zip(*counts)]
        assert value == pytest.approx(np.mean(each)) and spread == pytest.approx(
            np.std(each, ddof=1))
        # and the Wasserstein distance likewise, in the units of the figure
        each = [plot.wasserstein(g, m, edges) for g, m in zip(*counts)]
        assert distance[1:3] == (name, "W1")
        assert distance[3] == pytest.approx(np.mean(each))
        assert distance[4] == pytest.approx(np.std(each, ddof=1))
        assert plot.unit_of("pi0", name) == ("°" if unit == "degree" else unit)
        (note,) = [text.get_text() for text in top.texts]
        assert note.startswith("tau_pairs, good $\\pi^0$s\n")
        assert "× 3 seeds (Geant4)" in note and "\nbands: seed spread" in note, note
        assert ("photons energy" in note) == (name == "energy")
        assert "JS %.2e ± %.1e" % (value, spread) in [t.get_text() for t in
                                                           top.get_legend().get_texts()][1]
    # The scale of the model's energies shows in the energy, below Geant4's
    model_energy = drawn_steps(figures[1].axes[0])[2]
    centres = np.linspace(-4, 4, 33)[:-1] + 0.125
    assert np.sum(model_energy * centres) < np.sum(drawn_steps(figures[1].axes[0])[0] * centres)
    # and its wider spread of directions in the angles, lower in the middle
    for figure in figures[2:4]:
        geant4_steps, _, model_steps = drawn_steps(figure.axes[0])
        middle = slice(len(model_steps) // 2 - 2, len(model_steps) // 2 + 2)
        assert model_steps[middle].sum() < 0.95 * geant4_steps[middle].sum()
    assert figures[2].axes[1].get_xlabel().startswith(r"$\theta_{\pi^0}^\mathrm{reco}")
    assert figures[3].axes[1].get_xlabel().startswith(r"$\phi_{\pi^0}^\mathrm{reco}")


def test_the_pi0_angles_are_of_the_direction_less_the_mc_pi0s_phi_the_shorter_way():
    G, F = measure.GOOD, measure.FAKE
    nan = float("nan")
    # Event 0: a good pi0 and a fake; event 1: none; events 2 and 3: a good
    # one each, the first across phi = 180 degrees
    stored = {"n_candidates": np.array([2, 0, 1, 1]), "match": np.array([G, F, G, G]),
              "mass": np.full(4, 0.135), "energy": np.array([20.0, 5.0, 10.0, 30.0]),
              "photon_energy": np.full(4, 1.0), "true_energy": np.array([21.0, nan, 9.0, 30.0]),
              "theta": np.array([80.01, 60.0, 99.98, 45.0]),
              "true_theta": np.array([80.0, nan, 100.0, 45.02]),
              "phi": np.array([179.99, 0.0, -179.995, 90.0]),
              "true_phi": np.array([-179.99, nan, 179.995, 89.97])}
    simulator = types.SimpleNamespace(data={"tau_pairs__n_events": np.array([2, 2]),
                                            "tau_pairs__seeds": np.array([7, 9])},
                                      stored=lambda observable, scenario: stored)
    values, seeds = plot.good_pi0s(simulator, "pi0", "tau_pairs", "energy")
    np.testing.assert_allclose(values["theta"], [0.01, -0.02, -0.02], atol=1e-9)
    np.testing.assert_allclose(values["phi"], [-0.02, 0.01, 0.03], atol=1e-9)
    np.testing.assert_allclose(values["energy"], [-1.0, 1.0, 0.0])
    np.testing.assert_array_equal(seeds, [7, 9, 9])
    np.testing.assert_allclose(measure.azimuth_residual(np.array([179.0, -179.0, 10.0, nan]),
                                                        np.array([-179.0, 179.0, -10.0, 0.0])),
                               [-2.0, 2.0, 20.0, nan])


def test_a_simulator_without_good_pi0s_is_left_out(figures, tmp_path):
    model = pi0_simulator("model", "Model", "#e66100", "--", 1.0, 2)
    stored = model.stored(None, None)
    stored["match"][:] = measure.CONFUSED
    results, _ = plot.pi0_figures("pi0", "tau_pairs", [model], None, {}, str(tmp_path), "",
                                  ["png"])
    # Only the categories are drawn, and without Geant4 there's no deviation
    assert results == [] and len(figures) == 1
    assert [p.name for p in tmp_path.iterdir()] == ["pi0_categories__tau_pairs.png"]


def categorised(n_candidates, match, true_index, n_true, seeds=None):
    stored = {"n_candidates": np.array(n_candidates), "match": np.array(match),
              "true_index": np.array(true_index), "n_true": np.array(n_true)}
    data = {} if seeds is None else {"tau_pairs__n_events": np.array([len(n_true) // len(seeds)]
                                                                     * len(seeds)),
                                     "tau_pairs__seeds": np.array(seeds)}
    return types.SimpleNamespace(data=data, stored=lambda observable, scenario: stored)


def test_the_pi0_categories_are_counted_per_event():
    G, C, F = measure.GOOD, measure.CONFUSED, measure.FAKE
    # Event 0: one good. Event 1: none of its 2. Event 2: a good, a confused
    # of the same MC pi0 (they share a photon) and a fake; its other pi0 missed.
    # Event 3: two confused of its two pi0s.
    simulator = categorised([1, 0, 3, 2], [G, G, C, F, C, C], [0, 1, 1, -1, 0, 1], [1, 2, 2, 2])
    categories, seeds = plot.pi0_categories(simulator, "pi0", "tau_pairs")
    expected = {"n_reco": [1, 0, 3, 2], "n_good": [1, 0, 1, 0], "n_confused": [0, 0, 1, 2],
                "n_fake": [0, 0, 1, 0], "n_missed": [0, 2, 1, 0]}
    for key, values in expected.items():
        np.testing.assert_array_equal(categories[key], values, err_msg=key)
    np.testing.assert_array_equal(seeds, [-1] * 4)


def test_the_category_deviations_carry_the_spreads_of_both(figures, tmp_path):
    keys = [key for key, _ in plot.PI0_CATEGORIES]
    geant4 = types.SimpleNamespace(name="geant4", label="Geant4", color="#8c8c8c", linestyle="-")
    model = types.SimpleNamespace(name="model", label="Model", color="#e66100", linestyle="--")

    def totals(mean, error):
        return {"seeds": [1, 2, 3], "mean": dict(zip(keys, mean)),
                "error": dict(zip(keys, error)), "spread": True,
                "events": {1: 100, 2: 100, 3: 100}}

    g_mean, g_error = [400.0, 50.0, 280.0, 70.0, 500.0], [10.0, 5.0, 12.0, 2.0, 11.0]
    m_mean, m_error = [440.0, 50.0, 320.0, 75.0, 460.0], [9.0, 6.0, 4.0, 1.0, 5.0]
    results = plot.plot_pi0_categories("pi0", "tau_pairs", [(geant4, totals(g_mean, g_error)),
                                                            (model, totals(m_mean, m_error))],
                                       geant4, str(tmp_path), "", ["png"])
    dev, band = plot.relative_deviation(m_mean, m_error, g_mean, g_error)
    assert [row[:3] for row in results] == [(model, "categories_" + key, "dev%") for key in keys
                                            ] + [(model, "categories", "chi2")]
    np.testing.assert_allclose([row[3] for row in results[:5]], dev)
    np.testing.assert_allclose([row[4] for row in results[:5]], band)
    # Below, the model's error bars alone, of that combined uncertainty
    (bars,) = figures[0].axes[1].collections
    np.testing.assert_allclose([s[1, 1] - s[0, 1] for s in bars.get_segments()], 2 * band)
    # and its chi^2 over the categories, with the errors drawn (the spreads),
    # in the legend as chi^2/ndf
    chi2 = sum((m - g) ** 2 / (me ** 2 + ge ** 2)
               for m, me, g, ge in zip(m_mean, m_error, g_mean, g_error))
    assert len(results[5]) == 4 and results[5][3] == pytest.approx(chi2)
    labels = [text.get_text() for text in figures[0].axes[0].get_legend().get_texts()]
    assert labels == ["Geant4", "Model ($\\chi^2$/ndf %.3g/5)" % chi2]


def test_the_pi0_category_totals_are_per_seed():
    G, C, F = measure.GOOD, measure.CONFUSED, measure.FAKE
    simulator = categorised([1, 2, 1, 0], [G, G, F, C], [0, 0, -1, 0], [1, 1, 1, 1], seeds=[7, 9])
    points = plot.category_totals(*plot.pi0_categories(simulator, "pi0", "tau_pairs"))
    assert points["seeds"] == [7, 9] and points["events"] == {7: 2, 9: 2} and points["spread"]
    np.testing.assert_array_equal(points["totals"]["n_reco"], [3, 1])
    np.testing.assert_array_equal(points["totals"]["n_fake"], [1, 0])
    assert points["mean"]["n_good"] == 1.0 and points["error"]["n_good"] == pytest.approx(
        np.std([2, 0], ddof=1))
    np.testing.assert_array_equal(points["totals"]["n_missed"], [0, 1])


def stored_style(folder, name, label, color, linestyle):
    """A measurement with nothing in it but how it was described when measured."""
    path = os.path.join(str(folder), name + ".npz")
    np.savez(path, meta__simulator=np.array(name), meta__label=np.array(label),
             meta__color=np.array(color), meta__linestyle=np.array(json.dumps(linestyle)))
    return path


def test_a_model_is_drawn_as_described_now_not_as_when_measured(tmp_path, monkeypatch,
                                                                    capsys):
    from ddml_bench import models
    described = tmp_path / "models"
    described.mkdir()
    (described / "model.yaml").write_text(
        'name: Model (new)\nplot:\n  color: "#123456"\n  linestyle: [0, [6, 2, 1, 2]]\n')
    (described / "broken.yaml").write_text("plot: [\n")
    monkeypatch.setattr(models, "MODELS_DIR", str(described))

    model = plot.Simulator(stored_style(tmp_path, "model", "Model (old)", "#abcdef", "-"))
    assert (model.label, model.color, model.linestyle) == ("Model (new)", "#123456",
                                                           (0, (6, 2, 1, 2)))
    assert "drawn as models/ describes it now" in capsys.readouterr().err

    # Geant4 is drawn as always, however it was stored
    geant4 = plot.Simulator(stored_style(tmp_path, "geant4", "G4", "#000000", ":"))
    assert (geant4.label, geant4.color, geant4.linestyle) == (
        models.GEANT4_STYLE["label"], models.GEANT4_STYLE["color"], "-")

    # A measurement whose model isn't described, or can't be read, as measured
    for name in ("fetched", "broken"):
        other = plot.Simulator(stored_style(tmp_path, name, "Other", "#fedcba", "--"))
        assert (other.label, other.color, other.linestyle) == ("Other", "#fedcba", "--")
