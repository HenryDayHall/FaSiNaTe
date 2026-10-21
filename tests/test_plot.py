"""The figures: what each kind of profile is drawn as, and how it is compared."""

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
