"""The Pareto fronts: of each scenario, every metric's points on one axis, each
model's Jensen-Shannon distance, or Wasserstein distance, or chi^2, from
Geant4, divided by the models' mean, against its time per event from the
timing benchmark's table, on each kind of node (ddml_bench/pareto.py)."""

import csv
import math
import os
import re
import types

import pytest

from ddml_bench import dashboard, pareto, plot
from ddml_bench.stages import BY_NAME
from test_dashboard import marks, measurement

GEANT4 = types.SimpleNamespace(name="geant4", label="Geant4", color="#8c8c8c", linestyle="-")
ALPHA = types.SimpleNamespace(name="alpha", label="Alpha", color="#e66100", linestyle="--")
BETA = types.SimpleNamespace(name="beta", label="Beta", color="#1f6fb4", linestyle="-.")
GAMMA = types.SimpleNamespace(name="gamma", label="Gamma", color="#2a9a3a", linestyle=":")
SIMULATORS = [GEANT4, ALPHA, BETA, GAMMA]
TIMING_COLUMNS = ("device", "scenario", "simulator", "input", "events", "warmup", "mean_s",
                  "sd_s", "sem_s", "speedup", "speedup_reference")
FIGURES = ["pareto_js_cpu__single_photon", "pareto_js_gpu__single_photon",
           "pareto_js_cpu__tau_pairs", "pareto_wasserstein_cpu__single_photon",
           "pareto_wasserstein_gpu__single_photon", "pareto_wasserstein_cpu__tau_pairs"]
CHI2 = "$\\chi^2$"


@pytest.fixture
def figures(monkeypatch):
    """The figures drawn, kept open to look at."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    close = plt.close
    kept = []
    monkeypatch.setattr(plt, "close", kept.append)
    yield kept
    for figure in kept:
        close(figure)


def write_timing(path, rows):
    """A timing table as timing_summary.sh writes it, of rows (device,
    scenario, simulator, input, events, mean_s, sem_s)."""
    with open(str(path), "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(TIMING_COLUMNS)
        for device, scenario, simulator, name, events, mean, sem in rows:
            writer.writerow([device, scenario, simulator, name, events, 10, mean, "", sem, "",
                             ""])
    return str(path)


def write_metrics(path, rows):
    """metrics.csv of rows (observable, scenario, profile, simulator, metric,
    value, uncertainty)."""
    with open(str(path), "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["observable", "scenario", "profile", "simulator", "metric", "value",
                         "uncertainty"])
        writer.writerows(rows)
    return str(path)


def read_rows(path):
    with open(str(path), newline="") as stream:
        return list(csv.DictReader(stream))


def close(value):
    """value, as a CSV file of 6 significant figures has it."""
    return pytest.approx(value, rel=1e-5)


def plain(ax):
    """Whether the axes draw no line, only points (and their error bars)."""
    return all(line.get_linestyle() in ("None", "none", "") for line in ax.lines)


def points_of(ax):
    """{(x, marker, colour): the point's line} of the axes' points."""
    return dict(((line.get_xdata()[0], line.get_marker(), line.get_color()), line)
                for line in ax.lines if line.get_marker() not in (None, "None", ""))


def test_the_distance_is_the_square_root_of_the_divergence_over_the_models_mean():
    distances, mean = pareto.relative_distances({"a": (0.04, None), "b": (0.01, 0.002)})
    assert mean == pytest.approx(0.15)
    value, distance, error, relative, relative_error = distances["a"]
    assert (value, distance, error) == (0.04, pytest.approx(0.2), None)
    assert relative == pytest.approx(0.2 / 0.15) and relative_error is None
    # The divergence's uncertainty, carried to first order: sigma / (2 sqrt(D))
    _, distance, error, relative, relative_error = distances["b"]
    assert distance == pytest.approx(0.1) and error == pytest.approx(0.002 / 0.2)
    assert relative == pytest.approx(0.1 / 0.15)
    assert relative_error == pytest.approx(0.01 / 0.15)
    # So the models' mean is at 1, whatever the divergences' scale (e.g. the
    # base of their logarithm)
    assert sum(d[3] for d in distances.values()) / len(distances) == pytest.approx(1)
    scaled, _ = pareto.relative_distances({"a": (4.0, None), "b": (1.0, None)})
    assert scaled["a"][3] == pytest.approx(distances["a"][3])
    # and nothing to divide by, no distances
    assert pareto.relative_distances({"a": (0.0, None)}) == ({}, 0.0)


def test_the_versions_measure_the_distributions_and_the_rest_by_the_chi_squared():
    distances, mean = pareto.relative_distances({"a": (0.3, 0.1), "b": (0.1, None)},
                                                pareto.as_distance)
    assert mean == pytest.approx(0.2)
    assert distances["a"] == (0.3, 0.3, 0.1, pytest.approx(1.5), pytest.approx(0.5))
    assert distances["b"] == (0.1, 0.1, None, pytest.approx(0.5), None)
    # The JS version and the Wasserstein version, of the figures of
    # distributions; the others, by their chi^2 in both
    assert [(name, [m[0] for m in measures]) for name, measures in pareto.VERSIONS] == [
        ("js", ["JS", "chi2"]), ("wasserstein", ["W1", "chi2"])]


def test_a_point_is_on_the_front_unless_another_is_as_fast_and_as_close_and_better():
    points = [(1.0, 0.0),   # Geant4: no other is as close
              (0.1, 1.0),   # the fastest
              (0.2, 0.5),
              (0.3, 0.6),   # slower and farther than (0.2, 0.5)
              (0.1, 1.2),   # as fast as (0.1, 1.0), but farther
              (0.2, 0.5),   # the same as another: neither is better
              (1.5, 0.4)]   # closer than all but Geant4, but slower than it
    assert pareto.pareto_front(points) == [True, True, True, False, False, True, False]


def test_the_time_of_a_scenario_is_its_row_over_all_its_inputs(tmp_path):
    path = write_timing(tmp_path / "summary.csv", [
        ("cpu", "single_photon", "geant4", "p22_en10_t200", 200, 0.6, 0.01),
        ("cpu", "single_photon", "geant4", "p22_en50_t200", 200, 2.6, 0.02),
        ("cpu", "single_photon", "geant4", "all", 400, 1.6, 0.011),
        # Of two numbers of events timed, the one of the most events
        ("cpu", "single_photon", "alpha", "all_t5", 10, 0.5, 0.2),
        ("cpu", "single_photon", "alpha", "all_t200", 400, 0.02, 0.001),
        ("gpu", "single_photon", "alpha", "all", 400, 0.01, ""),
        # No time, no row
        ("gpu", "single_photon", "beta", "all", 0, "", "")])
    assert pareto.read_timing(path) == {
        ("cpu", "single_photon", "geant4"): {"time": 1.6, "error": 0.011, "events": 400},
        ("cpu", "single_photon", "alpha"): {"time": 0.02, "error": 0.001, "events": 400},
        ("gpu", "single_photon", "alpha"): {"time": 0.01, "error": None, "events": 400}}
    with open(path, "w") as stream:
        stream.write("observable,scenario,profile,simulator,metric,value,uncertainty\n")
    with pytest.raises(pareto.TimingError, match="isn't the timing benchmark's table"):
        pareto.read_timing(path)


def test_geant4_is_taken_from_the_cpu_nodes_where_it_isnt_timed_on_the_node():
    times = {("cpu", "s", "geant4"): {"time": 2.0}, ("cpu", "s", "alpha"): {"time": 0.1},
             ("gpu", "s", "alpha"): {"time": 0.05}, ("gpu", "other", "geant4"): {"time": 9.0}}
    models, reference = pareto.node_times(times, "gpu", "s", "geant4")
    assert models == {"alpha": {"time": 0.05}}
    assert reference == ({"time": 2.0}, "cpu")
    assert pareto.node_times(times, "cpu", "s", "geant4")[1] == ({"time": 2.0}, "cpu")
    assert pareto.node_times(times, "cpu", "elsewhere", "geant4") == ({}, None)


def example(tmp_path):
    """metrics.csv and a timing table: single_photon timed on the CPU nodes
    (Geant4 and all three models) and on the GPU nodes (alpha), di_photon on
    neither."""
    metrics = write_metrics(tmp_path / "metrics.csv", [
        ("radial_profile", "single_photon", "r30", "alpha", "JS", 0.0004, ""),
        ("radial_profile", "single_photon", "r30", "beta", "JS", 0.0001, ""),
        ("radial_profile", "single_photon", "r30", "gamma", "JS", 0.0009, ""),
        ("radial_profile", "single_photon", "r30", "alpha", "W1", 0.6, ""),
        ("radial_profile", "single_photon", "r30", "beta", "W1", 0.3, ""),
        ("radial_profile", "single_photon", "r30", "gamma", "W1", 0.15, ""),
        ("energy", "single_photon", "linearity", "alpha", "L1", 0.2, ""),
        ("energy", "single_photon", "linearity", "alpha", "chi2", 300.0, ""),
        ("energy", "single_photon", "linearity", "beta", "chi2", 600.0, ""),
        ("energy", "single_photon", "linearity", "gamma", "chi2", 900.0, ""),
        ("pi0", "tau_pairs", "mass", "alpha", "JS", 0.0016, 0.0004),
        ("pi0", "tau_pairs", "mass", "beta", "JS", 0.0004, 0.0002),
        ("pi0", "tau_pairs", "mass", "alpha", "W1", 0.002, 0.0005),
        ("pi0", "tau_pairs", "mass", "beta", "W1", 0.001, 0.0002),
        ("pi0", "tau_pairs", "categories_n_good", "alpha", "dev%", 3.0, 1.0),
        ("pi0", "tau_pairs", "categories", "alpha", "chi2", 3.5, ""),
        ("pi0", "tau_pairs", "categories", "beta", "chi2", 37.5, ""),
        ("photon_separation", "di_photon", "20GeV", "alpha", "JS", 3e-5, 1e-5),
        ("photon_separation", "di_photon", "20GeV", "alpha", "W1", 0.2, 0.05)])
    timing = write_timing(tmp_path / "summary.csv", [
        ("cpu", "single_photon", "geant4", "all", 2000, 2.85, 0.008),
        ("cpu", "single_photon", "alpha", "all", 2000, 0.06, 0.002),
        ("cpu", "single_photon", "beta", "all", 2000, 0.5, 0.004),
        ("cpu", "single_photon", "gamma", "all", 2000, 2.5, 0.007),
        ("gpu", "single_photon", "alpha", "all", 2000, 0.02, 0.0004),
        ("cpu", "tau_pairs", "geant4", "all", 600, 6.8, 0.1),
        ("cpu", "tau_pairs", "alpha", "all", 600, 4.1, 0.1),
        ("cpu", "tau_pairs", "beta", "all", 600, 9.6, 0.16)])
    return metrics, timing


def test_a_figure_of_every_metric_of_each_scenario_version_and_node(figures, tmp_path,
                                                                     capsys):
    metrics, timing = example(tmp_path)
    out = tmp_path / "plots"
    os.makedirs(str(out))
    rows = pareto.fronts(str(out), "", ["png"], SIMULATORS, "geant4", timing,
                         metrics_path=metrics)
    assert sorted(name for name in os.listdir(str(out)) if name.endswith(".png")) == sorted(
        name + ".png" for name in FIGURES)
    # The di-photons aren't timed; the L1 and the categories' deviations
    # (dev%) aren't drawn, their chi^2 is
    assert "none of the models of di_photon is timed" in capsys.readouterr().err
    written = read_rows(out / "pareto.csv")
    assert len(rows) == len(written)
    single = (("radial_profile", "r30"), ("energy", "linearity"))
    tau = (("pi0", "mass"), ("pi0", "categories"))
    assert [(r["version"], r["node"], r["scenario"], r["observable"], r["profile"],
             r["simulator"]) for r in written] == [
        (version, node, scenario, observable, profile, simulator)
        for version in ("js", "wasserstein")
        for node, scenario, figures_of, simulators in (
            ("cpu", "single_photon", single, ("geant4", "alpha", "beta", "gamma")),
            ("gpu", "single_photon", single, ("geant4", "alpha")),
            ("cpu", "tau_pairs", tau, ("geant4", "alpha", "beta")))
        for observable, profile in figures_of for simulator in simulators]

    by = dict(((r["version"], r["node"], r["profile"], r["simulator"]), r) for r in written)
    # The JS distances, 0.02, 0.01 and 0.03, over their mean, 0.02 (pareto.csv
    # has 6 significant figures)
    mean = (0.02 + 0.01 + 0.03) / 3
    for name, distance in (("alpha", 0.02), ("beta", 0.01), ("gamma", 0.03)):
        row = by[("js", "cpu", "r30", name)]
        assert row["metric"] == "JS" and float(row["value"]) == close(distance ** 2)
        assert float(row["distance"]) == close(distance) and row["unit"] == ""
        assert float(row["relative_distance"]) == close(distance / mean)
        assert float(row["mean_distance"]) == close(mean)
        assert row["models"] == "3" and row["timed_on"] == "cpu"
    # Each metric its own front: alpha is the fastest, beta faster than Geant4
    # and closer than alpha; gamma is slower than beta and farther than both
    assert [by[("js", "cpu", "r30", n)]["on_front"] for n in
            ("geant4", "alpha", "beta", "gamma")] == ["yes", "yes", "yes", "no"]
    assert float(by[("js", "cpu", "r30", "alpha")]["speedup"]) == close(2.85 / 0.06)
    # The Wasserstein distances as they are, in mm, over their mean, 0.35:
    # gamma is the closest of all, so on this front
    for name, distance in (("alpha", 0.6), ("beta", 0.3), ("gamma", 0.15)):
        row = by[("wasserstein", "cpu", "r30", name)]
        assert row["metric"] == "W1" and float(row["value"]) == close(distance)
        assert float(row["distance"]) == close(distance) and row["unit"] == "mm"
        assert float(row["relative_distance"]) == close(distance / 0.35)
    assert [by[("wasserstein", "cpu", "r30", n)]["on_front"] for n in
            ("geant4", "alpha", "beta", "gamma")] == ["yes", "yes", "yes", "yes"]
    # The energy by its chi^2 in both versions, over its mean, 600: alpha is
    # the fastest and the closest, so the others are off its front
    for version in ("js", "wasserstein"):
        for name, value in (("alpha", 300.0), ("beta", 600.0), ("gamma", 900.0)):
            row = by[(version, "cpu", "linearity", name)]
            assert row["metric"] == "chi2" and float(row["distance"]) == close(value)
            assert float(row["relative_distance"]) == close(value / 600) and row["unit"] == ""
        assert [by[(version, "cpu", "linearity", n)]["on_front"] for n in
                ("geant4", "alpha", "beta", "gamma")] == ["yes", "yes", "no", "no"]
        assert [(by[(version, "cpu", "categories", n)]["metric"],
                 float(by[(version, "cpu", "categories", n)]["relative_distance"]),
                 by[(version, "cpu", "categories", n)]["on_front"]) for n in
                ("geant4", "alpha", "beta")] == [
            ("chi2", 0.0, "yes"), ("chi2", close(3.5 / 20.5), "yes"),
            ("chi2", close(37.5 / 20.5), "no")]
    # On the GPU nodes, alpha has the same distances as on the CPU nodes (the
    # means of all the models, timed there or not), against Geant4 on the CPU
    # nodes
    for version in ("js", "wasserstein"):
        for profile in ("r30", "linearity"):
            gpu = by[(version, "gpu", profile, "alpha")]
            assert gpu["relative_distance"] == \
                by[(version, "cpu", profile, "alpha")]["relative_distance"]
            assert float(gpu["time_s"]) == 0.02 and float(gpu["speedup"]) == close(2.85 / 0.02)
        reference = by[(version, "gpu", "r30", "geant4")]
        assert reference["timed_on"] == "cpu" and float(reference["time_s"]) == 2.85
    # The pi0s' uncertainties, carried to the distances
    alpha = by[("js", "cpu", "mass", "alpha")]
    assert float(alpha["distance_error"]) == close(0.0004 / (2 * 0.04))
    assert float(alpha["relative_distance_error"]) == close(
        0.0004 / (2 * 0.04) / ((0.04 + 0.02) / 2))
    alpha = by[("wasserstein", "cpu", "mass", "alpha")]
    assert alpha["unit"] == "GeV" and float(alpha["distance_error"]) == close(0.0005)
    # beta, slower than Geant4, is off the fronts however close it is
    assert by[("js", "cpu", "mass", "beta")]["on_front"] == "no"

    # Every figure of a version has the same axis of distance, from below 0
    # (so that Geant4's marker shows whole)
    assert len(set(fig.axes[0].get_ylim() for fig in figures[:3])) == 1
    assert len(set(fig.axes[0].get_ylim() for fig in figures[3:])) == 1
    low, high = figures[0].axes[0].get_ylim()
    assert low < 0 < 1.5 < high
    # Every metric of the scenario on one axis: a marker each, a colour each
    # model, filled on the metric's front and hollow off it, and no line
    cpu = figures[0].axes[0]
    assert all(plain(fig.axes[0]) for fig in figures)
    assert cpu.get_xscale() == "log"
    drawn = points_of(cpu)
    square, triangle = pareto.METRIC_MARKERS[:2]
    assert len(drawn) == 1 + 3 * 2
    assert drawn[(2.85, "o", "#8c8c8c")].get_ydata()[0] == 0
    radial, energy = drawn[(0.06, square, ALPHA.color)], drawn[(0.06, triangle, ALPHA.color)]
    assert radial.get_ydata()[0] == pytest.approx(0.02 / mean)
    assert energy.get_ydata()[0] == pytest.approx(0.5)
    assert radial.get_markerfacecolor() == energy.get_markerfacecolor() == ALPHA.color
    gamma = drawn[(2.5, square, GAMMA.color)]
    assert gamma.get_markerfacecolor() == "white" and gamma.get_markeredgecolor() == GAMMA.color
    assert drawn[(0.5, triangle, BETA.color)].get_markerfacecolor() == "white"
    # A model's metrics are drawn side by side about its time, at the time itself
    figures[0].canvas.draw()
    shown = [line.get_transform().transform(line.get_xydata())[0][0] for line in (radial, energy)]
    assert shown[1] - shown[0] == pytest.approx(pareto.DODGE * figures[0].dpi / 72)
    assert cpu.transData.transform((0.06, 0))[0] == pytest.approx(sum(shown) / 2)
    assert cpu.get_ylabel() == "JS distance or %s to Geant4 / the models' mean" % CHI2
    labels = [text.get_text() for text in cpu.get_legend().get_texts()]
    assert labels == ["Geant4", "Alpha, ×47.5", "Beta, ×5.7", "Gamma, ×1.14",
                      "Metrics (the models' mean):", "radial profile r30: 0.02",
                      "energy linearity (%s): 600" % CHI2]
    gpu_labels = [text.get_text() for text in figures[1].axes[0].get_legend().get_texts()]
    assert gpu_labels[:2] == ["Geant4 (cpu nodes)", "Alpha, ×142"]
    wasserstein = figures[3].axes[0]
    assert wasserstein.get_ylabel() == \
        "Wasserstein distance or %s to Geant4 / the models' mean" % CHI2
    assert [text.get_text() for text in wasserstein.get_legend().get_texts()][-2:] == [
        "radial profile r30: 0.35 mm", "energy linearity (%s): 600" % CHI2]
    assert points_of(wasserstein)[(2.5, square, GAMMA.color)].get_markerfacecolor() == \
        GAMMA.color
    tau_labels = [text.get_text() for text in figures[2].axes[0].get_legend().get_texts()]
    assert tau_labels[-2:] == ["pi0 mass: 0.03", "pi0 categories (%s): 20.5" % CHI2]


def test_the_dashboard_can_hide_each_model_in_the_fronts(figures, tmp_path):
    metrics, timing = example(tmp_path)
    out = tmp_path / "plots"
    os.makedirs(str(out))
    gathered = dashboard.Collector(str(out))
    with plot.collecting(gathered):
        pareto.fronts(str(out), "", [], SIMULATORS, "geant4", timing, metrics_path=metrics)
    # Each version apart, for the menu to pick, in the scenario's section
    assert [(f["name"], f["observable"], f["scenario"]) for f in gathered.figures] == [
        (name, name.split("_")[0] + "_" + name.split("_")[1], name.rsplit("__", 1)[1])
        for name in FIGURES]
    assert list(gathered.simulators) == ["alpha", "beta", "gamma"]
    page = dashboard.render(gathered)
    assert page.count('data-observable="pareto_js"') == 3
    assert page.count('data-observable="pareto_wasserstein"') == 3
    # Each model's points and legend entry, in each figure it is in, to hide
    for k, figures_in in enumerate((6, 4, 2)):
        assert len(re.findall(r'class="fsn-d fsn-s%d"' % k, page)) >= 2 * figures_in
        assert len(re.findall(r'class="fsn-k fsn-s%d"' % k, page)) == 2 * figures_in
        assert len(re.findall(r'class="fsn-t fsn-s%d"' % k, page)) == figures_in


def test_only_the_models_points_are_marked(figures, tmp_path):
    metrics, timing = example(tmp_path)
    pareto.fronts(str(tmp_path), "", [], SIMULATORS, "geant4", timing, metrics_path=metrics)
    found = marks(figures[0])
    for name in ("alpha", "beta", "gamma"):
        assert found[(dashboard.DATA, name)] >= 2
        assert found[(dashboard.KEY, name)] == 2
    # Geant4 and the metrics' markers in the legend are always there
    assert not any(name == "geant4" for _, name in found)
    assert sum(count for (kind, _), count in found.items() if kind == dashboard.KEY) == 6


def test_plotting_draws_the_fronts_where_there_is_a_timing_table(tmp_path):
    observables = tmp_path / "data" / "observables"
    os.makedirs(str(observables))
    for name, label, color, scale in (("geant4", "Geant4", "#8c8c8c", 5.0),
                                      ("alpha", "Alpha", "#e66100", 5.5),
                                      ("beta", "Beta", "#1f6fb4", 6.0)):
        measurement(observables, name, label, color, "-", scale)
    plots = observables / "plots"
    # Without a timing table, no fronts
    assert plot.main(["--dir", str(observables), "--format", "png", "--no-closure",
                      "--no-dashboard"]) == 0
    assert not [name for name in os.listdir(str(plots)) if name.startswith("pareto")]

    # With one in the data folder, timing/summary.csv
    os.makedirs(str(tmp_path / "data" / "timing"))
    write_timing(tmp_path / "data" / "timing" / "summary.csv", [
        ("cpu", "single_photon", "geant4", "all", 2000, 2.0, 0.01),
        ("cpu", "single_photon", "alpha", "all", 2000, 0.05, 0.001),
        ("cpu", "single_photon", "beta", "all", 2000, 0.1, 0.001)])
    assert plot.main(["--dir", str(observables), "--format", "png", "--no-closure"]) == 0
    for version in ("js", "wasserstein"):
        assert os.path.isfile(str(plots / ("pareto_%s_cpu__single_photon.png" % version)))
    metrics = read_rows(plots / "metrics.csv")
    rows = read_rows(plots / "pareto.csv")
    # The distances of the metrics plotted: the square roots of the divergences,
    # and the Wasserstein distances as they are
    for version, metric, to_distance in (("js", "JS", math.sqrt), ("wasserstein", "W1", float)):
        values = dict((r["simulator"], to_distance(float(r["value"])))
                      for r in metrics if r["metric"] == metric)
        assert sorted(values) == ["alpha", "beta"]
        mean = sum(values.values()) / 2
        for name in ("alpha", "beta"):
            (row,) = [r for r in rows if r["version"] == version and r["simulator"] == name]
            assert float(row["relative_distance"]) == pytest.approx(values[name] / mean,
                                                                    rel=1e-5)
    page = (plots / "dashboard.html").read_text(encoding="utf-8")
    assert page.count('data-observable="pareto_js"') == 1
    assert page.count('data-observable="pareto_wasserstein"') == 1

    # --no-pareto leaves them out, and --timing gives another table
    out = tmp_path / "elsewhere"
    assert plot.main(["--dir", str(observables), "--out", str(out), "--format", "png",
                      "--no-closure", "--no-dashboard", "--no-pareto"]) == 0
    assert not [name for name in os.listdir(str(out)) if name.startswith("pareto")]
    other = write_timing(tmp_path / "other.csv", [
        ("gpu", "single_photon", "alpha", "all", 2000, 0.01, 0.001)])
    assert plot.main(["--dir", str(observables), "--out", str(out), "--format", "png",
                      "--no-closure", "--no-dashboard", "--timing", other]) == 0
    assert sorted(name for name in os.listdir(str(out)) if name.startswith("pareto")) == [
        "pareto.csv", "pareto_js_gpu__single_photon.png",
        "pareto_wasserstein_gpu__single_photon.png"]


def test_the_plots_are_drawn_again_when_the_timing_table_is_newer(tmp_path):
    observables = tmp_path / "observables"
    os.makedirs(str(observables / "plots"))
    measurement(observables, "geant4", "Geant4", "#8c8c8c", "-", 5.0)
    metrics = observables / "plots" / "metrics.csv"
    metrics.write_text("observable\n")
    task = {"params": {"dir": str(observables), "suffix": ""}, "output": str(metrics)}
    os.utime(str(observables / "geant4.npz"), (1000, 1000))
    os.utime(str(metrics), (2000, 2000))
    stage = BY_NAME["plot"]
    assert stage.current(task) == (True, "")
    os.makedirs(str(tmp_path / "timing"))
    summary = tmp_path / "timing" / "summary.csv"
    summary.write_text("device\n")
    os.utime(str(summary), (1500, 1500))
    assert stage.current(task) == (True, "")
    os.utime(str(summary), (3000, 3000))
    done, why = stage.current(task)
    assert not done and why.startswith("timed again since the last plots")
