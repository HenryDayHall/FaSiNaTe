"""The dashboard: each model's parts of the figures marked with its name, and
every figure in one page with a menu to show or hide each model
(ddml_bench/dashboard.py)."""

import json
import os
import re
import types
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from ddml_bench import dashboard, measure, plot
from test_plot import (angle_simulator, di_photon_sample, energy_sample, pi0_simulator,
                       residual_sample)

GEANT4 = types.SimpleNamespace(name="geant4", label="Geant4", color="#8c8c8c", linestyle="-")
MODEL = types.SimpleNamespace(name="model", label="Model", color="#e66100", linestyle="--")
OTHER = types.SimpleNamespace(name="other.v2", label="Other", color="#2a9a3a",
                              linestyle=(0, (3, 1, 1, 1)))


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


def radial_profile(out, formats=("svg",), models=(MODEL, OTHER)):
    """A radial profile of Geant4 and the models, drawn to out."""
    edges = measure.bin_edges((0.0, 30.0, 30))
    centres = (edges[1:] + edges[:-1]) / 2
    curves = [(GEANT4, edges, np.exp(-centres / 5), 0.05 * np.exp(-centres / 5), 100)]
    for i, model in enumerate(models):
        mean = np.exp(-centres / (5.5 + i))
        curves.append((model, edges, mean, 0.05 * mean, 100))
    return plot.plot_profile("radial_profile", "single_photon", "r30", curves, GEANT4,
                             "jensen_shannon", str(out), list(formats))


def unmarked(fig, color):
    """The parts of a figure drawn in a colour that aren't marked."""
    from matplotlib.colors import to_hex

    def colors(artist):
        found = []
        for getter in ("get_color", "get_edgecolor", "get_facecolor"):
            try:
                value = getattr(artist, getter)()
            except (AttributeError, TypeError):
                continue
            values = np.atleast_2d(value) if not isinstance(value, str) else [value]
            for each in values:
                try:
                    found.append(to_hex(each, keep_alpha=False))
                except ValueError:
                    pass
        return found

    return [artist for artist in fig.findobj()
            if color in colors(artist) and not (artist.get_gid() or "").startswith("fsn.")]


def marks(fig):
    """{(role, simulator): how many parts} of the marks in a figure."""
    found = {}
    for artist in fig.findobj():
        parsed = dashboard.parse_mark((artist.get_gid() or "") + ".1")
        if parsed:
            found[parsed] = found.get(parsed, 0) + 1
    return found


def test_every_part_of_a_model_is_marked_in_every_kind_of_figure(figures, tmp_path):
    radial_profile(tmp_path / "radial", formats=())
    plot.plot_energy("energy", "single_photon",
                     [(GEANT4,) + energy_sample(0.98, 0.17, 1),
                      (MODEL,) + energy_sample(0.97, 0.19, 2)],
                     GEANT4, "l1", str(tmp_path), "", [])
    geant4 = angle_simulator("geant4", "Geant4", GEANT4.color, "-", residual_sample(1.0, 0, 1))
    model = angle_simulator("model", "Model", MODEL.color, "--", residual_sample(1.2, 0.2, 2))
    plot.angle_figures("intrinsic_angle", "single_photon", [geant4, model], geant4, {},
                       str(tmp_path), "", [])
    plot.plot_separation("photon_separation", "di_photon",
                         [(GEANT4,) + tuple(di_photon_sample(20, 1)),
                          (MODEL,) + tuple(di_photon_sample(24, 2))],
                         GEANT4, {}, str(tmp_path), "", [])
    geant4 = pi0_simulator("geant4", "Geant4", GEANT4.color, "-", 1.0, 1)
    model = pi0_simulator("model", "Model", MODEL.color, "--", 0.98, 2)
    plot.pi0_figures("pi0", "tau_pairs", [geant4, model], geant4, {}, str(tmp_path), "", [])
    # radial, 2 energy, 4 angle, 1 separation, and the pi0 mass, energy and categories
    assert len(figures) == 11
    for fig in figures:
        # Nothing drawn in the model's colour is left out, so the menu hides all of it
        assert unmarked(fig, MODEL.color) == []
        found = marks(fig)
        assert found[(dashboard.DATA, "model")] >= 2
        # and its legend entry, the sample of its line and its text, dims
        assert found[(dashboard.KEY, "model")] == 2
        # Geant4, the reference, is never marked, so it is always there
        assert not any(name == "geant4" for _, name in found)


def test_the_svg_has_each_mark_once_and_a_target_on_each_models_legend_entry(tmp_path):
    radial_profile(tmp_path / "radial")
    root = ET.parse(str(tmp_path / "radial.svg")).getroot()
    ids = [element.get("id") for element in root.iter() if element.get("id")]
    assert len(ids) == len(set(ids))
    found = {}
    for ident in ids:
        parsed = dashboard.parse_mark(ident)
        if parsed:
            found[parsed] = found.get(parsed, 0) + 1
    for name in ("model", "other.v2"):
        # its line and band, and its deviation from Geant4 with its band below
        assert found[(dashboard.DATA, name)] == 4
        assert found[(dashboard.KEY, name)] == 2
        assert found[(dashboard.TARGET, name)] == 1
    # Only the SVG was asked for
    assert sorted(os.listdir(str(tmp_path))) == ["radial.svg"]


def test_a_legend_entrys_target_covers_its_sample_and_text(figures):
    fig, axes = plot.new_figure(False)
    line, = axes[0][0].plot([0, 1], [0, 1], color=MODEL.color)
    drawn = plot.legend(axes[0][0], [(line, "Model (JS 1.00e-03)", MODEL)], fontsize=8,
                        loc="upper right")
    plot.legend_targets(fig)
    (target,) = [artist for artist in fig.artists
                 if (artist.get_gid() or "").startswith("fsn.target.model")]
    renderer = fig.canvas.get_renderer()
    box = target.get_window_extent(renderer)
    for part in (drawn.get_texts()[0], plot.legend_handles(drawn)[0]):
        inner = part.get_window_extent(renderer)
        assert box.x0 <= inner.x0 and inner.x1 <= box.x1
        assert box.y0 <= inner.y0 and inner.y1 <= box.y1
    # invisible, in any format
    assert target.get_alpha() == 0
    figures.append(fig)


def page_of(tmp_path):
    """The page of a radial profile and the energy figures of Geant4 and the
    models, through a collector as plot.main() makes it."""
    collector = dashboard.Collector(str(tmp_path))
    with plot.collecting(collector):
        collector.section(str(tmp_path), GEANT4)
        collector.context("radial_profile", "single_photon")
        radial_profile(tmp_path / "radial_profile_r30__single_photon", formats=())
        collector.context("energy", "single_photon")
        plot.plot_energy("energy", "single_photon",
                         [(GEANT4,) + energy_sample(0.98, 0.17, 1),
                          (MODEL,) + energy_sample(0.97, 0.19, 2),
                          (OTHER,) + energy_sample(0.99, 0.18, 3)],
                         GEANT4, "l1", str(tmp_path), "", [])
    return collector, dashboard.render(collector, folder="/data/observables")


def test_the_page_holds_each_figure_with_its_models_marked(tmp_path):
    collector, page = page_of(tmp_path)
    assert [figure["name"] for figure in collector.figures] == [
        "radial_profile_r30__single_photon", "energy_linearity__single_photon",
        "energy_resolution__single_photon"]
    assert list(collector.simulators) == ["model", "other.v2"]
    # A card for each figure, saying which models it has
    cards = re.findall(r'<figure class="fsn-card"([^>]*)>', page)
    assert len(cards) == 3
    assert all('data-scenario="single_photon"' in card and 'data-models="0 1"' in card
               for card in cards)
    assert re.findall(r'<span class="fsn-title">([^<]*)</span>', page) == [
        "radial profile · r30", "energy · linearity", "energy · resolution"]
    # each model's parts with its classes: fsn-s0 is the model, fsn-s1 the other
    for k in (0, 1):
        assert len(re.findall(r'class="fsn-d fsn-s%d"' % k, page)) >= 3 * 2
        assert len(re.findall(r'class="fsn-k fsn-s%d"' % k, page)) == 3 * 2
        assert len(re.findall(r'class="fsn-t fsn-s%d"' % k, page)) == 3
        assert ".fsn-card.fsn-h%d .fsn-d.fsn-s%d{display:none}" % (k, k) in page
    # The menu: the models in order, with their samples, and Geant4 as the reference
    assert json.loads(re.search(r'<div class="fsn" data-models="([^"]*)"', page).group(1)
                      .replace("&quot;", '"')) == ["model", "other.v2"]
    assert re.findall(r'<label class="fsn-model" title="([^"]*)"', page) == ["model", "other.v2"]
    assert 'stroke="#2a9a3a" stroke-width="2" stroke-dasharray="6 2 2 2"' in page
    assert re.search(r'fsn-reference" title="geant4".*?<span class="fsn-label">Geant4</span>',
                     page)
    assert '<option value="energy">energy</option>' in page
    assert "/data/observables" in page


def test_the_page_keeps_what_the_figures_share_once_and_names_nothing_twice(tmp_path):
    _, page = page_of(tmp_path)
    ids = re.findall(r'\sid="([^"]+)"', page)
    assert ids and len(ids) == len(set(ids))
    # every reference is to something in the page, kept once for every figure
    references = set(re.findall(r'href="#([^"]+)"', page) + re.findall(r'url\(#([^)]+)\)', page))
    assert references and references <= set(ids)
    shared = page[page.index('class="fsn-defs"'):]
    assert set(re.findall(r'\sid="([^"]+)"', shared[:shared.index("</svg>")])) == set(ids)
    # The marks became classes, and matplotlib's own ids and styles are gone
    assert "fsn.data." not in page and "patch_" not in page
    assert "<metadata" not in page and '<style type="text/css">' not in page
    assert "width=\"374.4pt\"" not in page and 'class="fsn-plot"' in page
    # and the paths are short
    for d in re.findall(r'\sd="([^"]*)"', page):
        assert not re.search(r"\d\.\d{3}", d) and "\n" not in d


def test_the_pages_styles_and_script_touch_only_its_own_element():
    """So that it can sit in a notebook's output."""
    for selectors in re.findall(r"([^{}]+)\{[^{}]*\}", dashboard.STYLE):
        for selector in selectors.split(","):
            assert selector.strip().startswith(".fsn"), selector
    script = dashboard.SCRIPT
    assert "document.body" in script  # only to tell whether it is the whole page
    assert "root.querySelector" in script and "document.querySelector(" not in script


def measurement(folder, name, label, color, linestyle, scale, file_name=None):
    """A measurement of a radial profile, as observables_from_reconstruction.sh
    writes it, of 100 events falling off as exp(-r/scale)."""
    edges = measure.bin_edges((0.0, 30.0, 30))
    centres = (edges[1:] + edges[:-1]) / 2
    mean = np.exp(-centres / scale)
    key = "radial_profile__single_photon__r30__"
    np.savez(os.path.join(str(folder), (file_name or name) + ".npz"),
             meta__simulator=np.array(name), meta__label=np.array(label),
             meta__color=np.array(color), meta__linestyle=np.array(json.dumps(linestyle)),
             single_photon__n_events=np.array([100]), **{
                 key + "edges": edges, key + "sum": (100 * mean)[None, :],
                 key + "sum_sq": (101 * mean ** 2)[None, :]})


def test_plotting_writes_the_dashboard_unless_told_not_to(tmp_path):
    for name, label, color, scale in (("geant4", "Geant4", "#8c8c8c", 5.0),
                                      ("alpha", "Alpha", "#e66100", 5.5),
                                      ("beta", "Beta", "#1f6fb4", 6.0)):
        measurement(tmp_path, name, label, color, "-", scale)
        measurement(tmp_path, name, label, color, "-", scale, file_name=name + "_n50")
    assert plot.main(["--dir", str(tmp_path), "--format", "png", "--no-closure"]) == 0
    plots = tmp_path / "plots"
    page = (plots / "dashboard.html").read_text(encoding="utf-8")
    assert page.count('<figure class="fsn-card"') == 1
    assert '<label class="fsn-model" title="alpha"' in page and 'title="beta"' in page
    assert not [name for name in os.listdir(str(plots)) if name.endswith(".svg")]

    # The first 50 events, with the SVG files too
    assert plot.main(["--dir", str(tmp_path), "--events", "50", "--format", "svg",
                      "--no-closure"]) == 0
    assert os.path.isfile(str(plots / "dashboard_n50.html"))
    assert os.path.isfile(str(plots / "radial_profile_r30__single_photon_n50.svg"))

    # and none at all
    out = tmp_path / "elsewhere"
    assert plot.main(["--dir", str(tmp_path), "--out", str(out), "--format", "png",
                      "--no-closure", "--no-dashboard"]) == 0
    assert sorted(os.listdir(str(out))) == ["fits.csv", "metrics.csv",
                                            "radial_profile_r30__single_photon.png"]


def test_paths_are_written_to_a_hundredth_of_a_point():
    assert dashboard.compact("M 54.059609 206.871987 \nL 357.35625 -0.001 \nz\n") \
        == "M54.06 206.87L357.36 0z"
    assert dashboard.compact("M 0 3 \nC 0.795609 3 1.55874 2.683901 2.12132 2.12132 \n") \
        == "M0 3C0.8 3 1.56 2.68 2.12 2.12"


def test_line_styles_become_dash_patterns_as_matplotlib_scales_them():
    from matplotlib import rcParams
    assert dashboard.dashes("-", 2) == dashboard.dashes("solid", 2) == ""
    dashed = rcParams["lines.dashed_pattern"]
    assert dashboard.dashes("--", 2) == " ".join("%g" % round(2 * v, 2) for v in dashed)
    assert dashboard.dashes((0, (10, 3, 2, 3)), 2) == "20 6 4 6"
    assert dashboard.dashes([0, [5, 1]], 1) == "5 1"


def test_marks_are_read_back_whatever_the_simulators_name():
    assert dashboard.parse_mark("fsn.data.caloclouds3.12") == ("data", "caloclouds3")
    assert dashboard.parse_mark("fsn.key.other.v2.3") == ("key", "other.v2")
    assert dashboard.parse_mark("fsn.target.with%20space.1") == ("target", "with space")
    for ident in ("patch_3", "fsn.other.model.1", "fsn.data"):
        assert dashboard.parse_mark(ident) is None


def test_a_notebook_shows_the_dashboard(tmp_path):
    pytest.importorskip("IPython")
    where = tmp_path / "dashboard.html"
    where.write_text('<div class="fsn">figures</div>', encoding="utf-8")
    assert dashboard.show(str(where)).data == '<div class="fsn">figures</div>'
