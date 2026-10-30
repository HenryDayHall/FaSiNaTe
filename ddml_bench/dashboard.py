"""The dashboard: every figure of a plot run in one page, dashboard.html next
to them (dashboard_n<N>.html for the first N events), with a menu to show or
hide each model in every figure at once, and each model's legend entry in a
figure to show or hide it in that figure alone.

plot.py draws each figure as it always does, and as SVG for this page too,
each part of it marked with the simulator it draws (plot.mark(): a group with
the id <MARK>.<role>.<simulator>.<n>, the role DATA for its curves, bands,
points, fits and bars, KEY for its legend entry, and TARGET for an invisible
rectangle over that entry, to click). The reference (Geant4) is never marked,
so it is always there to compare with.

The page is one self-contained file, with no styles, scripts or fonts from
elsewhere, so it opens the same way from a disk, a web server (e.g. python3
-m http.server through ssh -L), a GitHub artifact uploaded unzipped, or a
Jupyter notebook (show()). Its styles and script touch only the dashboard's
own element, so it can sit in a notebook's output. What the figures share
(clip paths, markers, the outlines of letters) is kept once, and the paths
are written to a hundredth of a point (a point is 0.35 mm).

The axes are as drawn, with every model: hiding a model doesn't rescale them.

  from ddml_bench.dashboard import show
  show()                       # in a notebook: the dashboard of the plots
"""

import datetime
import html
import json
import os
import re
import xml.etree.ElementTree as ET
from urllib.parse import unquote

from .layout import OBSERVABLES_DIR, PLOTS_DIR, SEP

NAME = "dashboard"
# The marks of plot.mark(): <MARK>.<role>.<simulator>.<n>
MARK = "fsn"
DATA, KEY, TARGET = "data", "key", "target"
# The classes the page gives what is marked: of a simulator's DATA, which its
# menu hides, its KEY, which it dims, and its TARGET, which is clicked
CLASSES = {DATA: "fsn-d", KEY: "fsn-k", TARGET: "fsn-t"}
# How many figures the page can show in a row: as many as fit at 420 px or
# more, or a number of them (the CSS grid's columns)
FIT = "repeat(auto-fill,minmax(min(420px,100%),1fr))"
PER_ROW = (("fit", "As many as fit", FIT),) + tuple(
    (str(n), str(n), "repeat(%d,minmax(0,1fr))" % n) for n in (1, 2, 3, 4))
# The width of the lines in the menu's samples of each model's line style
SAMPLE_WIDTH = 2.0

SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", XLINK_NS)
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
_COMMAND = re.compile(r" ?([MmLlHhVvCcSsQqTtAaZz]) ?")


def path(out, suffix=""):
    """Where the dashboard of the figures in out goes."""
    return os.path.join(out, "%s%s.html" % (NAME, suffix))


class Collector:
    """What plot.py draws while collecting (plot.collecting()): figures, in
    order, dicts of name (the figure's file, without its format), section (its
    folder relative to root, the plots folder: "" or closure/<model>),
    observable, scenario and svg; simulators, those marked in them,
    {name: (label, colour, line style)} in order; and references, of each
    section the simulator compared with, {section: (name, label) or None}."""

    def __init__(self, root):
        self.root = root
        self.figures = []
        self.simulators = {}
        self.references = {}
        self._section = ""
        self._context = (None, None)

    def section(self, folder, reference):
        """The figures from now on are in folder, compared with reference (a
        simulator, or None)."""
        section = os.path.relpath(folder, self.root)
        self._section = "" if section == os.curdir else section.replace(os.sep, "/")
        self.references[self._section] = (None if reference is None
                                          else (reference.name, reference.label))
        self._context = (None, None)

    def context(self, observable, scenario):
        """The figures from now on are of this observable and scenario."""
        self._context = (observable, scenario)

    def simulator(self, simulator):
        if simulator.name not in self.simulators:
            self.simulators[simulator.name] = (simulator.label, simulator.color,
                                               simulator.linestyle)

    def add(self, base, svg):
        """A figure, base its file without the format, svg its SVG (bytes)."""
        observable, scenario = self._context
        self.figures.append({"name": os.path.basename(base), "section": self._section,
                             "observable": observable, "scenario": scenario, "svg": svg})


def write(collector, where, folder=None, events=None, reference_colors=("#d9d9d9", "#8c8c8c")):
    """Write the dashboard of the collected figures to where, through a
    temporary file so that no one reads half of it; folder is that of the
    measurements, and events the number of events of each, if limited.
    reference_colors are the fill and edge of the reference's histograms."""
    page = render(collector, folder=folder, events=events, reference_colors=reference_colors)
    temporary = os.path.join(os.path.dirname(where) or os.curdir,
                             ".%s.part" % os.path.basename(where))
    with open(temporary, "w", encoding="utf-8") as stream:
        stream.write(page)
    os.replace(temporary, where)
    return where


def show(where=None, events=None):
    """The dashboard, to show in a Jupyter notebook (as the last line of a
    cell): the file where, or that of the plots in the configuration's data
    folder (of the first events, if given). Its script runs where the
    notebook's output is trusted, as that of a cell you ran yourself is: after
    opening a notebook again, run the cell again."""
    from IPython.display import HTML
    if where is None:
        from .config import load_config
        data_dir = load_config()["paths"]["data_dir"]
        where = path(os.path.join(data_dir, OBSERVABLES_DIR, PLOTS_DIR),
                     "" if events is None else "_n%d" % events)
    with open(where, encoding="utf-8") as stream:
        return HTML(stream.read())


# ---------------------------------------------------------------------------
# The figures' SVG, as the page holds it


def _tag(name):
    return "{%s}%s" % (SVG_NS, name)


def compact(d):
    """A path's data with its numbers to two decimal places and no spaces it
    doesn't need: "M 54.059609 206.871987 \\nL 357.35625 206.871987 \\n" is
    "M54.06 206.87L357.36 206.87"."""
    def short(match):
        text = "%.2f" % float(match.group())
        text = text.rstrip("0").rstrip(".")
        return "0" if text in ("", "-0") else text
    d = re.sub(r"\s+", " ", _NUMBER.sub(short, d)).strip()
    return _COMMAND.sub(r"\1", d)


def parse_mark(ident):
    """(role, simulator) of a mark's id, <MARK>.<role>.<simulator>.<n>; None
    for any other id."""
    parts = ident.split(".", 2)
    if len(parts) < 3 or parts[0] != MARK or parts[1] not in CLASSES:
        return None
    name, _, number = parts[2].rpartition(".")
    if not name or not number.isdigit():
        name = parts[2]
    return parts[1], unquote(name)


def figure_svg(svg, shared, index, labels, title):
    """The SVG of a figure for the page, and the indices (in index, {name:
    index}) of the simulators whose parts are marked in it. What it defines
    for its parts to share goes to shared, {id: element}, once for every
    figure: matplotlib names it after what it is (with plot.svg_of()'s hash
    salt), so that one name is one thing. The other ids go, and each mark's
    becomes classes: fsn-d, fsn-k or fsn-t (CLASSES), and fsn-s<index> of its
    simulator. labels, {name: label}, name the targets' tooltips."""
    root = ET.fromstring(svg)
    parents = dict((child, parent) for parent in root.iter() for child in parent)
    for definitions in [element for element in root.iter(_tag("defs"))]:
        for element in definitions:
            ident = element.get("id")
            # (matplotlib's style element is in the page's styles instead)
            if ident and element.tag != _tag("style") and ident not in shared:
                for part in element.iter(_tag("path")):
                    part.set("d", compact(part.get("d", "")))
                shared[ident] = element
        parents[definitions].remove(definitions)
    for element in [e for e in root if e.tag == _tag("metadata")]:
        root.remove(element)

    present = set()
    for element in list(root.iter()):
        ident = element.attrib.pop("id", None)
        if element.tag == _tag("path") and "d" in element.attrib:
            element.set("d", compact(element.get("d")))
        found = parse_mark(ident) if ident else None
        if found is None or found[1] not in index:
            continue
        role, name = found
        k = index[name]
        present.add(k)
        element.set("class", "%s fsn-s%d" % (CLASSES[role], k))
        if role == TARGET:
            text = "Show or hide %s in this figure" % labels.get(name, name)
            element.set("data-s", str(k))
            element.set("tabindex", "0")
            element.set("role", "button")
            element.set("aria-label", text)
            tooltip = ET.Element(_tag("title"))
            tooltip.text = text
            element.insert(0, tooltip)
    for attribute in ("width", "height", "version"):
        root.attrib.pop(attribute, None)
    root.set("class", "fsn-plot")
    # (a group, not an image, whose parts screen readers would then skip)
    root.set("role", "group")
    root.set("aria-label", title)
    return ET.tostring(root, encoding="unicode"), sorted(present)


def dashes(linestyle, width=SAMPLE_WIDTH):
    """The SVG stroke-dasharray of a matplotlib line style ("-", "--", ":",
    "-.", their names, or (offset, (on, off, ...))) at a line width, scaled as
    matplotlib scales them; "" for a solid line."""
    from matplotlib import rcParams
    names = {"--": "dashed", "dashed": "dashed", ":": "dotted", "dotted": "dotted",
             "-.": "dashdot", "dashdot": "dashdot"}
    if isinstance(linestyle, str):
        if linestyle not in names:
            return ""
        pattern = rcParams["lines.%s_pattern" % names[linestyle]]
    else:
        pattern = linestyle[1] if len(linestyle) > 1 else None
        if not pattern:
            return ""
    scale = width if rcParams["lines.scale_dashes"] else 1.0
    return " ".join("%g" % round(float(length) * scale, 2) for length in pattern)


# ---------------------------------------------------------------------------
# The page


def _attribute(value):
    return html.escape(str(value), quote=True)


def figure_title(figure):
    """A figure's title: its observable, and what of it (from its name,
    <observable>_<profile>__<scenario>), e.g. radial profile · r30."""
    name, observable = figure["name"], figure["observable"]
    head = name.rsplit(SEP, 1)[0] if SEP in name else name
    if observable and head.startswith(observable + "_"):
        return "%s · %s" % (observable.replace("_", " "),
                            head[len(observable) + 1:].replace("_", " "))
    return head.replace("_", " ")


def render(collector, folder=None, events=None, reference_colors=("#d9d9d9", "#8c8c8c")):
    """The page of the collected figures (see write())."""
    names = list(collector.simulators)
    index = dict((name, k) for k, name in enumerate(names))
    labels = dict((name, info[0]) for name, info in collector.simulators.items())
    shared = {}

    # The sections: the figures of each scenario, then each closure test's
    sections, order = {}, []
    scenarios, observables = [], []
    for figure in collector.figures:
        closure = figure["section"] != ""
        key = (figure["section"], None) if closure else ("", figure["scenario"])
        if key not in sections:
            sections[key] = []
            order.append(key)
        sections[key].append(figure)
        for value, seen in ((figure["scenario"], scenarios), (figure["observable"], observables)):
            if value and value not in seen:
                seen.append(value)
    order.sort(key=lambda key: key[0] != "")

    parts = []
    for key in order:
        section, scenario = key
        if section:
            reference = collector.references.get(section)
            heading = "Closure test: %s against %s" % (
                section.rsplit("/", 1)[-1], reference[1] if reference else "its twin")
        else:
            heading = scenario or "Figures"
        cards = []
        for figure in sections[key]:
            title = figure_title(figure)
            svg, present = figure_svg(figure["svg"], shared, index, labels, title)
            cards.append(
                '<figure class="fsn-card" data-scenario="%s" data-observable="%s" '
                'data-closure="%d" data-models="%s"><figcaption><span class="fsn-title">%s</span>'
                '<span class="fsn-file" title="%s">%s</span><button type="button" '
                'class="fsn-reset" title="Show the models as the menu does" hidden>Reset</button>'
                '</figcaption>%s</figure>' % (
                    _attribute(figure["scenario"] or ""), _attribute(figure["observable"] or ""),
                    1 if section else 0, " ".join(str(k) for k in present),
                    html.escape(title), _attribute(figure["name"]), html.escape(figure["name"]),
                    svg))
        parts.append('<section class="fsn-section"><h2>%s</h2><div class="fsn-grid">%s</div>'
                     '</section>' % (html.escape(heading), "".join(cards)))

    definitions = ET.Element(_tag("svg"), {"class": "fsn-defs", "aria-hidden": "true",
                                           "focusable": "false", "width": "0", "height": "0"})
    holder = ET.SubElement(definitions, _tag("defs"))
    for element in shared.values():
        holder.append(element)

    models = [{"name": name, "label": info[0], "color": info[1], "dash": dashes(info[2])}
              for name, info in collector.simulators.items()]
    rules = "".join(".fsn-card.fsn-h%d .fsn-d.fsn-s%d{display:none}"
                    ".fsn-card.fsn-h%d .fsn-k.fsn-s%d{opacity:.28}" % ((k,) * 4)
                    for k in range(len(models)))
    return PAGE % {
        "title": "FaSiNaTe plots",
        "models_json": _attribute(json.dumps([m["name"] for m in models])),
        "style": STYLE + rules,
        "definitions": ET.tostring(definitions, encoding="unicode"),
        "side": side(models, collector.references.get(""), reference_colors, scenarios,
                     observables, any(section for section, _ in order), folder, events),
        "sections": "".join(parts),
        "script": SCRIPT,
    }


def sample(color, dash, fill=None):
    """A small picture of a model's line in the menu (or of the reference's
    filled histogram, with fill)."""
    if fill is not None:
        return ('<svg class="fsn-sample" viewBox="0 0 26 12" aria-hidden="true"><rect x="1" '
                'y="2" width="24" height="8" fill="%s" stroke="%s" stroke-width="1"/></svg>'
                % (_attribute(fill), _attribute(color)))
    return ('<svg class="fsn-sample" viewBox="0 0 26 12" aria-hidden="true"><line x1="1" y1="6" '
            'x2="25" y2="6" stroke="%s" stroke-width="%g"%s/></svg>' % (
                _attribute(color), SAMPLE_WIDTH,
                ' stroke-dasharray="%s"' % _attribute(dash) if dash else ""))


def side(models, reference, reference_colors, scenarios, observables, closures, folder, events):
    """The menu: the models, and which figures to show."""
    rows = []
    if reference is not None:
        rows.append('<div class="fsn-model fsn-reference" title="%s">%s<span class="fsn-label">'
                    '%s</span><span class="fsn-note">reference</span></div>' % (
                        _attribute(reference[0]), sample(reference_colors[1], "",
                                                         fill=reference_colors[0]),
                        html.escape(reference[1])))
    for k, model in enumerate(models):
        rows.append('<label class="fsn-model" title="%s"><input type="checkbox" checked '
                    'data-k="%d">%s<span class="fsn-label">%s</span><button type="button" '
                    'class="fsn-only" data-k="%d" title="Show %s alone">only</button></label>' % (
                        _attribute(model["name"]), k, sample(model["color"], model["dash"]),
                        html.escape(model["label"]), k, _attribute(model["label"])))

    def choices(values, kind):
        options = "".join('<option value="%s">%s</option>' % (_attribute(v), html.escape(v))
                          for v in values)
        return ('<label class="fsn-field"><span>%s</span><select class="fsn-%s"><option '
                'value="">All</option>%s</select></label>' % (kind.capitalize(), kind, options))

    per_row = "".join('<option value="%s" data-columns="%s"%s>%s</option>' % (
        key, columns, " selected" if columns == FIT else "", label)
        for key, label, columns in PER_ROW)
    meta = [html.escape(folder)] if folder else []
    if events is not None:
        meta.append("the first %d events" % events)
    meta.append("made %s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    return (
        '<aside class="fsn-side"><h1>FaSiNaTe plots</h1><p class="fsn-meta">%s</p>'
        '<p class="fsn-off">The menu needs scripts, which are off here: in JupyterLab\'s viewer, '
        'click <b>Trust HTML</b>; in a notebook, run the cell again.</p>'
        '<div class="fsn-group"><div class="fsn-head"><span>Models</span><span><button '
        'type="button" class="fsn-all">all</button> · <button type="button" '
        'class="fsn-none">none</button></span></div>%s</div>'
        '<div class="fsn-group"><div class="fsn-head"><span>Figures</span><span class="fsn-count">'
        '</span></div>%s%s<label class="fsn-field"><span>Per row</span><select class="fsn-columns">'
        '%s</select></label>%s</div>'
        '<p class="fsn-tip">Untick a model to hide it in every figure, or click its entry in a '
        'figure\'s legend to hide or show it in that figure alone. The axes stay as drawn with '
        'every model.</p></aside>' % (
            "<br>".join(meta), "".join(rows) or '<p class="fsn-tip">No models.</p>',
            choices(scenarios, "scenario"), choices(observables, "observable"), per_row,
            '<label class="fsn-check"><input type="checkbox" class="fsn-closure" checked> '
            'Closure tests</label>' if closures else ""))


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>%(title)s</title>
</head>
<body style="margin:0">
<div class="fsn" data-models="%(models_json)s">
<style>%(style)s</style>
%(definitions)s
%(side)s
<main class="fsn-main">%(sections)s</main>
<script>%(script)s</script>
</div>
</body>
</html>
"""

# Every rule is of the dashboard's own element, .fsn, so that it can sit in a
# notebook's output without changing anything else there
STYLE = """
.fsn{--fsn-ink:#1f1f1f;--fsn-muted:#6b6b6b;--fsn-line:#e2e2de;--fsn-bg:#f4f4f1;
--fsn-card:#fff;--fsn-accent:#1f6fb4;--fsn-columns:FIT;display:grid;
grid-template-columns:250px minmax(0,1fr);min-height:100vh;background:var(--fsn-bg);
color:var(--fsn-ink);text-align:left;
font:14px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif}
.fsn *,.fsn *::before,.fsn *::after{box-sizing:border-box}
.fsn [hidden]{display:none!important}
.fsn button{font:inherit;color:var(--fsn-accent);background:none;border:0;padding:0;
cursor:pointer}
.fsn button:hover{text-decoration:underline}
.fsn-side{position:sticky;top:0;align-self:start;height:100vh;overflow:auto;
padding:18px 16px 24px;background:var(--fsn-card);border-right:1px solid var(--fsn-line)}
.fsn-side h1{font-size:16px;font-weight:650;margin:0 0 2px;letter-spacing:-.01em}
.fsn-meta{margin:0 0 16px;font-size:12px;color:var(--fsn-muted);overflow-wrap:anywhere}
.fsn-off{margin:0 0 16px;padding:8px 10px;font-size:12px;border-radius:6px;
background:#fdf3e1;color:#6b4a00}
.fsn-group{margin:0 0 18px}
.fsn-head{display:flex;justify-content:space-between;align-items:baseline;margin:0 0 6px;
font-size:11px;font-weight:650;letter-spacing:.06em;text-transform:uppercase;
color:var(--fsn-muted)}
.fsn-head button{font-size:12px;font-weight:400;letter-spacing:0;text-transform:none}
.fsn-count{font-weight:400;letter-spacing:0;text-transform:none}
.fsn-model{position:relative;display:flex;align-items:center;gap:8px;margin:0 -6px;
padding:3px 6px;border-radius:6px;cursor:pointer}
.fsn-model:hover{background:var(--fsn-bg)}
.fsn-model input{margin:0;flex:none;accent-color:var(--fsn-accent)}
.fsn-reference{cursor:default;padding-left:27px}
.fsn-reference:hover{background:none}
.fsn-sample{flex:none;width:26px;height:12px}
.fsn-label{flex:1;min-width:0;overflow-wrap:anywhere}
.fsn-note{font-size:12px;color:var(--fsn-muted)}
.fsn-only{position:absolute;right:4px;top:50%;transform:translateY(-50%);padding:0 4px;
font-size:12px;border-radius:4px;background:var(--fsn-bg);opacity:0}
.fsn-model:hover .fsn-only,.fsn-only:focus-visible{opacity:1}
@media (hover:none){.fsn-only{position:static;transform:none;opacity:1}}
.fsn-field{display:block;margin:0 0 8px}
.fsn-field span{display:block;margin:0 0 2px;font-size:12px;color:var(--fsn-muted)}
.fsn-field select{width:100%;font:inherit;padding:4px 6px;color:inherit;
background:var(--fsn-card);border:1px solid var(--fsn-line);border-radius:6px}
.fsn-check{display:flex;align-items:center;gap:8px;font-size:13px;cursor:pointer}
.fsn-check input{margin:0;accent-color:var(--fsn-accent)}
.fsn-tip{margin:0;font-size:12px;color:var(--fsn-muted)}
.fsn-main{min-width:0;padding:18px 22px 40px}
.fsn-section{margin:0 0 28px}
.fsn-section h2{margin:0 0 10px;font-size:15px;font-weight:650}
.fsn-grid{display:grid;gap:14px;grid-template-columns:var(--fsn-columns)}
.fsn-card{display:flex;flex-direction:column;min-width:0;margin:0;padding:10px 10px 6px;
background:var(--fsn-card);border:1px solid var(--fsn-line);border-radius:10px}
.fsn-card figcaption{display:flex;align-items:baseline;gap:8px;margin:0 2px 4px;font-size:13px}
.fsn-title{flex:none;font-weight:600}
.fsn-file{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
font-size:11px;color:var(--fsn-muted)}
.fsn-reset{flex:none;font-size:12px}
.fsn-plot{display:block;width:100%;height:auto}
.fsn-plot *{stroke-linejoin:round;stroke-linecap:butt}
.fsn-t{cursor:pointer;outline:none}
.fsn-t path{pointer-events:all}
.fsn-t:hover path,.fsn-t:focus-visible path{fill:var(--fsn-accent)!important;opacity:.1!important}
.fsn-k{transition:opacity .15s}
.fsn-defs{position:absolute;width:0;height:0;overflow:hidden}
@media (max-width:760px){.fsn{grid-template-columns:1fr}
.fsn-side{position:static;height:auto;border-right:0;border-bottom:1px solid var(--fsn-line)}
.fsn-main{padding:14px 12px 32px}}
""".replace("FIT", FIT)

# The menu's behaviour. Each model's box is the menu's choice for it, shown or
# hidden in every figure; a click on its legend entry in a figure changes that
# figure alone, which the box then shows as neither ticked nor unticked, and
# the figure's Reset puts back. A figure hides a model with the class
# fsn-h<k> (the rules after STYLE). On a page of its own, what the menu hides
# and shows is kept in the address (#hide=a,b&...), to come back to and to
# share.
SCRIPT = r"""
(function () {
  "use strict";
  var me = document.currentScript, root = me && me.parentNode;
  if (!root || !root.classList || !root.classList.contains("fsn")) {
    var all = document.querySelectorAll(".fsn");
    root = all[all.length - 1];
  }
  if (!root || root.getAttribute("data-ready")) return;
  root.setAttribute("data-ready", "1");
  var off = root.querySelector(".fsn-off");
  if (off) off.hidden = true;
  var each = function (list, f) { Array.prototype.forEach.call(list, f); };
  var names = JSON.parse(root.getAttribute("data-models") || "[]");
  var chosen = names.map(function () { return true; });
  var cards = Array.prototype.slice.call(root.querySelectorAll(".fsn-card"));
  var boxes = {};
  each(root.querySelectorAll(".fsn-model input"), function (box) {
    boxes[box.getAttribute("data-k")] = box;
  });
  cards.forEach(function (card) {
    var list = card.getAttribute("data-models");
    card.fsnModels = list ? list.split(" ").map(Number) : [];
  });
  var pick = function (kind) { return root.querySelector(".fsn-" + kind); };
  var scenario = pick("scenario"), observable = pick("observable");
  var columns = pick("columns"), closure = pick("closure"), count = pick("count");
  var standalone = root.parentNode === document.body, restoring = false;

  function hidden(card, k) { return card.classList.contains("fsn-h" + k); }
  function differs(card, k) { return hidden(card, k) === chosen[k]; }
  function hide(card, k, yes) {
    if (card.fsnModels.indexOf(k) < 0) return;
    card.classList.toggle("fsn-h" + k, yes);
    card.querySelector(".fsn-reset").hidden =
      !card.fsnModels.some(function (m) { return differs(card, m); });
  }
  function choose(k, show) {
    chosen[k] = show;
    cards.forEach(function (card) { hide(card, k, !show); });
  }
  function sync() {
    names.forEach(function (name, k) {
      var box = boxes[k];
      if (!box) return;
      box.checked = chosen[k];
      box.indeterminate = cards.some(function (card) {
        return card.fsnModels.indexOf(k) >= 0 && differs(card, k);
      });
    });
    save();
  }
  function filter() {
    var shown = 0;
    cards.forEach(function (card) {
      var show = (!scenario || !scenario.value ||
                  card.getAttribute("data-scenario") === scenario.value) &&
                 (!observable || !observable.value ||
                  card.getAttribute("data-observable") === observable.value) &&
                 (!closure || closure.checked || card.getAttribute("data-closure") !== "1");
      card.hidden = !show;
      if (show) shown += 1;
    });
    each(root.querySelectorAll(".fsn-section"), function (section) {
      section.hidden = !section.querySelector(".fsn-card:not([hidden])");
    });
    if (count) count.textContent = shown + " of " + cards.length;
    var option = columns && columns.options[columns.selectedIndex];
    if (option) root.style.setProperty("--fsn-columns", option.getAttribute("data-columns"));
    save();
  }
  function save() {
    if (!standalone || restoring) return;
    var parts = [], gone = [];
    names.forEach(function (name, k) {
      if (!chosen[k]) gone.push(encodeURIComponent(name));
    });
    if (gone.length) parts.push("hide=" + gone.join(","));
    [["scenario", scenario], ["observable", observable]].forEach(function (pair) {
      if (pair[1] && pair[1].value) parts.push(pair[0] + "=" + encodeURIComponent(pair[1].value));
    });
    if (columns && !columns.options[columns.selectedIndex].defaultSelected) {
      parts.push("columns=" + columns.value);
    }
    if (closure && !closure.checked) parts.push("closure=0");
    try {
      history.replaceState(null, "", parts.length ? "#" + parts.join("&")
                                                   : location.pathname + location.search);
    } catch (error) { /* an address that can't change, e.g. in a sandbox */ }
  }
  function load() {
    if (!standalone || !location.hash) return;
    location.hash.slice(1).split("&").forEach(function (part) {
      var at = part.indexOf("="), key = part.slice(0, at), value = part.slice(at + 1);
      if (at < 0) return;
      if (key === "hide") {
        value.split(",").forEach(function (name) {
          var k = names.indexOf(decodeURIComponent(name));
          if (k >= 0) choose(k, false);
        });
      } else if (key === "closure" && closure) {
        closure.checked = value !== "0";
      } else {
        var select = {scenario: scenario, observable: observable, columns: columns}[key];
        value = decodeURIComponent(value);
        if (select) each(select.options, function (option) {
          if (option.value === value) select.value = value;
        });
      }
    });
  }

  each(root.querySelectorAll(".fsn-model input"), function (box) {
    box.addEventListener("change", function () {
      choose(Number(box.getAttribute("data-k")), box.checked);
      sync();
    });
  });
  each(root.querySelectorAll(".fsn-only"), function (button) {
    button.addEventListener("click", function (event) {
      event.preventDefault();
      var only = Number(button.getAttribute("data-k"));
      names.forEach(function (name, k) { choose(k, k === only); });
      sync();
    });
  });
  [["all", true], ["none", false]].forEach(function (pair) {
    var button = pick(pair[0]);
    if (button) button.addEventListener("click", function () {
      names.forEach(function (name, k) { choose(k, pair[1]); });
      sync();
    });
  });
  function toggle(target) {
    var card = target.closest(".fsn-card"), k = Number(target.getAttribute("data-s"));
    if (!card) return;
    hide(card, k, !hidden(card, k));
    sync();
  }
  root.addEventListener("click", function (event) {
    var target = event.target.closest && event.target.closest(".fsn-t");
    if (target) { toggle(target); return; }
    var reset = event.target.closest && event.target.closest(".fsn-reset");
    if (reset) {
      var card = reset.closest(".fsn-card");
      card.fsnModels.forEach(function (k) { hide(card, k, !chosen[k]); });
      sync();
    }
  });
  root.addEventListener("keydown", function (event) {
    if (event.key !== "Enter" && event.key !== " ") return;
    var target = event.target.closest && event.target.closest(".fsn-t");
    if (target) { event.preventDefault(); toggle(target); }
  });
  [scenario, observable, columns, closure].forEach(function (control) {
    if (control) control.addEventListener("change", filter);
  });

  restoring = true;
  load();
  restoring = false;
  sync();
  filter();
})();
"""
