"""Stage 6: plot every measurement against Geant4, and write metrics.csv; one
task, of whatever is measured (ddml_bench/plot.py draws the figures, and the
Pareto fronts against the timing benchmark's table, if there is one: a run
fetches the published one before, where there's none, ddml_bench/fetch.py).

In a low-cost run (ddml_bench/low_cost.py), the one task draws the models'
low-cost sample against the full samples of every other simulator measured,
in a folder of its own (layout.low_cost_plots()), leaving the full sample's
plots as they are."""

import os

from ..layout import measurement_files, timing_summary_of
from .base import Stage, StageError, missing_modules


def compared(folder, suffix, samples):
    """[(simulator, .npz file)] that a low-cost plot draws: the samples'
    measurements of the sample (suffix), and the full ones of every other
    simulator measured; those of a sample not measured yet are left out."""
    found = [(name, path) for name, path in measurement_files(folder, suffix)
             if name in samples]
    return found + [(name, path) for name, path in measurement_files(folder)
                    if name not in samples]


class Plot(Stage):
    name = "plot"
    gathers = True
    takes_what_there_is = True

    def add_tasks(self, graph):
        layout = graph.layout
        plan = graph.plan
        if plan.get("low_cost"):
            # The models' sample, against the full samples of the rest
            samples = list(plan["only_simulators"])
            measuring = [task for task in graph.measuring if task["simulator"] in samples]
            out = layout.low_cost_plots_dir
            graph.add(self.name, os.path.join(out, "metrics.csv"),
                      [task["key"] for task in measuring], name="low-cost plots",
                      inputs=[task["output"] for task in measuring],
                      params={"dir": layout.observables_dir, "events": plan["events"],
                              "suffix": layout.suffix, "samples": samples, "out": out,
                              "timing_to_fetch": plan.get("timing_to_fetch")})
            return
        graph.add(self.name, layout.metrics, [task["key"] for task in graph.measuring],
                  name="plots", inputs=[task["output"] for task in graph.measuring],
                  params={"dir": layout.observables_dir, "events": plan["events"],
                          "suffix": layout.suffix,
                          # where a dry run would fetch the timing table from
                          "timing_to_fetch": plan.get("timing_to_fetch")})

    def measured(self, task):
        params = task["params"]
        if params.get("samples"):
            return compared(params["dir"], params["suffix"], params["samples"])
        return measurement_files(params["dir"], params["suffix"])

    def current(self, task):
        found = self.measured(task)
        if not found:
            return False, "there are no measurements to plot"
        samples = task["params"].get("samples") or []
        lacking = [name for name in samples if name not in [simulator for simulator, _ in found]]
        if lacking:
            return False, "%s isn't measured yet" % lacking[0]
        if not os.path.isfile(task["output"]):
            return False, "not plotted yet"
        if max(os.path.getmtime(path) for _, path in found) > os.path.getmtime(task["output"]):
            return False, "measured again since the last plots"
        # The Pareto fronts take the time per event from the timing table
        timing = timing_summary_of(task["params"]["dir"])
        if not os.path.isfile(timing) and task["params"].get("timing_to_fetch"):
            return False, "the timing table is to be fetched from %s" % (
                task["params"]["timing_to_fetch"])
        if os.path.isfile(timing) and os.path.getmtime(timing) > os.path.getmtime(task["output"]):
            # (timed again, or fetched)
            return False, "the timing table is newer than the plots (%s)" % timing
        return True, ""

    def nothing_to_use(self, task):
        found = self.measured(task)
        samples = task["params"].get("samples") or []
        if samples and not any(name in samples for name, _ in found):
            return "%s isn't measured yet" % " or ".join(samples)
        return None if found else "there is nothing measured to plot yet"

    def check_environment(self, tasks):
        return missing_modules(("numpy", "matplotlib"), "plotting")

    def run(self, task):
        from .. import plot
        params = task["params"]
        args = ["--dir", params["dir"]]
        if params.get("samples"):
            args += ["--out", params["out"]]
            for name in params["samples"]:
                args += ["--sample", "%s:%s" % (name, params["suffix"])]
        elif params["events"] is not None:
            args += ["--events", str(params["events"])]
        if plot.main(args) != 0:
            raise StageError("plotting failed (see above)")

    def describe(self, task):
        if task["params"].get("samples"):
            return "the low-cost plots"
        return "the plots"

    def label(self, task):
        if task["params"].get("samples"):
            return "%s, against the full samples" % ", ".join(task["params"]["samples"])
        return "every .npz file"
