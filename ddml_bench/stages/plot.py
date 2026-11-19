"""Stage 6: plot every measurement against Geant4, and write metrics.csv; one
task, of whatever is measured (ddml_bench/plot.py draws the figures, and the
Pareto fronts against the timing benchmark's table, if there is one: a run
fetches the published one before, where there's none, ddml_bench/fetch.py)."""

import os

from ..layout import measurement_files, timing_summary_of
from .base import Stage, StageError, missing_modules


class Plot(Stage):
    name = "plot"
    gathers = True
    takes_what_there_is = True

    def add_tasks(self, graph):
        layout = graph.layout
        graph.add(self.name, layout.metrics, [task["key"] for task in graph.measuring],
                  name="plots", inputs=[task["output"] for task in graph.measuring],
                  params={"dir": layout.observables_dir, "events": graph.plan["events"],
                          "suffix": layout.suffix,
                          # where a dry run would fetch the timing table from
                          "timing_to_fetch": graph.plan.get("timing_to_fetch")})

    def measured(self, task):
        return measurement_files(task["params"]["dir"], task["params"]["suffix"])

    def current(self, task):
        found = self.measured(task)
        if not found:
            return False, "there are no measurements to plot"
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
        return None if self.measured(task) else "there is nothing measured to plot yet"

    def check_environment(self, tasks):
        return missing_modules(("numpy", "matplotlib"), "plotting")

    def run(self, task):
        from .. import plot
        args = ["--dir", task["params"]["dir"]]
        if task["params"]["events"] is not None:
            args += ["--events", str(task["params"]["events"])]
        if plot.main(args) != 0:
            raise StageError("plotting failed (see above)")

    def describe(self, task):
        return "the plots"

    def label(self, task):
        return "every .npz file"
