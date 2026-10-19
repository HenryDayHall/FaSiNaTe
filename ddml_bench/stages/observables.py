"""Stage 5: measure the observables on the reconstructed events, one task per
simulator, which gathers the numbers every plot needs into one .npz file
(ddml_bench/measure.py has the observables and the keys of the files).

A simulator is measured once every input of every scenario the observables
use is reconstructed, and measured again when its .npz file is out of date:
when the observables' settings change, or its reconstructions do.
"""

import os
import sys

from .. import measure
from ..layout import Layout
from .base import Stage, StageError, missing_modules


def say(message):
    print("observables: " + message, file=sys.stderr)


class Observables(Stage):
    name = "observables"
    gathers = True

    def add_tasks(self, graph):
        plan = graph.plan
        settings = plan["observables"]
        for simulator in [plan["reference"]] + plan["models"]:
            name = simulator["name"]
            # The reconstructions of every input of the scenarios measured
            files, needs = {}, []
            for scenario in settings["scenarios"]:
                tasks = [task for task in graph.reconstructions.get((name, scenario), [])
                         if task["input_file"] is not None]
                files[scenario] = [task["output"] for task in tasks]
                needs += [task["key"] for task in tasks]
            style = dict((key, simulator[key])
                         for key in ("name", "label", "color", "linestyle", "compact_file"))
            task = graph.add(self.name, graph.layout.observables(name), needs, simulator=name,
                             name=name, inputs=[p for s in files for p in files[s]],
                             params={"observables": settings["observables"], "files": files,
                                     "simulator": style, "data_dir": plan["data_dir"]})
            graph.measuring.append(task)

    def current(self, task):
        files = task["params"]["files"]
        paths = [path for scenario in sorted(files) for path in files[scenario]]
        absent = [path for path in paths if not os.path.isfile(path)]
        if absent:
            return False, "%d of its %d reconstructions aren't there" % (len(absent), len(paths))
        return measure.is_up_to_date(task["output"],
                                     measure.wanted(task["params"]["observables"]), files,
                                     Layout(task["params"]["data_dir"]).relative)

    def check_environment(self, tasks):
        return missing_modules(("numpy", "uproot", "awkward"), "measuring the observables")

    def run(self, task):
        params = task["params"]
        say("%s: measuring %s" % (task["simulator"], task["output"]))
        try:
            measure.compute_simulator(params["observables"], params["simulator"],
                                      params["files"], measure.wanted(params["observables"]),
                                      task["output"], Layout(params["data_dir"]).relative)
        except (measure.ObservableError, OSError) as error:
            raise StageError("measuring %s failed: %s" % (task["simulator"], error))
        except Exception as error:  # e.g. a file uproot can't read
            raise StageError("measuring %s failed reading its reconstructed files: %s: %s"
                             % (task["simulator"], type(error).__name__, error))
        say("%s: wrote %s" % (task["simulator"], task["output"]))

    def describe(self, task):
        return "the measurements of %s" % task["simulator"]

    def label(self, task):
        return task["simulator"]

    def job_name(self, task):
        return "obs-" + task["simulator"]

    def log_name(self, task):
        return [self.name, task["simulator"]]
