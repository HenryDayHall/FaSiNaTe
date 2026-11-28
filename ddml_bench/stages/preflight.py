"""Stage 0: check each model in models/ before it is simulated, one task per
model (ddml_bench/preflight.py has the checks), which every simulation of the
model waits for. ./preflight_models.sh runs it alone, and ./simulate_models.sh
runs it first.

A model's task is done while preflight_output/<model>/passed_<digest>.json is
there: it passed as it is now (ddml_bench/preflight.py's digest(): its
description, its own code, its weights, the preflight settings and the
software pinned). When any of those changes the record has another name, so
the task is to do again, but what was simulated before stays done, as a
simulation that is there always does (ddml_bench/graph.py): only what is still
to be simulated waits for it to pass again. Checking a model again (--force)
doesn't make its simulations again either (checks).

It is intermediate: in a run with the models stage (which always brings this
stage with it), a model is checked only if something of it is still to be
simulated, so a data folder with every simulator's measurements fetched
checks no model but one added to it. Run on its own (./preflight_models.sh),
it checks every model that hasn't passed as it is now.
"""

import os
import shutil
import sys

from .. import preflight
from ..layout import CUSTOM_SCENARIO
from .base import Stage, StageError, compact_problems, missing_modules


class Preflight(Stage):
    name = "preflight"
    intermediate = True
    checks = True

    def add_tasks(self, graph):
        layout = graph.layout
        plan = graph.plan
        settings = plan["preflight"]
        pins = plan["published"]["pins"]
        scenarios = [group["scenario"] for group in graph.groups]
        if plan["custom_inputs"]:
            scenarios.append(CUSTOM_SCENARIO)
        graph.preflight = {}
        for model in plan["models"]:
            digest = preflight.digest(model, settings, pins)
            task = graph.add(self.name, layout.preflight_passed(model["name"], digest), [],
                             simulator=model["name"], name=model["name"],
                             params={"model": model, "settings": settings,
                                     "folder": layout.preflight_model_dir(model["name"]),
                                     "slurm_log_dir": (plan.get("slurm") or {}).get("log_dir")})
            # (It serves every scenario, so --scenario doesn't leave it out)
            task["scenarios"] = scenarios
            graph.preflight[model["name"]] = task["key"]

    def check_environment(self, tasks):
        """What every preflight needs here: the DDML build, ddsim, the
        detectors, and the Python modules that write the shots, read their
        simulation and draw it."""
        from .simulate import _ddml_problems
        problems = _ddml_problems()
        if not shutil.which("ddsim"):
            problems.append("ddsim isn't on PATH")
        problems += compact_problems(sorted({t["params"]["model"]["compact_file"]
                                             for t in tasks}), "the preflight")
        problems += missing_modules(("yaml", "pyLCIO", "podio", "particle", "numpy",
                                     "matplotlib"), "the preflight")
        return problems

    def run(self, task):
        params = task["params"]
        model = params["model"]
        context = dict(params.get("context") or {})
        if os.environ.get("SLURM_JOB_ID"):
            context.update(slurm_job=os.environ["SLURM_JOB_ID"],
                           slurm_log_dir=params.get("slurm_log_dir"))
        if os.environ.get("GITHUB_RUN_ID"):
            context.setdefault("github_run", os.environ["GITHUB_RUN_ID"])
        record = preflight.run_model(model, params["settings"], params["folder"],
                                     task["output"], model, context)
        summary = os.path.join(params["folder"], preflight.SUMMARY)
        print("preflight:   summary: %s" % summary)
        sys.stdout.flush()
        if record["status"] != preflight.PASSED:
            step = record["steps"][record["failed_step"] - 1]
            raise StageError("the preflight of %s failed at step %d, %s: %s\n  summary: %s\n  "
                             "full log: %s" % (model["name"], step["number"], step["title"],
                                               step["details"][0] if step["details"] else "?",
                                               summary,
                                               os.path.join(params["folder"], preflight.LOG)))

    def describe(self, task):
        return "the preflight of %s" % task["simulator"]

    def label(self, task):
        return task["simulator"]

    def job_name(self, task):
        return task["simulator"]

    def log_name(self, task):
        return [self.name, task["simulator"]]
