"""Stage 1: make the scenarios' inputs, one task per input file.

The inputs made elsewhere (input type external) are tasks too, done when they
are there, so that the inputs of every scenario go through the same graph and
can be fetched (ddml_bench/fetch.py); but nothing makes them here, so forcing
the stage leaves them alone, and a scenario without all of them is left out
of a run (ddml_bench/graph.py). With input.segments, each segment is an input
and a task: the guns make it themselves, and the segments of inputs made
elsewhere are split from them by scripts/split_data.py.
"""

import os
import shutil
import sys
import tempfile

from .. import process
from ..layout import CUSTOM_SCENARIO, input_stem
from ..scenarios import INPUT_TYPES, with_option
from .base import Stage, StageError, missing_modules


def say(message):
    print("generate: " + message, file=sys.stderr)


def made_here(task):
    """Whether this stage makes the task's input (or it is made elsewhere)."""
    return task["params"]["command"] is not None


class Generate(Stage):
    name = "generate"

    def add_tasks(self, graph):
        for group in graph.groups:
            for entry in group["inputs"]:
                path = entry["file"]
                task = graph.by_key.get("%s:%s" % (self.name, path))
                if task is None:
                    params = {"type": group["type"], "command": entry["command"]}
                    if "whole" in entry:
                        # A segment of this input (input.segments)
                        params["whole"] = entry["whole"]
                    task = graph.add(self.name, path, [], scenario=group["scenario"],
                                     name=input_stem(path), input_file=path, params=params)
                    task["scenarios"] = []
                elif task["params"]["command"] != entry["command"]:
                    if None in (task["params"]["command"], entry["command"]):
                        raise StageError("scenarios %s and %s both have %s, but one makes it and "
                                         "the other takes it as made elsewhere"
                                         % (task["scenario"], group["scenario"], path))
                    raise StageError("scenarios %s and %s both make %s, but in different ways; "
                                     "the gun's file names don't include every option (e.g. "
                                     "n_events), so give them different seeds"
                                     % (task["scenario"], group["scenario"], path))
                task["scenarios"].append(group["scenario"])
                graph.inputs.append((group["scenario"], path, task, entry["seeds"]))
        # Inputs given directly are there already, and are simulated as "custom",
        # once, with ddsim's own seed
        for path in graph.plan["custom_inputs"]:
            graph.inputs.append((CUSTOM_SCENARIO, path, None, [None]))

    def nothing_to_use(self, task):
        if not made_here(task) and not os.path.exists(task["output"]):
            whole = task["params"].get("whole")
            if whole:
                return ("it is a segment of %s, which is made outside this repository, so it "
                        "can't be made here: put that there and split it into segments with "
                        "scripts/split_data.py, or put this at %s, or publish it in a record of "
                        "`published`" % (whole, task["output"]))
            return ("it is made outside this repository, so it can't be made here: put it at %s, "
                    "or publish it in a record of `published`" % task["output"])
        return None

    def remakes(self, task):
        return made_here(task)

    def check_environment(self, tasks):
        problems = []
        made = [task for task in tasks if made_here(task)]
        for kind in sorted({task["params"]["type"] for task in made}):
            if kind in INPUT_TYPES:
                command = next(task["params"]["command"] for task in made
                               if task["params"]["type"] == kind)
                # (the script, after python3)
                problems += missing_modules(INPUT_TYPES[kind].modules,
                                            os.path.basename(command[1]))
        return problems

    def run(self, task):
        """Make one input with its command, writing to a hidden folder next to it,
        then move what it made into place, the input itself last, so that an
        input that exists is complete."""
        output = task["output"]
        if not made_here(task):
            raise StageError("%s is made outside this repository (scenario %s), so it can't be "
                             "made here: put it there, or publish it in a record of `published`"
                             % (output, task["scenario"]))
        kind = task["params"]["type"]
        option = INPUT_TYPES[kind].save_dir_option if kind in INPUT_TYPES else None
        if option is None:
            raise StageError("%s inputs can't be made one at a time (INPUT_TYPES in "
                             "ddml_bench/scenarios.py has no option saying where they go)" % kind)
        folder = os.path.dirname(output)
        os.makedirs(folder, exist_ok=True)
        # Named without the input's extension, as ILD_Particle_Gun.py names its
        # .h5 file by replacing ".slcio" anywhere in the path
        work = tempfile.mkdtemp(dir=folder, prefix=".%s.%s.making." % (input_stem(output),
                                                                        process.host()))
        try:
            command = with_option(task["params"]["command"], option, work)
            say("making %s with" % output)
            say("  " + process.quote(command))
            with process.interruptible():
                status = process.run(command)
            if status != 0:
                raise StageError("making %s failed: the command exited with status %d"
                                 % (output, status))
            made = os.path.join(work, os.path.basename(output))
            if not os.path.isfile(made) or os.path.getsize(made) == 0:
                raise StageError("the command made no %s (it made: %s)" % (
                    os.path.basename(output), ", ".join(sorted(os.listdir(work))) or "nothing"))
            # What else it made (the guns' .h5 files of the truth) first, and
            # the input itself last, as that marks it done
            for name in sorted(os.listdir(work)):
                if name != os.path.basename(output):
                    os.replace(os.path.join(work, name), os.path.join(folder, name))
            os.replace(made, output)
        finally:
            shutil.rmtree(work, ignore_errors=True)
        say("made %s" % output)

    def describe(self, task):
        return "the input %s" % task["name"]

    def label(self, task):
        return "%s/%s" % (task["scenario"], task["name"])

    def log_name(self, task):
        return [self.name, task["scenario"], task["name"]]
