"""Executors: what runs the tasks of the graph.

  local  (local.py)  runs every task to do, one after another, in the
                     environment from setup.sh: pipeline.sh and the stage scripts
  slurm  (slurm.py)  submits a slurm job for every task to do, each waiting for
                     the jobs of the tasks it needs: slurm_pipeline.sh

Either way, a task is run by run_task(), which is what `ddml-bench run-task`
runs in each slurm job, and what a GitHub Actions shard will run.
"""

import os
import sys

from ..graph import BLOCKED, DONE, DONE_WITHOUT, TODO, UNNEEDED, UNSELECTED
from ..stages import BY_NAME, STAGE_NAMES, StageError


class ExecutorError(Exception):
    pass


def say(message):
    print("ddml-bench: " + message, file=sys.stderr)


class Report(object):
    """Lines printed, and kept, e.g. for a report file."""

    def __init__(self):
        self.lines = []

    def __call__(self, line=""):
        print(line)
        sys.stdout.flush()
        self.lines.append(line)

    def save(self, path):
        with open(path, "w") as stream:
            stream.write("\n".join(self.lines) + "\n")


def label(task):
    return BY_NAME[task["stage"]].label(task)


def describe(task):
    return BY_NAME[task["stage"]].describe(task)


def report_stages(out, tasks, queued=None):
    """A table of what each stage has done and has to do; with queued (the
    tasks that jobs of earlier submissions are queued for), as slurm sees it."""
    columns = ["done", "to submit", "queued"] if queued is not None else ["done", "to run"]
    columns += ["blocked", "not needed", "not selected"]
    out("ddml-bench: what each stage has done, and has to do:")
    out("  %-12s" % "stage" + "".join(" %*s" % (max(len(c), 6), c) for c in columns))
    for stage in STAGE_NAMES:
        counts = dict((status, 0) for status in (DONE, TODO, BLOCKED, UNNEEDED, UNSELECTED))
        waiting = 0
        for task in tasks:
            if task["stage"] == stage:
                counts[task["status"]] += 1
                waiting += queued is not None and task["key"] in queued
        values = [counts[DONE], counts[TODO] - waiting]
        if queued is not None:
            values.append(waiting)
        values += [counts[BLOCKED], counts[UNNEEDED], counts[UNSELECTED]]
        out("  %-12s" % stage + "".join(" %*d" % (max(len(c), 6), v)
                                        for c, v in zip(columns, values)))
    # What the tasks not needed are for, which is done without them: e.g. the
    # measurements of simulators whose .npz files are up to date, fetched or
    # made before their reconstructions were deleted
    done_with = []
    for task in tasks:
        if task["status"] == UNNEEDED and task["why"][len(DONE_WITHOUT):] not in done_with:
            done_with.append(task["why"][len(DONE_WITHOUT):])
    if done_with:
        out("ddml-bench: not needed, so not made, as what they are for is done: %s%s "
            "(--force makes them anyway)"
            % (", ".join(done_with[:3]), ", ..." if len(done_with) > 3 else ""))
    blocked = [task for task in tasks if task["status"] == BLOCKED]
    if blocked:
        out("ddml-bench: blocked, so not run:")
        for task in blocked:
            out("  %-12s %s: %s" % (task["stage"], label(task), task["why"]))


def run_task(task, check=False):
    """Make one task's output. Unless the task makes its output again (redo),
    a task whose output is there by now is left alone. With check, the stage's
    environment checks run first, for this task. Raises StageError."""
    stage = BY_NAME[task["stage"]]
    if not task.get("redo"):
        done, _ = stage.current(task)
        if done:
            print("ddml-bench: %s is there already, so it isn't made again" % describe(task))
            return False
    if check:
        problems = stage.check_environment([task])
        if problems:
            raise StageError("this environment can't run it:\n  " + "\n  ".join(problems))
    try:
        stage.run(task)
    except OSError as error:  # e.g. a folder that can't be written to
        raise StageError("%s failed: %s" % (describe(task), error))
    if not os.path.exists(task["output"]):
        raise StageError("%s finished without making %s" % (task["stage"], task["output"]))
    return True
