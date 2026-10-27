"""The local executor: run every task to do, one after another, in this
environment (the one from setup.sh). A task whose needs failed is skipped;
the others go on.
"""

from ..graph import BLOCKED, TODO, UNNEEDED
from ..stages import BY_NAME, StageError
from . import ExecutorError, Report, describe, label, report_stages, run_task, say


def run(graph, dry_run=False, out=None):
    """Run the graph's tasks to do (after find_status); returns the exit status:
    1 if a task failed, or if a stage whose blocked tasks are errors has some."""
    out = out or Report()
    tasks = graph.list
    report_stages(out, tasks, left_out=graph.left_out_reasons())
    todo = graph.tasks(TODO)
    blocked = [t for t in graph.tasks(BLOCKED) if BY_NAME[t["stage"]].blocked_is_error]
    problems = graph.environment_problems() if todo else []
    for problem in problems:
        say(problem)
    if problems and not dry_run:
        raise ExecutorError("this environment can't run the tasks (see above), so nothing was run")
    if not todo:
        if blocked:
            out("ddml-bench: nothing to run, and some tasks are blocked (see above)")
        elif graph.tasks(BLOCKED):
            out("ddml-bench: nothing to run, but some tasks are blocked (see above)")
        else:
            out("ddml-bench: everything selected is done%s, so there is nothing to run"
                % (" or not needed" if graph.tasks(UNNEEDED) else ""))
        return 1 if blocked else 0
    if dry_run:
        out("ddml-bench: would run %d task(s), in this order:" % len(todo))
        for task in todo:
            out("  %-12s %s  (%s)" % (task["stage"], label(task), task["why"]))
        out("ddml-bench: dry run, so nothing was run")
        if problems:
            say("but this environment can't run them (see above)")
        return 1 if problems or blocked else 0

    failed = {}  # key: why
    made = 0
    for number, task in enumerate(todo, 1):
        stuck = [key for key in task["after"] if key in failed]
        out("ddml-bench: [%d/%d] %s %s" % (number, len(todo), task["stage"], label(task)))
        if stuck:
            failed[task["key"]] = "skipped, as it needs %s" % describe(graph.by_key[stuck[0]])
            out("ddml-bench:   %s" % failed[task["key"]])
            continue
        try:
            made += run_task(task)
        except StageError as error:
            failed[task["key"]] = str(error).split("\n")[0]
            say("FAILED: %s" % error)
    out("ddml-bench: made %d of %d" % (made, len(todo)))
    if failed:
        say("these failed, or were skipped as they need what failed (logs are next to the "
            "outputs):")
        for task in todo:
            if task["key"] in failed:
                say("  %-12s %s: %s" % (task["stage"], label(task), failed[task["key"]]))
    if blocked:
        say("and %d task(s) are blocked (see above)" % len(blocked))
    return 1 if failed or blocked else 0
