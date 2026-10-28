"""What every stage of the pipeline declares.

A stage adds its tasks to the task graph (ddml_bench/graph.py), one per file it
makes: each task names the files it reads (inputs), the file that marks it
done (output, as the layout names it), the tasks it needs (needs, the keys of
the tasks making its inputs), and everything needed to run it (params), so
that it can be run anywhere from its record alone. The stage then says when a
task is done (current), what in the environment would stop its tasks
(check_environment), and how to run one (run).
"""

import importlib.util
import os


class StageError(Exception):
    """A task can't be run, or failed."""


class Stage(object):
    #: the stage's name, which --stages and --force take
    name = ""
    #: a task of this stage gathers what many others make (the measurements of
    #: all reconstructions of a simulator, the plots of all measurements), so it
    #: is out of date while any of them is still to be made
    gathers = False
    #: a task of this stage records what it was made from (the observables'
    #: .npz file, the reconstructions it measured), so current() tells from
    #: that record alone whether it is up to date: what it needs that is to be
    #: made only because it isn't there doesn't make it out of date, and isn't
    #: made for it while it is up to date
    records_inputs = False
    #: the stage's outputs are there only for what is made from them (the
    #: simulations and reconstructions, for the measurements), so a task of it
    #: whose output isn't there is made only if something still to be made
    #: needs it, or if nothing needs it at all (ddml_bench/graph.py)
    intermediate = False
    #: a task of this stage uses what there is of what it needs, rather than
    #: waiting for all of it (the plots, of whatever is measured)
    takes_what_there_is = False
    #: tasks of this stage that can't run, because what they need isn't there
    #: and won't be made, make a run fail rather than just being reported
    blocked_is_error = False

    def add_tasks(self, graph):
        """Add this stage's tasks to the graph."""
        raise NotImplementedError

    def current(self, task):
        """(done, why not): whether the task's output is there and up to date,
        from what is on disk, whatever the tasks it needs."""
        if os.path.exists(task["output"]):
            return True, ""
        return False, "not made yet"

    def nothing_to_use(self, task):
        """Why a task can't run with what there is, though nothing it needs is
        still to be made, or None: e.g. the plots, with nothing measured, or an
        input made elsewhere that isn't there."""
        return None

    def remakes(self, task):
        """Whether forcing the stage (--force) makes the task's output again:
        not for what the stage can't make, e.g. the inputs made elsewhere."""
        return True

    def check_environment(self, tasks):
        """What in this environment would make these tasks fail: a list of
        problems, empty if there are none."""
        return []

    def run(self, task):
        """Make the task's output; raise StageError if that fails."""
        raise NotImplementedError

    # How tasks are named in messages, job names and log files

    def describe(self, task):
        """What a task makes, in words."""
        return "the %s output %s" % (self.name, task["output"])

    def label(self, task):
        """A task, briefly, for the lists of tasks and jobs."""
        return task["name"]

    def job_name(self, task):
        return self.name

    def log_name(self, task):
        return [self.name]


def missing_modules(modules, what):
    """Problems for the Python modules that this environment doesn't have
    (found without importing them)."""
    problems = []
    for module in modules:
        try:
            found = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            problems.append("%s needs the Python module %s, which this environment doesn't have"
                            % (what, module))
    return problems


def resolve_compact(compact):
    """A compact file's path: relative to k4geo ($k4geo_DIR) unless absolute.
    None if k4geo_DIR isn't set and it is needed."""
    if os.path.isabs(compact):
        return compact
    k4geo = os.environ.get("k4geo_DIR") or os.environ.get("K4GEO")
    return os.path.join(k4geo, compact) if k4geo else None


def compact_problems(compacts, what):
    """Problems finding compact files."""
    problems = []
    for compact in compacts:
        path = resolve_compact(compact)
        if path is None:
            problems.append("k4geo_DIR isn't set, so the compact file %s for %s can't be found"
                            % (compact, what))
        elif not os.path.isfile(path):
            problems.append("compact file not found: %s" % path)
    return problems
