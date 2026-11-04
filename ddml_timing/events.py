"""Reading ddsim's output event by event, and the statistics of the times.

ddsim prints a line as each event starts, before it reads the event from the
input (DDG4's generator action, GenerationInit):

  GenerationInit   INFO  +++ Initializing event 12. Within run:0 event 12.

so an event's time is the time from its line to the next event's: reading it,
simulating it, the models' showers in it, and writing it out. The runner
(runner.py) notes the moment each line comes out of ddsim, and at each of
these lines the CPU time ddsim has used; the last event begun has no next
line, so it isn't timed (unless the input runs out after it: ddsim then
begins one more, which finds nothing to read). Within an event:

  PolyhedraBarrelGeometry::localToGlobal INFO  nLayer = 30
      DDML places one shower of a model in the barrel (one line per shower)
  ddml-timing: inference <module>.<model> <seconds>
      one call of a model's embedded-Python inference, and how long it took
      (inference.py prints these)

DD4hep flushes its output before each line it prints, and the runner runs
ddsim with its output line-buffered where it can (stdbuf), so the lines come
out as they are printed.

This needs only the standard library.
"""

import math
import re

START = re.compile(r"\+\+\+ Initializing event (\d+)\.")
BEGIN = re.compile(r"\+\+\+ Event (\d+) Begin event action")
SHOWER = re.compile(r"PolyhedraBarrelGeometry::localToGlobal\s+INFO\b")
INFERENCE = re.compile(r"^ddml-timing: inference (\S+) ([0-9.eE+-]+)\s*$")
# What the steering file says it set up (inference.py)
MODEL = re.compile(r"^ddml-timing: model (\S+) on (\S+)(?: \((.*)\))?\s*$")
INPUT_ENDED = re.compile(r"End of file reached")
# ddsim's own summary, at the end: user CPU time, from the run's start
DDSIM_TOTAL = re.compile(r"Total Time:\s+([0-9.]+) s \(User\),\s+([0-9.]+) s \(System\)")
DDSIM_STARTUP = re.compile(r"StartUp Time:\s+([0-9.]+) s, Processing and Init:\s+([0-9.]+) s")

# The columns of the .events.csv files
COLUMNS = ("event", "warmup", "start_s", "wall_s", "cpu_s", "model_calls", "inference_s",
           "barrel_showers")


class Events(object):
    """What ddsim printed, event by event: feed it each line as it comes out."""

    def __init__(self):
        # (number, wall, cpu) as each event's first line came out
        self.starts = []
        self.showers = []
        self.calls = []
        self.inference = []
        # Events that began their simulation (the last start may only have
        # found the input ended)
        self.begun = 0
        self.input_ended = False
        self.models = []
        self.ddsim = {}

    @staticmethod
    def is_start(line):
        return START.search(line) is not None

    def line(self, line, wall, cpu=None):
        """One line of ddsim's output, with the moment it came out (wall, in
        seconds) and, for an event's first line, ddsim's CPU time then."""
        match = START.search(line)
        if match:
            self.starts.append((int(match.group(1)), wall, cpu))
            self.showers.append(0)
            self.calls.append(0)
            self.inference.append(0.0)
            return
        match = INFERENCE.match(line)
        if match:
            if self.starts:
                self.calls[-1] += 1
                self.inference[-1] += float(match.group(2))
            return
        if SHOWER.search(line):
            if self.starts:
                self.showers[-1] += 1
            return
        if BEGIN.search(line):
            self.begun += 1
            return
        match = MODEL.match(line)
        if match:
            self.models.append({"model": match.group(1), "device": match.group(2),
                                "detail": match.group(3) or ""})
            return
        if INPUT_ENDED.search(line):
            self.input_ended = True
            return
        match = DDSIM_TOTAL.search(line)
        if match:
            self.ddsim.update(user_s=float(match.group(1)), system_s=float(match.group(2)))
            return
        match = DDSIM_STARTUP.search(line)
        if match:
            self.ddsim.update(startup_s=float(match.group(1)),
                              processing_and_init_s=float(match.group(2)))

    def records(self, warmup):
        """One record (a dict of COLUMNS) per event with a next event's line to
        end it, in order, the first warmup of them marked as warm-up; start_s
        is when the event began, from the first event's start."""
        found = []
        if not self.starts:
            return found
        origin = self.starts[0][1]
        for index in range(len(self.starts) - 1):
            number, wall, cpu = self.starts[index]
            _, next_wall, next_cpu = self.starts[index + 1]
            found.append({
                "event": number,
                "warmup": int(index < warmup),
                "start_s": wall - origin,
                "wall_s": next_wall - wall,
                "cpu_s": None if cpu is None or next_cpu is None else next_cpu - cpu,
                "model_calls": self.calls[index],
                "inference_s": self.inference[index] if self.calls[index] else None,
                "barrel_showers": self.showers[index],
            })
        return found


def statistics(values):
    """The statistics of a time per event over the events timed: n, mean,
    sd (the sample's), sem (sd / sqrt(n)), rel_sem (sem / mean), median, min,
    max and total, and, to show whether the times drift (e.g. a warm-up
    longer than the events left out), the means of the first and second
    halves of the events and their difference in units of its uncertainty,
    drift_z. None for those that need more events than there are."""
    values = [v for v in values if v is not None]
    n = len(values)
    found = {"n": n, "mean": None, "sd": None, "sem": None, "rel_sem": None, "median": None,
             "min": None, "max": None, "total": None, "first_half_mean": None,
             "second_half_mean": None, "drift_z": None}
    if not n:
        return found
    mean = sum(values) / n
    ordered = sorted(values)
    middle = n // 2
    found.update(mean=mean, min=ordered[0], max=ordered[-1], total=sum(values),
                 median=ordered[middle] if n % 2 else (ordered[middle - 1] + ordered[middle]) / 2)
    if n > 1:
        sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
        found.update(sd=sd, sem=sd / math.sqrt(n), rel_sem=sd / math.sqrt(n) / mean if mean else None)
    if n >= 4:
        halves = [_mean_and_sem(values[:middle]), _mean_and_sem(values[middle:])]
        found.update(first_half_mean=halves[0][0], second_half_mean=halves[1][0])
        spread = math.sqrt(halves[0][1] ** 2 + halves[1][1] ** 2)
        if spread > 0:
            found["drift_z"] = (halves[0][0] - halves[1][0]) / spread
    return found


def _mean_and_sem(values):
    n = len(values)
    mean = sum(values) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1)) if n > 1 else 0.0
    return mean, sd / math.sqrt(n)


def events_needed(rel_sem, n, target=0.1):
    """How many events a relative uncertainty of rel_sem with n events says
    are needed for target: n * (rel_sem / target)^2, or None."""
    if rel_sem is None or not n:
        return None
    return int(math.ceil(n * (rel_sem / target) ** 2))
