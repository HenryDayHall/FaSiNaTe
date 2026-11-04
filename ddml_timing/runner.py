"""Running one timing task, in the environment from setup.sh (a slurm job's,
or this shell's): ddsim on the input's first warmup_events + events events,
plus one, whose start ends the last event timed.

The command is the pipeline's for the simulator (ddml_bench/stages/simulate.py:
its steering file, compact file and options, from DDML's scripts/ folder), with
ddml_timing/ddsim_steer_timing.py as the steering file, which runs the
pipeline's after setting up the timing of the models' embedded-Python
inference, on the GPU if the task's kind of node runs it there
(ddml_timing/inference.py). ddsim runs on one CPU (timing.pin_cpu: the first
the job may use), with its output line-buffered (stdbuf, where there is one),
into a pipe: each line is written to the log as it comes, and as each event's
first line comes, the moment and the CPU time ddsim has used so far are noted
(ddml_timing/events.py). The simulation goes to the node's temporary folder and
is deleted after (unless timing.keep_simulations).

The results (ddml_timing/plan.py names them) are written only when ddsim
succeeds, to hidden files that are renamed once complete; a failure writes
.failed.json, saying why, instead.
"""

import csv
import io
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time

from ddml_bench import process
from ddml_bench.stages.base import resolve_compact
from ddml_bench.stages.simulate import Geant4, Models

from .events import COLUMNS, Events, statistics
from .plan import FORMAT, models_path

STEERING = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ddsim_steer_timing.py")
# What the job script passes in for the record (the clean shell of the
# environment keeps only FASINATE_* variables)
JOB_VARIABLES = (("FASINATE_JOB_ID", "SLURM_JOB_ID"),
                 ("FASINATE_JOB_PARTITION", "SLURM_JOB_PARTITION"),
                 ("FASINATE_JOB_NODELIST", "SLURM_JOB_NODELIST"),
                 ("FASINATE_JOB_CUDA_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"))
# How often the log is flushed while ddsim runs (s)
FLUSH_EVERY = 2.0


class TimingError(Exception):
    """A task can't be run, or failed."""


def say(message):
    print("ddml-timing: " + message)
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# The node


def _first_line(command, timeout=30):
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                universal_newlines=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout if result.returncode == 0 else None


def gpus():
    """[{index, name, memory, driver}] of the node's NVIDIA GPUs, as
    nvidia-smi lists them; empty without nvidia-smi."""
    if not shutil.which("nvidia-smi"):
        return []
    text = _first_line(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version",
                        "--format=csv,noheader"]) or ""
    found = []
    for line in text.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 4:
            found.append(dict(zip(("index", "name", "memory", "driver"), fields)))
    return found


def cpu_model():
    try:
        with open("/proc/cpuinfo") as stream:
            for line in stream:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


def memory_gb():
    try:
        with open("/proc/meminfo") as stream:
            for line in stream:
                if line.startswith("MemTotal:"):
                    return round(int(line.split()[1]) / 1024.0 ** 2, 1)
    except (OSError, ValueError, IndexError):
        pass
    return None


def allowed_cpus():
    try:
        return sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return list(range(os.cpu_count() or 1))


def node_info():
    """The node a task runs on, for its results."""
    job = {}
    for ours, slurms in JOB_VARIABLES:
        value = os.environ.get(ours) or os.environ.get(slurms)
        if value:
            job[slurms.lower()] = value
    return {"host": socket.gethostname(), "cpu_model": cpu_model(),
            "cpus": os.cpu_count(), "cpus_allowed": len(allowed_cpus()),
            "memory_gb": memory_gb(), "gpus": gpus(), "job": job}


def process_cpu_seconds(pid):
    """The CPU time (user and system) process pid has used, all its threads, in
    seconds, to the clock tick; None if it can't be read."""
    try:
        with open("/proc/%d/stat" % pid) as stream:
            text = stream.read()
    except OSError:
        return None
    fields = text[text.rfind(")") + 2:].split()
    try:
        return (int(fields[11]) + int(fields[12])) / float(os.sysconf("SC_CLK_TCK"))
    except (IndexError, ValueError, OSError):
        return None


# ---------------------------------------------------------------------------
# The command


def ddsim_command(task, compact, output, n_events):
    """(command, the steering file the pipeline runs it with): the pipeline's
    ddsim command for the task's simulator, with the timing steering file in
    place of its own."""
    stand_in = {"inputs": [task["input"]], "scenario": task["scenario"],
                "simulator": task["simulator"],
                "params": {"presets": task["presets"], "model_configs": task["model_configs"],
                           "events": n_events, "seed": task["seed"], "model": task["model"]}}
    stage = Geant4() if task["model"] is None else Models()
    command = stage.command(stand_in, compact, output)
    index = command.index("--steeringFile") + 1
    steering = command[index]
    command[index] = STEERING
    return command, steering


def _header(task, plan, command, cwd, node, cpu, buffered):
    source = os.environ.get("DDML_SOURCE_DIR")
    stand_in = {"scenario": task["scenario"], "simulator": task["simulator"],
                "params": {"model": task["model"]}}
    lines = ["# ddml-timing: %s on %s, on a %s node (inference: %s)" % (
        task["simulator"], task["scenario"], task["device"], task["inference_on"])]
    if task["model"] is not None:
        lines += Models().header(stand_in)[1:]
        lines.append(Models().ddml_line(source))
    else:
        lines.append(Geant4().ddml_line(source))
    gpu_names = ", ".join("%s (%s)" % (g["name"], g["memory"]) for g in node["gpus"]) or "none"
    lines += [
        "# input     %s" % task["input"],
        "# seed      %s" % (task["seed"] if task["seed"] is not None
                            else "ddsim's own, chosen at random (see below)"),
        "# events    %d timed, after %d of warm-up" % (task["events"], task["warmup_events"]),
        "# started   %s on %s" % (process.now(), node["host"]),
        "# node      %s; %s CPUs, %s allowed; %s GB; GPUs: %s" % (
            node["cpu_model"], node["cpus"], node["cpus_allowed"], node["memory_gb"], gpu_names),
        "# job       %s" % (", ".join("%s=%s" % item for item in sorted(node["job"].items()))
                            or "none"),
        "# pinned    %s" % ("CPU %d" % cpu if cpu is not None else "no"),
        "# Key4hep   %s" % (os.environ.get("DDML_KEY4HEP_RELEASE") or "unknown"),
        "# command   (in %s%s) %s" % (cwd, ", line-buffered" if buffered else "",
                                       process.quote(command)),
    ]
    return lines


# ---------------------------------------------------------------------------
# Running it


def _write_atomic(path, write):
    folder, name = os.path.split(path)
    partial = os.path.join(folder, ".%s.%d.%s" % (process.host(), os.getpid(), name))
    try:
        with open(partial, "w", newline="") as stream:
            write(stream)
        os.replace(partial, path)
    finally:
        process.remove(partial)


def write_failure(task, reason):
    record = {"format": FORMAT, "identity": task["identity"], "key": task["key"],
              "reason": reason, "finished": process.now(), "log": task["log"]}
    try:
        os.makedirs(os.path.dirname(task["failed"]), exist_ok=True)
        _write_atomic(task["failed"], lambda stream: json.dump(record, stream, indent=1))
    except OSError:
        pass


def run(task, plan):
    """Time one task, writing its results; raises TimingError if it fails."""
    if not os.environ.get("DDML_SOURCE_DIR"):
        raise TimingError("DDML isn't set up in this environment; source setup.sh first")
    settings = plan["settings"]
    compact = resolve_compact(task["compact"])
    if compact is None:
        raise TimingError("k4geo_DIR isn't set, so the compact file %s can't be found"
                          % task["compact"])
    if not os.path.isfile(task["input"]):
        raise TimingError("its input isn't there: %s" % task["input"])
    folder = os.path.dirname(task["result"])
    os.makedirs(folder, exist_ok=True)
    keep = settings.get("keep_simulations")
    scratch = tempfile.mkdtemp(prefix=".ddml-timing.%s." % process.host(),
                               dir=folder if keep else None)
    output = os.path.join(scratch, ".%s.root" % task["name"])
    # One more event than are timed and left out, whose start ends the last one
    n_events = task["warmup_events"] + task["events"] + 1
    command, steering = ddsim_command(task, compact, output, n_events)
    stdbuf = shutil.which("stdbuf")
    if stdbuf:
        command = [stdbuf, "-oL", "-eL"] + command
    cwd = os.path.join(os.environ["DDML_SOURCE_DIR"], "scripts")
    cpus = allowed_cpus()
    cpu = cpus[0] if settings.get("pin_cpu") and cpus else None
    env = process.child_environment(
        FASINATE_TIMED_STEERING=steering,
        FASINATE_TIMED_INFERENCE=task["inference"] if task["python_modules"] else "cpu",
        FASINATE_TIMED_PYTHON_MODULES=",".join(task["python_modules"]),
        FASINATE_TIMED_MODULE_PATH=models_path(task) or "")
    visible = os.environ.get("FASINATE_JOB_CUDA_VISIBLE_DEVICES")
    if visible and not env.get("CUDA_VISIBLE_DEVICES"):
        env["CUDA_VISIBLE_DEVICES"] = visible
    node = node_info()
    process.remove(task["failed"])
    events = Events()
    say("%s: log %s" % (task["key"], task["log"]))
    try:
        with process.interruptible():
            status, started, ended, wall = _run_ddsim(
                command, cwd, env, cpu, task, plan, node, events, bool(stdbuf))
    except process.Interrupted:
        write_failure(task, "interrupted (signal); the log is %s" % task["log"])
        shutil.rmtree(scratch, ignore_errors=True)
        raise
    except OSError as error:
        shutil.rmtree(scratch, ignore_errors=True)
        write_failure(task, "ddsim couldn't be run: %s" % error)
        raise TimingError("ddsim couldn't be run: %s" % error)
    records = events.records(task["warmup_events"])
    timed = [record for record in records if not record["warmup"]]
    why = None
    if status != 0:
        why = "ddsim exited with status %d" % status
    elif not events.starts:
        why = "ddsim printed no event's start (GenerationInit's \"Initializing event\" lines)"
    elif not timed:
        why = ("no event to time: %d event(s) ended, and the first %d are the warm-up"
               % (len(records), task["warmup_events"]))
    if why is None and keep:
        simulation = os.path.join(folder, task["name"] + ".root")
        if os.path.isfile(output):
            os.replace(output, simulation)
    shutil.rmtree(scratch, ignore_errors=True)
    if why is not None:
        process.append(task["log"], "# FAILED    %s: %s" % (process.now(), why))
        write_failure(task, why)
        raise TimingError("%s; the end of its log, %s:\n%s" % (why, task["log"],
                                                              process.log_tail(task["log"])))
    result = results(task, plan, node, events, records, timed, cpu, command, steering,
                     bool(stdbuf), started, ended, wall)
    _write_atomic(task["events_file"], lambda stream: _write_events(stream, records))
    _write_atomic(task["result"], lambda stream: json.dump(result, stream, indent=1))
    process.append(task["log"], "# finished  %s, after %d s: %d events timed, %.6g s each "
                                "(+- %.2g%%)" % (process.now(), round(wall), len(timed),
                                                 result["wall"]["mean"],
                                                 100 * (result["wall"]["rel_sem"] or 0)))
    say("%s: %d events timed, %.6g s each on average" % (task["key"], len(timed),
                                                          result["wall"]["mean"]))
    return result


def _run_ddsim(command, cwd, env, cpu, task, plan, node, events, buffered):
    """Run ddsim, its output to the log line by line, each line read by
    events as it comes out; returns (exit status, start, end, wall-clock
    seconds)."""
    def pin():
        if cpu is not None:
            os.sched_setaffinity(0, {cpu})

    with io.open(task["log"], "w", encoding="utf-8", errors="replace") as log:
        for line in _header(task, plan, command, cwd, node, cpu, buffered):
            log.write(line + "\n")
        log.flush()
        sys.stdout.flush()
        started = process.now()
        origin = time.monotonic()
        child = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 preexec_fn=pin)
        flushed = 0.0  # when the log was last flushed, as now is: from the start
        try:
            for raw in iter(child.stdout.readline, b""):
                now = time.monotonic() - origin
                line = raw.decode("utf-8", "replace").rstrip("\n")
                cpu_time = process_cpu_seconds(child.pid) if Events.is_start(line) else None
                events.line(line, now, cpu_time)
                log.write(line + "\n")
                if now - flushed > FLUSH_EVERY:
                    log.flush()
                    flushed = now
            status = child.wait()
        except process.Interrupted:
            child.terminate()
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            log.write("# INTERRUPTED %s\n" % process.now())
            raise
        finally:
            child.stdout.close()
        wall = time.monotonic() - origin
    return status, started, process.now(), wall


def _write_events(stream, records):
    writer = csv.writer(stream)
    writer.writerow(COLUMNS)
    for record in records:
        writer.writerow(["" if record[c] is None else
                         ("%.9f" % record[c] if isinstance(record[c], float) else record[c])
                         for c in COLUMNS])


def results(task, plan, node, events, records, timed, cpu, command, steering, buffered,
            started, ended, wall):
    """What a task's .json records."""
    calls = [record["model_calls"] for record in timed]
    inference = [record["inference_s"] or 0.0 for record in timed] if any(calls) else []
    first, last = events.starts[0][1], events.starts[len(records)][1]
    return {
        "format": FORMAT,
        "identity": task["identity"],
        "key": task["key"],
        "device": task["device"],
        "scenario": task["scenario"],
        "simulator": task["simulator"],
        "input": task["input"],
        "seed": task["seed"],
        "node": node,
        "software": {"key4hep": os.environ.get("DDML_KEY4HEP_RELEASE"),
                     "ddml_commit": os.environ.get("DDML_COMMIT"),
                     "ddml_install": os.environ.get("DDML_INSTALL_DIR")},
        "inference": {"setting": task["inference"], "on": task["inference_on"],
                      "backends": task["backends"], "models": events.models},
        "run": {"command": command, "steering": steering, "pinned_cpu": cpu,
                "line_buffered": buffered, "started": started, "finished": ended,
                "wall_s": wall, "before_first_event_s": first,
                "events_span_s": last - first, "ddsim": events.ddsim},
        "events": {"requested": task["events"], "warmup": task["warmup_events"],
                   "ended": len(records), "begun": events.begun, "timed": len(timed),
                   "input_ended": events.input_ended,
                   "timed_fewer": len(timed) < task["events"]},
        "wall": statistics([record["wall_s"] for record in timed]),
        "cpu": statistics([record["cpu_s"] for record in timed]),
        "model_calls": statistics(calls),
        # Per event, of the models' embedded-Python inference calls (timed by
        # ddml_timing/inference.py); None where nothing was timed
        "inference_time": statistics(inference) if inference else None,
        "barrel_showers": statistics([record["barrel_showers"] for record in timed]),
        "warmup_wall": statistics([record["wall_s"] for record in records if record["warmup"]]),
    }


__all__ = ["TimingError", "node_info", "run"]
