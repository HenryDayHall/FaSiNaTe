"""Running the timing tasks: here, one after another (timing.sh), or as a slurm
job each, on its kind of node (slurm_timing.sh).

Either way, what is done, what each model's inference runs with, and whether
the environment can run the rest are found in the environment the tasks run in
(ddml_timing/plan.py: find_status(), environment_problems()), and each task is
run by ddml_timing/runner.py.

slurm: each submission gets a folder, <paths.data_dir>/timing/slurm/
timing_<time>, with the plan (plan.json), every task and its status
(tasks.json), the job script (job.sh), the jobs submitted (jobs.tsv: job ID,
kind of node, task, slurm log), what was printed (report.txt) and each job's
slurm log (logs/). A job is submitted to its kind of node's partition, with
its constraint, time limit and sbatch options (timing.devices), after
slurm.sbatch_args and before any given on the command line. No job waits for
another. A task that a job of an earlier submission is still queued or
running for isn't submitted again, and submissions take turns.
"""

import errno
import getpass
import glob
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import time

from ddml_bench import process
from ddml_bench.environment import LAUNCHER
from ddml_bench.executors import Report
from ddml_bench.executors.slurm import active_jobs

from .plan import BLOCKED, DONE, SKIPPED, TODO, current, environment_problems, find_status
from .runner import TimingError, run

ENTRY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts",
                     "ddml-timing")
RECORDS = "jobs.tsv"
RECORDS_HEADER = "# job\tdevice\ttask\tlog\n"
LOCK = ".slurm_timing.lock"
SUBMISSION = "timing_"


class ExecutorError(Exception):
    pass


def say(message):
    print("ddml-timing: " + message, flush=True)


def write_json(path, data):
    partial = "%s.%d.partial" % (path, os.getpid())
    with open(partial, "w") as stream:
        json.dump(data, stream, indent=1)
        stream.write("\n")
    os.replace(partial, path)


def read_json(path):
    try:
        with open(path) as stream:
            return json.load(stream)
    except (OSError, ValueError) as error:
        raise ExecutorError("can't read %s: %s" % (path, error))


def in_environment_command(args, binds=()):
    """The command running `ddml-timing ARGS` in the environment the tasks run
    in (scripts/run_in_environment.sh)."""
    command = [LAUNCHER]
    for folder in binds:
        command += ["--bind", folder]
    return command + ["--", "python3", ENTRY] + list(args)


def run_in_environment(args, binds=()):
    with process.interruptible():
        return process.run(in_environment_command(args, binds), env=dict(os.environ))


def report(out, tasks, queued=None):
    """A table of what each kind of node has done and has to do, and why
    tasks are skipped or blocked."""
    devices = list(dict.fromkeys(task["device"] for task in tasks))
    columns = ["done"] + (["to submit", "queued"] if queued is not None else ["to run"]) \
        + ["skipped", "blocked"]
    out("ddml-timing: the timing tasks (one per input, simulator and kind of node):")
    out("  %-10s" % "node" + "".join(" %*s" % (max(len(c), 6), c) for c in columns))
    for device in devices:
        mine = [task for task in tasks if task["device"] == device]
        counts = dict((status, sum(task["status"] == status for task in mine))
                      for status in (DONE, TODO, SKIPPED, BLOCKED))
        waiting = sum(1 for task in mine if queued is not None and task["key"] in queued)
        values = [counts[DONE], counts[TODO] - waiting] + ([waiting] if queued is not None else [])
        values += [counts[SKIPPED], counts[BLOCKED]]
        out("  %-10s" % device + "".join(" %*d" % (max(len(c), 6), v)
                                         for c, v in zip(columns, values)))
    for status, what in ((SKIPPED, "skipped"), (BLOCKED, "blocked, so not run")):
        reasons = {}
        for task in tasks:
            if task["status"] == status:
                reasons.setdefault((task["device"], task["why"]), []).append(task)
        if reasons:
            out("ddml-timing: %s:" % what)
        for (device, why), some in sorted(reasons.items()):
            simulators = sorted({task["simulator"] for task in some})
            out("  %s: %s (%d task%s, %s): %s" % (
                device, ", ".join(simulators), len(some), "" if len(some) == 1 else "s",
                ", ".join(sorted({task["scenario"] for task in some})), why))


# ---------------------------------------------------------------------------
# Here, one after another


def run_local(plan, dry_run=False):
    """In this environment: time every task to do, one after another;
    returns the exit status (1 if a task failed, or some are blocked)."""
    out = Report()
    tasks = find_status(plan)
    report(out, tasks)
    todo = [task for task in tasks if task["status"] == TODO]
    blocked = [task for task in tasks if task["status"] == BLOCKED]
    problems = environment_problems(tasks) if todo else []
    for problem in problems:
        say(problem)
    if problems and not dry_run:
        raise ExecutorError("this environment can't run the timing (see above), so nothing was "
                            "run")
    if not todo:
        out("ddml-timing: nothing to time%s" % (", and some tasks are blocked (see above)"
                                               if blocked else ": everything selected is done"))
        return 1 if blocked else 0
    if dry_run:
        out("ddml-timing: would time %d task(s), in this order:" % len(todo))
        for task in todo:
            out("  %s  (%s)" % (task["key"], task["why"]))
        out("ddml-timing: dry run, so nothing was run")
        return 1 if problems or blocked else 0
    failed = []
    for number, task in enumerate(todo, 1):
        out("ddml-timing: [%d/%d] %s" % (number, len(todo), task["key"]))
        try:
            run(task, plan)
        except TimingError as error:
            failed.append((task, str(error).split("\n")[0]))
            say("FAILED: %s" % error)
    out("ddml-timing: timed %d of %d" % (len(todo) - len(failed), len(todo)))
    for task, why in failed:
        say("failed: %s: %s" % (task["key"], why))
    return 1 if failed or blocked else 0


def run_task_in_job(directory, key):
    """In a job: time the task KEY of DIR/tasks.json, unless it is done by now."""
    plan = read_json(os.path.join(directory, "plan.json"))
    tasks = read_json(os.path.join(directory, "tasks.json"))["tasks"]
    task = next((task for task in tasks if task["key"] == key), None)
    if task is None:
        raise ExecutorError("%s/tasks.json has no task %s" % (directory, key))
    done, _ = current(task)
    if done and not plan.get("force"):
        say("%s is timed already, so it isn't timed again" % key)
        return
    problems = environment_problems([dict(task, status=TODO)])
    if problems:
        raise ExecutorError("this environment can't run it:\n  " + "\n  ".join(problems))
    say("%s (%s)" % (key, task["why"] or "to do"))
    run(task, plan)


# ---------------------------------------------------------------------------
# As slurm jobs


def earlier_jobs(log_dir, directory):
    """{task: job ID} of the earlier submissions in log_dir, the latest for
    each task."""
    found = {}
    pattern = os.path.join(glob.escape(log_dir), SUBMISSION + "*", RECORDS)
    for path in sorted(glob.glob(pattern)):
        if os.path.realpath(os.path.dirname(path)) == os.path.realpath(directory):
            continue
        try:
            with open(path) as stream:
                lines = stream.read().splitlines()
        except OSError:
            continue
        for line in lines:
            fields = line.split("\t")
            if not line.startswith("#") and len(fields) >= 3:
                found[fields[2]] = fields[0]
    return found


def job_name(plan, task):
    return "%s-%s-%s" % (plan["settings"]["job_name_prefix"], task["device"], task["simulator"])


def log_name(number, task):
    parts = [task["device"], task["simulator"], task["scenario"], task["name"]]
    return "%03d-%s.out" % (number, re.sub(r"[^A-Za-z0-9_.+-]", "_", "-".join(parts)))


def sbatch_command(plan, task, log, job_script, extra=()):
    """The sbatch command of a task, on its kind of node."""
    device = plan["settings"]["devices"][task["device"]]
    command = ["sbatch", "--parsable", "--job-name=" + job_name(plan, task),
               # slurm reads % in file names as a pattern
               "--output=" + log.replace("%", "%%"), "--chdir=" + plan["repo_dir"]]
    for option in ("time", "partition", "constraint"):
        if device.get(option):
            command.append("--%s=%s" % (option, device[option]))
    command += [str(arg) for arg in plan["slurm"]["sbatch_args"]]
    command += [str(arg) for arg in device["sbatch_args"]] + [str(arg) for arg in extra]
    return command + [job_script, task["key"]]


def write_job_script(directory, plan):
    """job.sh KEY: time the task KEY of this submission, in the environment
    from setup.sh, with the GPU passed into the container on a GPU node."""
    cuda = ["%s/*" % name for name, device in plan["settings"]["devices"].items()
            if device["inference"] == "cuda"]
    lines = [
        "#!/usr/bin/env bash",
        "# Written by ddml-timing (slurm_timing.sh) on %s. Every job of this" % process.now(),
        "# submission runs it, as job.sh KEY: it times the task KEY of tasks.json, in",
        "# the environment from setup.sh. The jobs are listed in jobs.tsv.",
        "set -euo pipefail",
        "repo=%s" % shlex.quote(plan["repo_dir"]),
        "dir=%s" % shlex.quote(directory),
        "export FASINATE_CONFIG=%s" % shlex.quote(":".join(plan["config_files"])),
    ]
    if os.environ.get("FASINATE_PYTHON"):
        lines.append("export FASINATE_PYTHON=%s" % shlex.quote(os.environ["FASINATE_PYTHON"]))
    lines += [
        "# sbatch hands the job the environment it was submitted from; if that had",
        "# sourced setup.sh, the task would run in it rather than set up its own",
        "unset DDML_SOURCE_DIR KEY4HEP_STACK",
        "key=$1",
        "# What the results record of the job: the environment's clean shell keeps",
        "# only the FASINATE_* variables",
        "export FASINATE_JOB_ID=${SLURM_JOB_ID:-} FASINATE_JOB_PARTITION=${SLURM_JOB_PARTITION:-}",
        "export FASINATE_JOB_NODELIST=${SLURM_JOB_NODELIST:-}",
        "export FASINATE_JOB_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}",
    ]
    if cuda:
        lines += [
            "# On a GPU node, the GPU goes into the container too (apptainer --nv), if",
            "# there is one (container.enabled)",
            "case $key in",
            "    %s)" % "|".join(cuda),
            "        if command -v apptainer >/dev/null; then export APPTAINER_NV=1",
            "        else export SINGULARITY_NV=1; fi ;;",
            "esac",
        ]
    lines += [
        "echo \"ddml-timing: job ${SLURM_JOB_ID:-} on $(hostname), $(date '+%F %T'): $key\"",
        "cd \"$repo\"",
        "status=0",
        "%s --bind \"$dir\" -- python3 %s run-task \"$dir\" \"$key\" || status=$?"
        % (shlex.quote(LAUNCHER), shlex.quote(ENTRY)),
        "echo \"ddml-timing: job ${SLURM_JOB_ID:-} finished $(date '+%F %T'), with status "
        "$status\"",
        "exit \"$status\"",
    ]
    path = os.path.join(directory, "job.sh")
    with open(path, "w") as stream:
        stream.write("\n".join(lines) + "\n")
    os.chmod(path, 0o755)
    return path


def submit(directory, dry_run=False, extra=()):
    """Submit a job for every task still to do in DIR/tasks.json; returns the
    exit status."""
    plan = read_json(os.path.join(directory, "plan.json"))
    found = read_json(os.path.join(directory, "tasks.json"))
    tasks = found["tasks"]
    job_script = os.path.join(directory, "job.sh")
    records_path = os.path.join(directory, RECORDS)
    out = Report()

    earlier = earlier_jobs(plan["slurm"]["log_dir"], directory)
    active = active_jobs()
    if active is None:
        if earlier:
            say("warning: squeue can't tell which jobs of earlier submissions are still queued, "
                "so this may submit jobs again that are queued already")
        active = {}
    queued = dict((task["key"], earlier[task["key"]]) for task in tasks
                  if task["status"] == TODO and earlier.get(task["key"]) in active)
    report(out, tasks, queued)
    to_submit = [task for task in tasks if task["status"] == TODO and task["key"] not in queued]
    if found["problems"]:
        for problem in found["problems"]:
            say(problem)
        if to_submit and not dry_run:
            raise ExecutorError("the environment can't run the jobs (see above), so nothing was "
                                "submitted")
    if not to_submit:
        if queued:
            out("ddml-timing: nothing new to submit; %d job(s) of earlier submissions are still "
                "queued or running" % len(queued))
        elif any(task["status"] == BLOCKED for task in tasks):
            out("ddml-timing: nothing to submit, but some tasks are blocked (see above)")
        else:
            out("ddml-timing: everything selected is timed, so there is nothing to submit; "
                "./timing_summary.sh tabulates it")
        return 0

    out("ddml-timing: %s %d job(s) (job, node, task):" % (
        "would submit" if dry_run else "submitting", len(to_submit)))
    examples = {}
    submitted = 0
    try:
        for number, task in enumerate(to_submit, 1):
            log = os.path.join(directory, "logs", log_name(number, task))
            command = sbatch_command(plan, task, log, job_script, extra)
            examples.setdefault(task["device"], command)
            if dry_run:
                job = "#%d" % number
            else:
                result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        universal_newlines=True)
                job = result.stdout.strip().split(";")[0]
                if result.returncode != 0 or not job.isdigit():
                    raise ExecutorError("sbatch failed for %s: %s" % (
                        task["key"], (result.stderr or result.stdout).strip()))
                new = not os.path.exists(records_path)
                with open(records_path, "a") as stream:
                    if new:
                        stream.write(RECORDS_HEADER)
                    stream.write("%s\t%s\t%s\t%s\n" % (job, task["device"], task["key"], log))
                submitted += 1
            out("  %-10s %-6s %s" % (job, task["device"], task["key"]))
    except (ExecutorError, OSError) as error:
        say(str(error))
        if submitted:
            say("the %d job(s) submitted before that are in %s, so running slurm_timing.sh again "
                "waits for them instead of submitting them again" % (submitted, records_path))
        out.save(os.path.join(directory, "report.txt"))
        return 1
    if queued:
        out("ddml-timing: and waiting for %d job(s) of earlier submissions" % len(queued))
    if dry_run:
        out("ddml-timing: dry run, so nothing was submitted; the jobs would be submitted as")
        for device, command in examples.items():
            out("  %s" % process.quote(command))
        if found["problems"]:
            say("but the environment can't run the jobs (see above), so slurm_timing.sh wouldn't "
                "submit them")
            return 1
        return 0
    out("ddml-timing: job script, slurm logs (logs/) and the list of jobs (%s) in %s"
        % (RECORDS, directory))
    out("ddml-timing: the results go to %s; ./timing_summary.sh tabulates them"
        % plan["timing_dir"])
    out("ddml-timing: follow the jobs with: squeue -u %s -o '%%.12i %%.40j %%.9T %%.11M %%R'"
        % (os.environ.get("USER") or getpass.getuser()))
    out("ddml-timing: run slurm_timing.sh again, with the same options, to resubmit whatever "
        "fails or times out")
    out.save(os.path.join(directory, "report.txt"))
    return 0


def _lock(log_dir):
    """Take turns with other submissions: hold the lock in log_dir until the
    file returned is closed."""
    import fcntl
    stream = open(os.path.join(log_dir, LOCK), "w")
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        if error.errno not in (errno.EAGAIN, errno.EACCES):
            raise
        say("waiting for another slurm_timing.sh to finish submitting")
        fcntl.flock(stream, fcntl.LOCK_EX)
    return stream


def run_slurm(plan, dry_run=False, extra=()):
    """Work out what is done in the environment the jobs run in, and submit the
    rest; returns the exit status."""
    log_dir = plan["slurm"]["log_dir"]
    if not dry_run and not shutil.which("sbatch"):
        raise ExecutorError("sbatch isn't on PATH; run this on a slurm login node (--dry-run "
                            "shows what it would do)")
    os.makedirs(log_dir, exist_ok=True)
    lock = None
    if dry_run:
        directory = tempfile.mkdtemp(dir=log_dir, prefix=".dry_run.")
    else:
        lock = _lock(log_dir)
        directory = os.path.join(log_dir, "%s%s_%d" % (SUBMISSION, time.strftime("%Y%m%d_%H%M%S"),
                                                       os.getpid()))
        os.makedirs(directory)
    try:
        os.makedirs(os.path.join(directory, "logs"))
        write_json(os.path.join(directory, "plan.json"), plan)
        say("finding what is done, in the environment the jobs have")
        status = run_in_environment(["prepare", os.path.join(directory, "plan.json"),
                                     os.path.join(directory, "tasks.json")], [directory])
        if status != 0:
            raise ExecutorError("working out the jobs failed (see above)")
        write_job_script(directory, plan)
        return submit(directory, dry_run, extra)
    finally:
        if dry_run or not os.path.exists(os.path.join(directory, RECORDS)):
            shutil.rmtree(directory, ignore_errors=True)
        if lock is not None:
            lock.close()


def prepare(plan_path, tasks_path):
    """In the environment: every task and its status, and what in this
    environment would stop the tasks to do, to tasks_path."""
    plan = read_json(plan_path)
    tasks = find_status(plan)
    write_json(tasks_path, {"tasks": tasks, "problems": environment_problems(tasks)})
