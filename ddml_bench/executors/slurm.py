"""The slurm executor: a slurm job for every task to do, each waiting for the
jobs of the tasks it needs (sbatch --dependency=afterok, with
--kill-on-invalid-dep=yes, so a job waiting for one that fails is cancelled).

Each submission gets a folder, <slurm.log_dir>/pipeline_<time>, with the plan
(plan.json), every task and its status (tasks.json), the job script (job.sh),
the jobs submitted (jobs.tsv: job ID, stage, task, slurm log), what was printed
(report.txt) and each job's slurm log (logs/). Every job runs
`job.sh KEY`, which runs `ddml-bench run-task DIR KEY` in the environment from
setup.sh.

A task that a job of an earlier submission is still queued or running for
isn't submitted again: its job is waited for instead. Submissions take turns,
so that two at once don't submit the same jobs.
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

from .. import process
from ..environment import ENTRY, LAUNCHER, run_in_environment
from ..graph import BLOCKED, TODO, UNNEEDED
from ..stages import BY_NAME
from . import ExecutorError, Report, describe, label, report_stages, say

# The states squeue shows for jobs that are still to run or to finish
ACTIVE_STATES = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED",
                 "REQUEUED", "REQUEUE_HOLD", "REQUEUE_FED", "RESIZING", "SIGNALING",
                 "STAGE_OUT"}
# A record of a submitted job in jobs.tsv: job ID, stage, task, slurm log
RECORDS = "jobs.tsv"
RECORDS_HEADER = "# job\tstage\ttask\tlog\n"
LOCK = ".slurm_pipeline.lock"


def _write_json(path, data):
    """Write data to path as JSON, whole or not at all."""
    partial = "%s.%d.partial" % (path, os.getpid())
    with open(partial, "w") as stream:
        json.dump(data, stream, indent=1)
        stream.write("\n")
    os.replace(partial, path)


def _read_json(path):
    try:
        with open(path) as stream:
            return json.load(stream)
    except (OSError, ValueError) as error:
        raise ExecutorError("can't read %s: %s" % (path, error))


def earlier_jobs(log_dir, directory):
    """{task: job ID} from the earlier submissions in log_dir, the latest
    submission for each task."""
    found = {}
    for path in sorted(glob.glob(os.path.join(glob.escape(log_dir), "pipeline_*", RECORDS))):
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


def active_jobs():
    """{job ID: state} of this user's jobs that are still to run or finishing,
    or None if squeue can't tell."""
    user = os.environ.get("USER") or getpass.getuser()
    try:
        result = subprocess.run(["squeue", "-h", "-u", user, "-o", "%i|%T|%r"],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                universal_newlines=True)
    except OSError:
        return None
    if result.returncode != 0:
        return None
    jobs = {}
    for line in result.stdout.splitlines():
        fields = line.strip().split("|")
        if len(fields) < 2:
            continue
        reason = fields[2] if len(fields) > 2 else ""
        # A job waiting for a job that failed never runs
        if fields[1] in ACTIVE_STATES and reason != "DependencyNeverSatisfied":
            jobs[fields[0]] = fields[1]
    return jobs


def job_name(slurm, task):
    return "%s-%s" % (slurm.get("job_name_prefix") or "fasinate",
                      BY_NAME[task["stage"]].job_name(task))


def log_name(number, task):
    parts = BY_NAME[task["stage"]].log_name(task)
    return "%03d-%s.out" % (number, re.sub(r"[^A-Za-z0-9_.+-]", "_", "-".join(parts)))


def sbatch_command(plan, task, after, log, job_script, extra):
    """The sbatch command of a task, waiting for the jobs in after."""
    slurm = plan["slurm"]
    settings = slurm["stages"][task["stage"]]
    command = ["sbatch", "--parsable", "--job-name=" + job_name(slurm, task),
               # slurm reads % in file names as a pattern
               "--output=" + log.replace("%", "%%"), "--chdir=" + plan["repo_dir"]]
    for option in ("time", "partition", "constraint"):
        if settings.get(option) not in (None, ""):
            command.append("--%s=%s" % (option, settings[option]))
    for args in (slurm.get("sbatch_args"), settings.get("sbatch_args"), extra):
        command += [str(arg) for arg in args or []]
    if after:
        command += ["--dependency=afterok:" + ":".join(after), "--kill-on-invalid-dep=yes"]
    return command + [job_script, task["key"]]


def write_job_script(directory, plan):
    """job.sh KEY: run the task KEY of this submission, in the environment from
    setup.sh, and fail unless it makes its output."""
    lines = [
        "#!/usr/bin/env bash",
        "# Written by ddml-bench (slurm_pipeline.sh) on %s. Every job of this" % process.now(),
        "# submission runs it, as job.sh KEY: it runs the task KEY of tasks.json, in",
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
        "# sourced setup.sh, the task would run in it rather than set up its own, the",
        "# one the jobs were checked in",
        "unset DDML_SOURCE_DIR KEY4HEP_STACK",
        "key=$1",
        "echo \"ddml-bench: job ${SLURM_JOB_ID:-} on $(hostname), $(date '+%F %T'): $key\"",
        "cd \"$repo\"",
        "status=0",
        "%s --bind \"$dir\" -- python3 %s run-task \"$dir\" \"$key\" || status=$?"
        % (shlex.quote(LAUNCHER), shlex.quote(ENTRY)),
        "echo \"ddml-bench: job ${SLURM_JOB_ID:-} finished $(date '+%F %T'), with status $status\"",
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
    plan = _read_json(os.path.join(directory, "plan.json"))
    found = _read_json(os.path.join(directory, "tasks.json"))
    tasks = found["tasks"]
    by_key = dict((task["key"], task) for task in tasks)
    job_script = os.path.join(directory, "job.sh")
    records_path = os.path.join(directory, RECORDS)
    out = Report()

    # Jobs of earlier submissions that are still to run
    earlier = earlier_jobs(plan["slurm"]["log_dir"], directory)
    active = active_jobs()
    if active is None:
        if earlier:
            say("warning: squeue can't tell which jobs of earlier submissions are still queued, "
                "so this may submit jobs again that are queued already")
        active = {}
    queued = {}  # task: job ID
    for task in tasks:
        job = earlier.get(task["key"])
        if task["status"] == TODO and job in active:
            queued[task["key"]] = job

    report_stages(out, tasks, queued, found.get("left_out"))
    # Published files to fetch, on this node before submitting, are only left
    # to do in a dry run (ddml_bench/fetch.py); no job waits for them
    fetching = [task for task in tasks if task["status"] == TODO and task.get("fetch")]
    to_submit = [task for task in tasks if task["status"] == TODO and not task.get("fetch")
                 and task["key"] not in queued]
    if found["problems"]:
        for problem in found["problems"]:
            say(problem)
        if to_submit and not dry_run:
            raise ExecutorError("the environment can't run the jobs (see above), so nothing was "
                                "submitted")
    if not to_submit:
        if fetching:
            out("ddml-bench: nothing to submit once the %d published file(s) are fetched"
                % len(fetching))
        elif queued:
            out("ddml-bench: nothing new to submit; %d job(s) of earlier submissions are still "
                "queued or running" % len(queued))
        elif any(task["status"] == BLOCKED for task in tasks):
            out("ddml-bench: nothing to submit, but some tasks are blocked (see above)")
        else:
            out("ddml-bench: everything selected is done%s, so there is nothing to submit"
                % (" or not needed" if any(task["status"] == UNNEEDED for task in tasks) else ""))
        return 0

    ids = dict(queued)
    submitted = []
    example = None
    out("ddml-bench: %s %d job(s) (job, stage, what, after which jobs):"
        % ("would submit" if dry_run else "submitting", len(to_submit)))
    try:
        for number, task in enumerate(to_submit, 1):
            after = [ids[key] for key in task["after"] if not by_key[key].get("fetch")]
            log = os.path.join(directory, "logs", log_name(number, task))
            command = sbatch_command(plan, task, after, log, job_script, extra)
            if example is None or (after and "--kill-on-invalid-dep=yes" not in example):
                example = command
            if dry_run:
                job = "#%d" % number
            else:
                result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        universal_newlines=True)
                job = result.stdout.strip().split(";")[0]
                if result.returncode != 0 or not job.isdigit():
                    raise ExecutorError("sbatch failed for %s: %s" % (
                        describe(task), (result.stderr or result.stdout).strip()))
                new = not os.path.exists(records_path)
                with open(records_path, "a") as stream:
                    if new:
                        stream.write(RECORDS_HEADER)
                    stream.write("%s\t%s\t%s\t%s\n" % (job, task["stage"], task["key"], log))
            ids[task["key"]] = job
            submitted.append(task)
            out("  %-10s %-12s %s%s" % (job, task["stage"], label(task),
                                        "  after " + ", ".join(after) if after else ""))
    except (ExecutorError, OSError) as error:
        say(str(error))
        if submitted:
            say("the %d job(s) submitted before that are in %s, so running slurm_pipeline.sh "
                "again waits for them instead of submitting them again"
                % (len(submitted), records_path))
        out.save(os.path.join(directory, "report.txt"))
        return 1

    if queued:
        out("ddml-bench: and waiting for %d job(s) of earlier submissions: %s" % (
            len(queued), ", ".join(sorted(set(queued.values()), key=lambda j: (len(j), j)))))
    if dry_run:
        out("ddml-bench: dry run, so nothing was submitted; a job would be submitted as")
        out("  " + process.quote(example))
        if found["problems"]:
            say("but the environment can't run the jobs (see above), so slurm_pipeline.sh "
                "wouldn't submit them")
            return 1
        return 0
    out("ddml-bench: job script, slurm logs (logs/) and the list of jobs (%s) in %s"
        % (RECORDS, directory))
    for task in submitted:
        if task["stage"] == "plot":
            out("ddml-bench: the figures and metrics go to %s" % os.path.dirname(task["output"]))
    out("ddml-bench: follow them with: squeue -u %s -o '%%.12i %%.32j %%.9T %%.11M %%R'"
        % (os.environ.get("USER") or getpass.getuser()))
    out("ddml-bench: run slurm_pipeline.sh again, with the same options, to resubmit whatever "
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
        print("ddml-bench: waiting for another slurm_pipeline.sh to finish submitting")
        fcntl.flock(stream, fcntl.LOCK_EX)
    return stream


def run(plan, dry_run=False, extra=()):
    """Work out what is done in the environment the jobs run in, and submit the
    rest; returns the exit status."""
    log_dir = plan["slurm"].get("log_dir")
    if not log_dir:
        raise ExecutorError("the configuration has no slurm.log_dir")
    if not dry_run and not shutil.which("sbatch"):
        raise ExecutorError("sbatch isn't on PATH; run this on a slurm login node (--dry-run "
                            "shows what it would do)")
    os.makedirs(log_dir, exist_ok=True)
    lock = None
    if dry_run:
        directory = tempfile.mkdtemp(dir=log_dir, prefix=".dry_run.")
    else:
        lock = _lock(log_dir)
        directory = os.path.join(log_dir, "pipeline_%s_%d" % (time.strftime("%Y%m%d_%H%M%S"),
                                                              os.getpid()))
        os.makedirs(directory)
    try:
        os.makedirs(os.path.join(directory, "logs"))
        _write_json(os.path.join(directory, "plan.json"), plan)
        # What each stage has done and has to do, worked out in the environment
        # the jobs run in, which also checks that it can run them
        print("ddml-bench: finding what is done, in the environment the jobs have")
        status = run_in_environment(["tasks", os.path.join(directory, "plan.json"),
                                     os.path.join(directory, "tasks.json")], [directory])
        if status != 0:
            raise ExecutorError("working out the jobs failed (see above)")
        write_job_script(directory, plan)
        return submit(directory, dry_run, extra)
    finally:
        # Keep the folder only if it has jobs
        if dry_run or not os.path.exists(os.path.join(directory, RECORDS)):
            shutil.rmtree(directory, ignore_errors=True)
        if lock is not None:
            lock.close()


def write_tasks(plan_path, tasks_path):
    """The tasks step, in the environment: every task and its status, and what
    in this environment would stop the tasks to do, to tasks_path."""
    from ..graph import Graph
    graph = Graph(_read_json(plan_path)).find_status()
    _write_json(tasks_path, {"events": graph.plan["events"], "tasks": graph.list,
                             "problems": graph.environment_problems(),
                             "left_out": graph.left_out_reasons()})


def run_one(directory, key):
    """In a job: run the task KEY of DIR/tasks.json."""
    from . import run_task
    tasks = _read_json(os.path.join(directory, "tasks.json"))["tasks"]
    task = next((task for task in tasks if task["key"] == key), None)
    if task is None:
        raise ExecutorError("%s/tasks.json has no task %s" % (directory, key))
    print("ddml-bench: %s (%s)" % (describe(task), task["why"] or "to do"))
    run_task(task, check=True)
