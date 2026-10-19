"""Running the programs a task runs (ddsim, k4run, the particle gun), with
their output in a log and what they were making removed if they are stopped.
"""

import os
import shlex
import signal
import socket
import subprocess
import sys
import time
from contextlib import contextmanager


class Interrupted(Exception):
    """SIGINT or SIGTERM (Ctrl-C, scancel, or a slurm time limit) arrived."""

    def __init__(self, signum):
        Exception.__init__(self, "interrupted by signal %d" % signum)
        self.signum = signum
        # The exit status of a process killed by the signal, as bash reports it
        self.status = 128 + signum


@contextmanager
def interruptible():
    """Within this, SIGINT and SIGTERM raise Interrupted, so that whatever a
    task was making can be removed on the way out."""
    def handler(signum, frame):
        raise Interrupted(signum)

    previous = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.signal(signum, handler)
    try:
        yield
    finally:
        for signum, old in previous.items():
            signal.signal(signum, old)


def host():
    """This machine's short name."""
    return socket.gethostname().split(".")[0]


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def quote(command):
    return " ".join(shlex.quote(str(arg)) for arg in command)


def child_environment(**settings):
    """This environment, with these variables set: one thread, as in the
    paper's timings, and Python's output in step with the rest of the log."""
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    env.update(settings)
    return env


def run(command, cwd=None, stdout=None, env=None):
    """Run command in cwd with no input, its output going to stdout (a file
    object; this process's own by default); return its exit status. If
    Interrupted, stop it, then raise again."""
    sys.stdout.flush()
    sys.stderr.flush()
    process = subprocess.Popen(command, cwd=cwd, stdin=subprocess.DEVNULL, stdout=stdout,
                               stderr=subprocess.STDOUT if stdout is not None else None,
                               env=env if env is not None else child_environment())
    try:
        return process.wait()
    except Interrupted:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise


def run_logged(command, cwd, log, env=None):
    """run(), with the output added to the file log."""
    with open(log, "a") as stream:
        return run(command, cwd=cwd, stdout=stream, env=env)


def log_tail(log, lines=20):
    """The end of a log, indented, for an error message."""
    try:
        with open(log, errors="replace") as stream:
            text = stream.read().splitlines()[-lines:]
    except OSError:
        return ""
    return "\n".join("    " + line for line in text)


def append(log, line):
    try:
        with open(log, "a") as stream:
            stream.write(line + "\n")
    except OSError:
        pass


def remove(*paths):
    for path in paths:
        try:
            os.remove(path)
        except OSError:
            pass
