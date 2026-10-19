"""Getting into the environment the tasks run in: the Key4hep release and DDML
build from setup.sh, in a clean shell, inside the container unless
container.enabled is false. scripts/run_in_environment.sh does it, building
DDML (and what else setup.sh needs) first if needed.
"""

import os

from . import process
from .config import REPO_DIR

LAUNCHER = os.path.join(REPO_DIR, "scripts", "run_in_environment.sh")
# The command line of the harness, which the launcher runs in there
ENTRY = os.path.join(REPO_DIR, "scripts", "ddml-bench")


def in_environment():
    """Whether this process runs in the environment from setup.sh already."""
    return bool(os.environ.get("DDML_SOURCE_DIR")) and bool(os.environ.get("KEY4HEP_STACK"))


def launcher_command(args, binds=()):
    """The command running `ddml-bench ARGS` in the environment, with the
    folders binds mounted in the container too (it mounts the repository, the
    data folder, the software and the configuration files already)."""
    command = [LAUNCHER]
    for folder in binds:
        command += ["--bind", folder]
    return command + ["--", "python3", ENTRY] + list(args)


def run_in_environment(args, binds=()):
    """Run `ddml-bench ARGS` in the environment; return its exit status."""
    with process.interruptible():
        return process.run(launcher_command(args, binds), env=dict(os.environ))
