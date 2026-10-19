"""The inputs of the benchmark scenarios: the commands that make them, and the
files those commands make.

Each scenario's input.type in the configuration says how its input is made:

  particle_gun  Run ILD_Particle_Gun.py, writing to the inputs folder of the
                layout (input_single_particle_gun in paths.data_dir). With
                `energies`, it runs once per energy, the i-th with --energy E
                and --seed (seed + i), so the samples are independent; without,
                once. Every other key is passed on as --KEY VALUE (a list as
                several values; null keeps the gun's default).

A new input type needs an entry in INPUT_TYPES: a function giving the
commands, one giving the file each command makes (without running it), and
the option that makes a command write into a given folder instead, so that one
input can be made in a hidden folder and moved into place when it is complete.

  python3 -m ddml_bench.scenarios [SCENARIO ...]

prints the commands, a comment line naming each scenario before its own.
"""

import argparse
import os
import shlex
import sys

from .config import REPO_DIR, ConfigError, load_config
from .layout import Layout, check_name, LayoutError

GUN_SCRIPT = os.path.join(REPO_DIR, "ILD_Particle_Gun.py")


class InputType(object):
    def __init__(self, commands, files, save_dir_option):
        self.commands = commands
        self.files = files
        self.save_dir_option = save_dir_option


def particle_gun(name, options, layout):
    """Commands running ILD_Particle_Gun.py for one scenario."""
    energies = options.pop("energies", None)
    options.setdefault("save_dir", layout.input_dir)
    if energies is None:
        runs = [options]
    else:
        seed = options.pop("seed", 0)
        if not isinstance(energies, list) or not energies:
            raise ConfigError("scenarios.%s.input.energies should be a list of energies" % name)
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ConfigError("scenarios.%s.input.seed should be a whole number" % name)
        runs = [dict(options, energy=energy, seed=seed + i)
                for i, energy in enumerate(energies)]
    return [["python3", GUN_SCRIPT] + _flags(run) for run in runs]


def _flags(options):
    args = []
    for key, value in options.items():
        if value is None:
            continue
        args.append("--" + key)
        for item in value if isinstance(value, list) else [value]:
            if isinstance(item, bool):
                args.append("true" if item else "false")
            else:
                args.append(str(item))
    return args


def particle_gun_files(commands):
    """The .slcio files that particle_gun's commands write, as the gun names them.

    ILD_Particle_Gun.py only imports what it needs to generate events (pyLCIO
    and friends) when it does, so this works with any Python.
    """
    if REPO_DIR not in sys.path:
        sys.path.insert(0, REPO_DIR)
    import ILD_Particle_Gun as gun

    parser = gun.get_parser()
    files = []
    for command in commands:
        # python3, the script, then the gun's options
        try:
            args = parser.parse_args(command[2:])
        except SystemExit:
            raise ConfigError("ILD_Particle_Gun.py doesn't accept the options %s"
                              % " ".join(shlex.quote(a) for a in command[2:]))
        files.append(os.path.abspath(gun.get_file_name(args)))
    return files


INPUT_TYPES = {
    "particle_gun": InputType(particle_gun, particle_gun_files, "--save_dir"),
}


def with_option(command, option, value):
    """The command with a single-valued option set to value, in place of any
    value it has."""
    result, skip = [], False
    for arg in command:
        if skip:
            skip = False
        elif arg == option:
            skip = True
        elif not arg.startswith(option + "="):
            result.append(arg)
    return result + [option, value]


def scenario_inputs(config, layout, names=()):
    """[{"scenario", "type", "inputs": [{"file", "command"}]}] for every scenario
    in the configuration, or the ones named."""
    scenarios = config["scenarios"] or {}
    result = []
    for name in names or scenarios:
        if name not in scenarios:
            raise ConfigError("there is no scenario '%s'; the configuration has: %s"
                              % (name, ", ".join(scenarios) or "none"))
        try:
            check_name(name, "scenario")
        except LayoutError as error:
            raise ConfigError(str(error))
        if not isinstance(scenarios[name], dict) or "input" not in scenarios[name]:
            raise ConfigError("scenarios.%s has no input section" % name)
        options = dict(scenarios[name]["input"] or {})
        kind = options.pop("type", None)
        if kind not in INPUT_TYPES:
            raise ConfigError("scenarios.%s.input.type is %r; the known types are: %s"
                              % (name, kind, ", ".join(INPUT_TYPES)))
        commands = INPUT_TYPES[kind].commands(name, options, layout)
        files = INPUT_TYPES[kind].files(commands)
        if len(files) != len(commands):
            raise ConfigError("scenario %s: %d commands make %d files"
                              % (name, len(commands), len(files)))
        result.append({"scenario": name, "type": kind,
                       "inputs": [{"file": path, "command": command}
                                  for path, command in zip(files, commands)]})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Print the commands that make the scenarios' input events.",
        epilog="See the module docstring for the input types.")
    parser.add_argument("scenarios", nargs="*", metavar="SCENARIO",
                        help="only these scenarios (default: all of them)")
    args = parser.parse_args(argv)
    try:
        config = load_config()
        groups = scenario_inputs(config, Layout(config["paths"]["data_dir"]), args.scenarios)
    except ConfigError as error:
        print("scenarios: %s" % error, file=sys.stderr)
        return 1
    for group in groups:
        print("# %s (%s)" % (group["scenario"], group["type"]))
        for entry in group["inputs"]:
            print(" ".join(shlex.quote(arg) for arg in entry["command"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
