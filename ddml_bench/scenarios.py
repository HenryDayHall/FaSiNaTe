"""The inputs of the benchmark scenarios: the commands that make them, and the
files those commands make.

Each scenario's input.type in the configuration says how its input is made:

  particle_gun   Run ILD_Particle_Gun.py, writing to the inputs folder of the
                 layout (input_single_particle_gun in paths.data_dir). With
                 `energies`, it runs once per energy, the i-th with --energy E
                 and --seed (seed + i), so the samples are independent;
                 without, once. Every other key is passed on as --KEY VALUE (a
                 list as several values; null keeps the gun's default).
  di_photon_gun  Run ILD_Di_Photon_Gun.py, which makes pairs of photons from
                 the IP into the ECAL barrel's top stave, in the same way.

Besides its input, a scenario may list simulation_seeds: each input is then
simulated once with each of these ddsim seeds (scenario_inputs() gives them as
the group's seeds), so that the spread of what is measured over them can be
taken. Without them, each input is simulated once, with ddsim's own random
seed ([None]).

A new input type needs an entry in INPUT_TYPES: a function giving the
commands, one giving the file each command makes (without running it), and
the option that makes a command write into a given folder instead, so that one
input can be made in a hidden folder and moved into place when it is complete;
and the Python modules its commands need, and whether its events are a
particle gun's (which the reconstruction is told).

  python3 -m ddml_bench.scenarios [SCENARIO ...]

prints the commands, a comment line naming each scenario before its own.
"""

import argparse
import importlib
import os
import shlex
import sys

from .config import REPO_DIR, ConfigError, load_config
from .layout import Layout, check_name, LayoutError

GUN_SCRIPT = os.path.join(REPO_DIR, "ILD_Particle_Gun.py")
DI_PHOTON_SCRIPT = os.path.join(REPO_DIR, "ILD_Di_Photon_Gun.py")


class InputType(object):
    def __init__(self, commands, files, save_dir_option, modules=(), particle_gun=False):
        self.commands = commands
        self.files = files
        self.save_dir_option = save_dir_option
        # The Python modules its commands need, which the generate stage checks
        # the environment for
        self.modules = tuple(modules)
        # Whether its events are a particle gun's, rather than a generator's
        # (the reconstruction is then run with --usingParticleGun)
        self.particle_gun = particle_gun


def gun_commands(script):
    """A function giving the commands that run a gun script for one scenario."""
    def commands(name, options, layout):
        energies = options.pop("energies", None)
        options.setdefault("save_dir", layout.input_dir)
        if energies is None:
            runs = [options]
        else:
            seed = options.pop("seed", 0)
            if not isinstance(energies, list) or not energies:
                raise ConfigError("scenarios.%s.input.energies should be a list of energies"
                                  % name)
            if not isinstance(seed, int) or isinstance(seed, bool):
                raise ConfigError("scenarios.%s.input.seed should be a whole number" % name)
            runs = [dict(options, energy=energy, seed=seed + i)
                    for i, energy in enumerate(energies)]
        return [["python3", script] + _flags(run) for run in runs]
    commands.__doc__ = "Commands running %s for one scenario." % os.path.basename(script)
    return commands


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


def gun_files(module):
    """A function giving the .slcio files that a gun script's commands write,
    as the script (the module of that name) names them.

    The guns only import what they need to generate events (pyLCIO and
    friends) when they do, so this works with any Python.
    """
    def files(commands):
        if REPO_DIR not in sys.path:
            sys.path.insert(0, REPO_DIR)
        gun = importlib.import_module(module)
        parser = gun.get_parser()
        found = []
        for command in commands:
            # python3, the script, then the gun's options
            try:
                args = parser.parse_args(command[2:])
            except SystemExit:
                raise ConfigError("%s.py doesn't accept the options %s"
                                  % (module, " ".join(shlex.quote(a) for a in command[2:])))
            found.append(os.path.abspath(gun.get_file_name(args)))
        return found
    files.__doc__ = "The .slcio files that %s.py's commands write." % module
    return files


particle_gun = gun_commands(GUN_SCRIPT)
particle_gun_files = gun_files("ILD_Particle_Gun")
di_photon_gun = gun_commands(DI_PHOTON_SCRIPT)
di_photon_gun_files = gun_files("ILD_Di_Photon_Gun")

INPUT_TYPES = {
    "particle_gun": InputType(particle_gun, particle_gun_files, "--save_dir",
                              modules=("numpy", "pyLCIO", "h5py", "particle"),
                              particle_gun=True),
    "di_photon_gun": InputType(di_photon_gun, di_photon_gun_files, "--save_dir",
                               modules=("numpy", "pyLCIO", "h5py"), particle_gun=True),
}


def simulation_seeds(name, seeds):
    """The ddsim seeds each input of a scenario is simulated with, once each:
    its simulation_seeds, or [None] (once, with ddsim's own random seed)."""
    if seeds is None:
        return [None]
    if not isinstance(seeds, list) or not seeds or not all(
            isinstance(s, int) and not isinstance(s, bool) and s >= 1 for s in seeds):
        raise ConfigError("scenarios.%s.simulation_seeds should be a list of ddsim's random "
                          "seeds, whole numbers of at least 1 (e.g. [1, 2, 3]), or null" % name)
    if len(set(seeds)) != len(seeds):
        raise ConfigError("scenarios.%s.simulation_seeds has a seed twice: %s"
                          % (name, ", ".join(str(s) for s in seeds)))
    return list(seeds)


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
    """[{"scenario", "type", "inputs": [{"file", "command"}], "seeds"}] for every
    scenario in the configuration, or the ones named; seeds are those each input
    is simulated with (simulation_seeds())."""
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
        seeds = simulation_seeds(name, scenarios[name].get("simulation_seeds"))
        result.append({"scenario": name, "type": kind,
                       "inputs": [{"file": path, "command": command}
                                  for path, command in zip(files, commands)],
                       "seeds": seeds})
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
        seeds = [s for s in group["seeds"] if s is not None]
        print("# %s (%s)%s" % (group["scenario"], group["type"],
                               ", each simulated with the ddsim seeds %s"
                               % ", ".join(str(s) for s in seeds) if seeds else ""))
        for entry in group["inputs"]:
            print(" ".join(shlex.quote(arg) for arg in entry["command"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
