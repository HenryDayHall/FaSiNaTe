"""The inputs of the benchmark scenarios: the commands that make them, and the
files those commands make (or, for inputs made elsewhere, the files named).

Each scenario's input.type in the configuration says how its input is made:

  particle_gun   Run ILD_Particle_Gun.py, writing to the scenario's folder of
                 inputs (input.folder of paths.data_dir; by default
                 input_single_particle_gun). With `energies`, it runs once per
                 energy, the i-th with --energy E and --seed (seed + i), so the
                 samples are independent; without, once. Every other key is
                 passed on as --KEY VALUE (a list as several values; null keeps
                 the gun's default).
  di_photon_gun  Run ILD_Di_Photon_Gun.py, which makes pairs of photons from
                 the IP into the ECAL barrel's top stave, in the same way.
  external       Inputs made outside this repository, e.g. the physics samples
                 of ILD's MC production: input.files names the files (LCIO,
                 .slcio), in the scenario's folder of inputs (by default
                 input_physics_samples). Nothing here makes them. One that
                 isn't there is fetched from a published record that has it
                 (ddml_bench/fetch.py); without it, its scenario is left out of
                 the run, and the observables with it (ddml_bench/graph.py).

A scenario with enabled: false is left out of everything; scenario_inputs()
gives it all the same, marked so (the manifests describe its inputs).

Besides its input, a scenario may list simulation_seeds: each input is then
simulated once with each of these ddsim seeds, so that the spread of what is
measured over them can be taken. With one_seed_per_input, each input is
simulated once, with the seed in the same place in the list instead (one seed
for each input, in the order of the inputs). Without seeds, each input is
simulated once, with ddsim's own random seed ([None]). scenario_inputs() gives
each input the seeds it is simulated with.

A new input type needs an entry in INPUT_TYPES: for inputs made here, a
function giving the commands, one giving the file each command makes (without
running it), and the option that makes a command write into a given folder
instead, so that one input can be made in a hidden folder and moved into place
when it is complete; and the Python modules its commands need, whether its
events are a particle gun's (which the reconstruction is told), and the folder
its inputs go in unless the scenario says.

  python3 -m ddml_bench.scenarios [SCENARIO ...]

prints the commands, a comment line naming each scenario before its own, and
for inputs made elsewhere the files and whether they are there.
"""

import argparse
import importlib
import os
import shlex
import sys

from .config import REPO_DIR, ConfigError, load_config
from .layout import INPUT_DIR, PHYSICS_INPUT_DIR, Layout, LayoutError, check_input_folder, \
    check_name

GUN_SCRIPT = os.path.join(REPO_DIR, "ILD_Particle_Gun.py")
DI_PHOTON_SCRIPT = os.path.join(REPO_DIR, "ILD_Di_Photon_Gun.py")
# The inputs a scenario's files can be, when they are made elsewhere: what ddsim
# reads and the manifests publish as inputs
EXTERNAL_ENDINGS = (".slcio",)


class InputType(object):
    def __init__(self, commands=None, files=None, save_dir_option=None, modules=(),
                 particle_gun=False, folder=INPUT_DIR, external=False):
        self.commands = commands
        self.files = files
        self.save_dir_option = save_dir_option
        # The Python modules its commands need, which the generate stage checks
        # the environment for
        self.modules = tuple(modules)
        # Whether its events are a particle gun's, rather than a generator's
        # (the reconstruction is then run with --usingParticleGun)
        self.particle_gun = particle_gun
        # The folder of paths.data_dir its inputs are in, unless a scenario says
        self.folder = folder
        # Whether the generate stage makes its inputs, or they are made elsewhere
        self.made_here = not external

    def inputs(self, name, options, folder):
        """[(file, command)] of a scenario's inputs, in the folder given; the
        command is None for inputs made elsewhere."""
        if not self.made_here:
            return [(path, None) for path in external_files(name, options, folder)]
        commands = self.commands(name, options, folder)
        files = self.files(commands)
        if len(files) != len(commands):
            raise ConfigError("scenario %s: %d commands make %d files"
                              % (name, len(commands), len(files)))
        return list(zip(files, commands))


def gun_commands(script):
    """A function giving the commands that run a gun script for one scenario,
    writing to a folder."""
    def commands(name, options, folder):
        energies = options.pop("energies", None)
        options.setdefault("save_dir", folder)
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


def external_files(name, options, folder):
    """The files of a scenario's inputs made elsewhere (input.files), in the
    folder given."""
    files = options.pop("files", None)
    if options:
        raise ConfigError("scenarios.%s.input has %s, which inputs made elsewhere (type external) "
                          "don't take: they are the files named in input.files as they are"
                          % (name, ", ".join(sorted(options))))
    if not isinstance(files, list) or not files or not all(isinstance(f, str) for f in files):
        raise ConfigError("scenarios.%s.input.files should be a list of the names of the input "
                          "files, in %s" % (name, folder))
    for file_name in files:
        try:
            check_name(file_name, "input file")
        except LayoutError as error:
            raise ConfigError("scenarios.%s.input.files: %s (they are files in the scenario's "
                              "folder, %s)" % (name, error, folder))
        if not file_name.endswith(EXTERNAL_ENDINGS):
            raise ConfigError("scenarios.%s.input.files: %s isn't an LCIO file (%s), which the "
                              "pipeline takes as inputs" % (name, file_name,
                                                            ", ".join(EXTERNAL_ENDINGS)))
    doubled = sorted({f for f in files if files.count(f) > 1})
    if doubled:
        raise ConfigError("scenarios.%s.input.files has %s twice" % (name, ", ".join(doubled)))
    return [os.path.join(folder, file_name) for file_name in files]


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
    "external": InputType(folder=PHYSICS_INPUT_DIR, external=True),
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


def seeds_of_inputs(name, seeds, n_inputs, one_each):
    """The seeds each of a scenario's inputs is simulated with: all of them
    for every input, or with one_each (one_seed_per_input) the i-th seed for
    the i-th input."""
    if not isinstance(one_each, bool):
        raise ConfigError("scenarios.%s.one_seed_per_input should be true or false" % name)
    if not one_each:
        return [list(seeds) for _ in range(n_inputs)]
    if seeds == [None]:
        raise ConfigError("scenarios.%s.one_seed_per_input needs simulation_seeds, one for each "
                          "input" % name)
    if len(seeds) != n_inputs:
        raise ConfigError("scenarios.%s has %d input%s but %d simulation_seeds; with "
                          "one_seed_per_input it needs one seed for each input, in the same order"
                          % (name, n_inputs, "" if n_inputs == 1 else "s", len(seeds)))
    return [[seed] for seed in seeds]


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
    """[{"scenario", "type", "enabled", "folder", "inputs": [{"file", "command",
    "seeds"}], "seeds"}] for every scenario in the configuration, or the ones
    named, whether enabled or not: folder is the folder of paths.data_dir its
    inputs are in, an input's command is None if it is made elsewhere, and its
    seeds are those it is simulated with (the group's seeds are all of them,
    simulation_seeds())."""
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
        settings = scenarios[name]
        if not isinstance(settings, dict) or "input" not in settings:
            raise ConfigError("scenarios.%s has no input section" % name)
        enabled = settings.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ConfigError("scenarios.%s.enabled should be true or false" % name)
        options = dict(settings["input"] or {})
        kind = options.pop("type", None)
        if kind not in INPUT_TYPES:
            raise ConfigError("scenarios.%s.input.type is %r; the known types are: %s"
                              % (name, kind, ", ".join(INPUT_TYPES)))
        folder = options.pop("folder", None) or INPUT_TYPES[kind].folder
        try:
            check_input_folder(folder, "scenarios.%s.input.folder" % name)
        except LayoutError as error:
            raise ConfigError(str(error))
        inputs = INPUT_TYPES[kind].inputs(name, options, layout.inputs(folder))
        seeds = simulation_seeds(name, settings.get("simulation_seeds"))
        each = seeds_of_inputs(name, seeds, len(inputs), settings.get("one_seed_per_input", False))
        result.append({"scenario": name, "type": kind, "enabled": enabled, "folder": folder,
                       "inputs": [{"file": path, "command": command, "seeds": input_seeds}
                                  for (path, command), input_seeds in zip(inputs, each)],
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
        paired = seeds and all(len(entry["seeds"]) == 1 for entry in group["inputs"]) \
            and len(seeds) > 1
        print("# %s (%s)%s%s" % (
            group["scenario"], group["type"],
            ", each input simulated with %s %s" % (
                "its own of the ddsim seeds" if paired else "the ddsim seeds",
                ", ".join(str(s) for s in seeds)) if seeds else "",
            ", disabled (enabled: false)" if not group["enabled"] else ""))
        for entry in group["inputs"]:
            if entry["command"] is None:
                print("# made elsewhere: %s (%s)" % (entry["file"], "there"
                                                     if os.path.isfile(entry["file"])
                                                     else "not there"))
            else:
                print(" ".join(shlex.quote(arg) for arg in entry["command"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
