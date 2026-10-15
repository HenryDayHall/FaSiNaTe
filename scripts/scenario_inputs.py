#!/usr/bin/env python3
"""Print the commands that make the input events of the benchmark scenarios.

generate_input.sh runs them in the Key4hep environment. This script only reads
the configuration, so it runs anywhere fasinate_config.py does.

  scenario_inputs.py [SCENARIO ...]

For every scenario in the configuration, or just the ones named, it prints a
comment line with the scenario's name, then one shell command per line. Each
scenario's input.type says how its input is made:

  particle_gun  Run ILD_Particle_Gun.py, which writes to
                input_single_particle_gun in paths.data_dir. With `energies`,
                it runs once per energy, the i-th with --energy E and --seed
                (seed + i), so the samples are independent; without, once.
                Every other key is passed on as --KEY VALUE (a list as several
                values; null keeps the gun's default).

New input types go in GENERATORS.
"""

import argparse
import os
import shlex
import sys

from fasinate_config import REPO_DIR, ConfigError, load_config


def particle_gun(name, options):
    """Commands running ILD_Particle_Gun.py for one scenario."""
    energies = options.pop("energies", None)
    if energies is None:
        runs = [options]
    else:
        seed = options.pop("seed", 0)
        if not isinstance(energies, list) or not energies:
            raise ConfigError("scenarios.%s.input.energies should be a list of energies" % name)
        if not isinstance(seed, int):
            raise ConfigError("scenarios.%s.input.seed should be a whole number" % name)
        runs = [dict(options, energy=energy, seed=seed + i)
                for i, energy in enumerate(energies)]
    script = os.path.join(REPO_DIR, "ILD_Particle_Gun.py")
    return [["python3", script] + _flags(run) for run in runs]


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


GENERATORS = {"particle_gun": particle_gun}


def input_commands(config, names=()):
    """Lines of shell: a comment naming each scenario, then its commands."""
    scenarios = config["scenarios"]
    lines = []
    for name in names or scenarios:
        if name not in scenarios:
            raise ConfigError("there is no scenario '%s'; the configuration has: %s"
                              % (name, ", ".join(scenarios) or "none"))
        if "input" not in scenarios[name]:
            raise ConfigError("scenarios.%s has no input section" % name)
        options = dict(scenarios[name]["input"])
        kind = options.pop("type", None)
        if kind not in GENERATORS:
            raise ConfigError("scenarios.%s.input.type is %r; the known types are: %s"
                              % (name, kind, ", ".join(GENERATORS)))
        lines.append("# %s (%s)" % (name, kind))
        for command in GENERATORS[kind](name, options):
            lines.append(" ".join(shlex.quote(arg) for arg in command))
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Print the commands that make the scenarios' input events.",
        epilog="See the module docstring for the input types.")
    parser.add_argument("scenarios", nargs="*", metavar="SCENARIO",
                        help="only these scenarios (default: all of them)")
    args = parser.parse_args(argv)
    try:
        lines = input_commands(load_config(), args.scenarios)
    except ConfigError as error:
        print("scenario_inputs: %s" % error, file=sys.stderr)
        return 1
    for line in lines:
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
