"""ddsim steering file for a model described by DDML model configurations
(ddml.model_configs in models/<name>.yaml) rather than DDML's presets alone:

  ddsim --steeringFile scripts/ddsim_steer_models.py --ml-config JSON ...

It runs DDML's own steering file, scripts/ddsim_steer.py in the DDML build that
setup.sh sets up (the one the models with presets run), with DDML's own Python
package, so every setting (the physics list, the crossing-angle boost, the
range cut, the filters, ...) is the same, and the fast simulation is set up by
DDML's ddml_physics() as for a preset. Only the configurations it is given
differ: DDML's get_fastsim_configuration(), which makes them of the presets
named with --ml-model, is replaced by ddml_bench.ddml_configs.from_command_line()
and build(), which make them of the --ml-config options, each a JSON object, a
DDML preset to start from and the fields of DDML's ModelConfig to set
(ddml_bench/ddml_configs.py says how), in the order given. --ml-model PRESET
and --record-calo-entry still work as with DDML's steering file. The
configurations made are printed, so the log says what ran.

ddsim runs a steering file with exec() and separate global and local
namespaces, so nothing defined in functions here may use the names defined at
its top level.
"""

import os
import runpy
import sys

# This repository, for ddml_bench: where this file is (ddsim compiles it under
# its own path), or else FASINATE_REPO_DIR, which setup.sh sets
_here = os.path.dirname(os.path.dirname(os.path.abspath(sys._getframe().f_code.co_filename)))
if not os.path.isfile(os.path.join(_here, "ddml_bench", "ddml_configs.py")):
    _here = os.environ.get("FASINATE_REPO_DIR", "")
if not os.path.isfile(os.path.join(_here, "ddml_bench", "ddml_configs.py")):
    raise RuntimeError("can't find FaSiNaTe's ddml_bench: source setup.sh first")
if _here not in sys.path:
    sys.path.insert(0, _here)
if not os.environ.get("DDML_SOURCE_DIR"):
    raise RuntimeError("DDML_SOURCE_DIR isn't set: source setup.sh first")
_steering = os.path.join(os.environ["DDML_SOURCE_DIR"], "scripts", "ddsim_steer.py")

import ddml  # noqa: E402  DDML's own package, the one its steering file imports


def _fastsim_configuration():
    """As DDML's get_fastsim_configuration(): the configurations to set up and
    whether to record the particles at the calorimeter entry, taking the
    options out of the command line before ddsim reads it."""
    import sys
    from ddml_bench import ddml_configs
    entries, record_calo_entry = ddml_configs.from_command_line(sys.argv)
    if not entries:
        raise RuntimeError("ddsim_steer_models.py: give the model's configurations with "
                           "%s (or DDML's presets with --ml-model)" % ddml_configs.OPTION)
    configs = ddml_configs.build(entries)
    for number, (entry, config) in enumerate(zip(entries, configs), 1):
        print("ddsim_steer_models.py: model configuration %d: %s" % (
            number, ddml_configs.option(entry)))
        print("ddsim_steer_models.py:   %s" % ddml_configs.describe(config))
    sys.stdout.flush()
    return configs, record_calo_entry


_real = ddml.get_fastsim_configuration
ddml.get_fastsim_configuration = _fastsim_configuration
try:
    SIM = runpy.run_path(_steering)["SIM"]
finally:
    ddml.get_fastsim_configuration = _real
