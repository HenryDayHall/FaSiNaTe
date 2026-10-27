"""A stand-in for DDML's Python package (python-inference branch), for the
tests: its presets (configs.py) and dataclasses (model.py, geometry.py) as
DDML declares them, and get_fastsim_configuration() and ddml_physics() as
DDML's steering file uses them. ddml_physics() sets nothing up, but the
function it returns has the configurations it was given (configs), for the
fake ddsim to look at."""

import argparse
import sys

from .model import ModelConfig

_PRESETS = {}


def _register_presets():
    from . import configs as _config_mod
    for attr in dir(_config_mod):
        if attr.startswith("_"):
            continue
        val = getattr(_config_mod, attr)
        if isinstance(val, ModelConfig):
            _PRESETS[attr] = val


_register_presets()


def get_fastsim_configuration():
    """As DDML's: the presets named with --ml-model, and --record-calo-entry,
    taken out of sys.argv."""
    _cli = argparse.ArgumentParser(add_help=False)
    _cli.add_argument("--ml-model", action="append", default=None)
    _cli.add_argument("--record-calo-entry", action="store_true", default=False)
    args, remainder = _cli.parse_known_args()
    sys.argv[:] = [sys.argv[0]] + remainder
    presets = []
    for preset in args.ml_model:
        presets.append(_PRESETS[preset])
    return presets, args.record_calo_entry


def ddml_physics(presets, record_calo_entry=False, verbose=True):
    """As DDML's, for SIM.physics.setupUserPhysics(), but setting up nothing."""
    if not presets:
        raise ValueError("user_physics requires at least one ShowerPreset")

    def _setup(kernel):
        pass

    _setup.configs = list(presets)
    _setup.record_calo_entry = record_calo_entry
    return _setup


__all__ = ["ModelConfig", "ddml_physics", "get_fastsim_configuration"]
