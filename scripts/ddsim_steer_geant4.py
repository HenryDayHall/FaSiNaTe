"""ddsim steering file for full Geant4, the reference the models are compared to.

It runs DDML's own steering file, scripts/ddsim_steer.py in the DDML build that
setup.sh sets up (the one simulate_models.sh uses), with a stand-in for DDML's
Python package that sets up no fast simulation. So every other setting (the
physics list, the crossing-angle boost, the range cut, the filters, ...) is the
same as when the models are simulated, and if DDML's steering file starts to use
DDML in some other way, this fails rather than simulate something different.

ddsim runs a steering file with exec() and separate global and local
namespaces, so nothing defined in functions here may use the names defined at
its top level.
"""

import os
import runpy
import sys
import types

_ddml = types.ModuleType("ddml")
_ddml.__doc__ = "Stand-in for DDML's Python package, with no fast simulation"
# No presets, whatever the command line says, and no recording at the
# calorimeter entry
_ddml.get_fastsim_configuration = lambda: ([], False)
# A physics setup that adds nothing, instead of DDML's fast simulation
_ddml.ddml_physics = lambda presets, record_calo_entry=False, verbose=True: (lambda kernel: None)

if not os.environ.get("DDML_SOURCE_DIR"):
    raise RuntimeError("DDML_SOURCE_DIR isn't set: source setup.sh first")
_steering = os.path.join(os.environ["DDML_SOURCE_DIR"], "scripts", "ddsim_steer.py")

_real_ddml = sys.modules.get("ddml")
sys.modules["ddml"] = _ddml
try:
    SIM = runpy.run_path(_steering)["SIM"]
finally:
    if _real_ddml is None:
        del sys.modules["ddml"]
    else:
        sys.modules["ddml"] = _real_ddml
