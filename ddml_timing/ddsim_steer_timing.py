"""ddsim steering file for the timing runs (ddml_timing/runner.py):

  ddsim --steeringFile ddml_timing/ddsim_steer_timing.py <the simulator's options>

with FASINATE_TIMED_STEERING the steering file the pipeline runs the simulator
with: scripts/ddsim_steer_geant4.py for Geant4, DDML's scripts/ddsim_steer.py
for a model with DDML presets, scripts/ddsim_steer_models.py for one with DDML
model configurations written out. The rest of ddsim's command line is the
pipeline's too (ddml_bench/stages/simulate.py), so the simulation is the
pipeline's in every setting.

Before running that steering file, it sets up the timing of the models'
embedded-Python inference, and its running on the GPU where the run asks for
it (ddml_timing/inference.py, from FASINATE_TIMED_INFERENCE and
FASINATE_TIMED_PYTHON_MODULES), which stops ddsim here if that can't be done.

ddsim runs a steering file with exec() and separate global and local
namespaces, so nothing defined in functions here may use the names defined at
its top level.
"""

import os
import runpy
import sys

# This repository, for ddml_timing: where this file is (ddsim compiles it under
# its own path), or else FASINATE_REPO_DIR, which setup.sh sets
_here = os.path.dirname(os.path.dirname(os.path.abspath(sys._getframe().f_code.co_filename)))
if not os.path.isfile(os.path.join(_here, "ddml_timing", "inference.py")):
    _here = os.environ.get("FASINATE_REPO_DIR", "")
if not os.path.isfile(os.path.join(_here, "ddml_timing", "inference.py")):
    raise RuntimeError("can't find FaSiNaTe's ddml_timing: source setup.sh first")
if _here not in sys.path:
    sys.path.insert(0, _here)
_steering = os.environ.get("FASINATE_TIMED_STEERING")
if not _steering or not os.path.isfile(_steering):
    raise RuntimeError("ddsim_steer_timing.py: FASINATE_TIMED_STEERING should be the simulator's "
                       "own steering file, not %r" % _steering)

from ddml_timing import inference as _inference  # noqa: E402

_inference.prepare()
SIM = runpy.run_path(_steering)["SIM"]
