"""CaloClouds3 with the energy of its showers rescaled: the inference code of
models/caloclouds3_rescaled.yaml, which DDML's embedded-Python plugin runs in
ddsim. The model's DDML model configuration names this module
(plugin_properties PythonModule), and scripts/ddsim_steer_models.py puts
models/ first on ddsim's Python path, so the plugin imports it from here.

It runs the model as DDML's own module for CaloClouds3 does
(python/examples/cc3_sf_2a_wrapper.py in DDML), keeping DDML's contract for
embedded Python: run_inference(inputs) is given a list of float32 numpy
arrays, the photon's energy in GeV and its polar and azimuthal angles in
degrees in the calorimeter's local frame, each of shape (1, 1), and returns
the model's output as one flat float32 array. DDML's CaloCloudsTwoAngleModel
reads that as the number of points in each of the 30 layers, then each point
as (x, y, layer, energy) (convertOutput() in its src/CaloCloudsTwoAngleModel.cc).
The energy of every point is multiplied by the description's energy_rescale.

What it runs with comes from the model's description beside it
(models/caloclouds3_rescaled.yaml), so it needs no settings of its own: the
weights, ddml.weights, relative to the DDML clone ($DDML_SOURCE_DIR, which
setup.sh sets), the file whose md5 sum simulate_models.sh checks, and the
factor, energy_rescale.
"""

import os

import numpy as np
import torch
import yaml

torch.set_num_threads(1)

# The output, as DDML's CaloCloudsTwoAngleModel reads it: the number of points
# in each layer (m_nLayer), then each point as (x, y, layer, energy)
N_LAYERS = 30
POINT_SIZE = 4
ENERGY = 3  # where in a point its energy is

_description_path = os.path.splitext(os.path.abspath(__file__))[0] + ".yaml"
with open(_description_path) as _stream:
    _description = yaml.safe_load(_stream)
_energy_rescale = float(_description["energy_rescale"])
_ddml_source = os.environ.get("DDML_SOURCE_DIR")
if not _ddml_source:
    raise RuntimeError("DDML_SOURCE_DIR isn't set, so the weights of %s can't be found: "
                       "source setup.sh first" % _description_path)
_model_path = os.path.join(_ddml_source, _description["ddml"]["weights"])
_model = torch.jit.load(_model_path).eval().to("cpu")


def rescale_energies(out, factor):
    """Multiply the energy of every point of an output (a flat float32 array,
    changed in place) by factor; returns it. Past its points, an output that is
    longer than they need holds only what DDML doesn't read."""
    out[N_LAYERS + ENERGY::POINT_SIZE] *= factor
    return out


@torch.inference_mode()
def run_inference(inputs):
    # inputs is a list of 3 float32 numpy arrays: energy, theta_deg, phi_deg.
    # torch.from_numpy aliases the C++ buffer — do NOT mutate the arrays.
    tensors = [torch.from_numpy(a) for a in inputs]
    out = _model(*tensors)

    # A contiguous flat float32 array, which the C++ side copies into its
    # pre-allocated std::vector<float>
    out = out.detach().cpu().numpy().astype(np.float32, copy=False).ravel()
    return rescale_energies(out, _energy_rescale)
