"""
Runs a model 
created with PotinCountFM https://github.com/HenryDayHall/PointCountFM/tree/CC3_integration
and Calocloudi https://github.com/HenryDayHall/CaloCloudi/tree/pcfm-unified-sampler
TODO, currently this thing clusters cells internally. Remove that.
"""
import os

import numpy as np
import torch
import yaml

torch.set_num_threads(1)

# The output, as DDML's CaloCloudsTwoAngleModel reads it: the number of points
# in each layer (m_nLayer), then each point as (x, y, layer, energy)
N_LAYERS = 78
POINT_SIZE = 4
ENERGY = 3  # where in a point its energy is

_description_path = os.path.splitext(os.path.abspath(__file__))[0] + ".yaml"
with open(_description_path) as _stream:
    _description = yaml.safe_load(_stream)
# these weights are local
_repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_model_path = os.path.join(_repo_dir, _description["ddml"]["weights"])
_model = torch.jit.load(_model_path).eval().to("cpu")


@torch.inference_mode()
def run_inference(inputs):
    # inputs is a list of 3 float32 numpy arrays: energy, theta_deg, phi_deg.
    # torch.from_numpy aliases the C++ buffer — do NOT mutate the arrays.
    tensors = [torch.from_numpy(a) for a in inputs]
    # get the inputs into the format this compile expects them in
    pdgs = torch.full(len(tensors[0]), 22)
    energy = tensors[0]
    theta_deg = tensors[1]
    phi_deg = tensors[2]
    direction = torch.concatenate(
        [
            torch.sin(theta_deg) * torch.cos(phi_deg),
            torch.sin(theta_deg) * torch.sin(phi_deg),
            torch.cos(theta_deg),
        ],
        dim=-1,
    )

    out = _model(energy, direction, pdgs)
    points_per_layer = out['points_per_layer'][0]
    points = out['points'][0]
    real_points = points[points[:, ENERGY] > 0] 

    full_report = np.concatenate([points_per_layer, real_points.ravel()])

    # A contiguous flat float32 array, which the C++ side copies into its
    # pre-allocated std::vector<float>
    full_report = full_report.astype(np.float32, copy=False).ravel()
    return full_report
