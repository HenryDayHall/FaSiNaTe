import os
import yaml
import numpy as np
import torch

torch.set_num_threads(1)

# get the configuration from the environment
_software_path = os.environ['FASINATE_SOFTWARE_PATH']
_model_config_path = os.environ['FASINATE_MODEL_CONFIG']
with open(model_config_path) as f:
    _model_config = yaml.safe_load(f)

_model_path = os.path.join(_software_path, "DDML", _model_config['ddml']['weights'])
_model = torch.jit.load(_model_path).eval().to("cpu")
_energy_rescale = _model_config['energy_rescale']


@torch.inference_mode()
def run_inference(inputs):
    # inputs is a list of 3 float32 numpy arrays: energy, theta_deg, phi_deg.
    # torch.from_numpy aliases the C++ buffer — do NOT mutate the arrays.
    tensors = [torch.from_numpy(a) for a in inputs]
    out = _model(*tensors)

    # Return a contiguous flat float32 view; the C++ side memcpy's into
    # the pre-allocated std::vector<float>.
    out = out.detach().cpu().numpy().astype(np.float32, copy=False).ravel()

    n_layers = 30
    energy_mask = np.zeros(len(out), dtype=bool)
    energy_mask[n_layers::4] = True
    out[energy_mask] *= _energy_rescale

    return out
