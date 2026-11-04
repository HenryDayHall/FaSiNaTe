"""As DDML's python/examples/cc3_sf_2a_wrapper.py, which DDML's preset
CC3_BARREL_PY_INTERFACE runs, with the timing tests' stand-in for PyTorch
(tests/fakes/timing/torch): its network at the top level, loaded on the CPU,
called by run_inference() with tensors made on the CPU."""

import os

import numpy as np
import torch

torch.set_num_threads(1)

_model_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../models/CC3_SF_2A.pt")
)
_model = torch.jit.load(_model_path).eval().to("cpu")


@torch.inference_mode()
def run_inference(inputs):
    tensors = [torch.from_numpy(a) for a in inputs]
    out = _model(*tensors)
    return out.detach().cpu().numpy().astype(np.float32, copy=False).ravel()
