"""A stand-in for PyTorch, for the timing tests (tests/test_timing.py puts it on
the Python path of the environment they run in): tensors that know the device
they are on, modules that refuse tensors from another device, as PyTorch does,
and a GPU, as there would be on a GPU node.

Its CUDA support follows files in $FAKE_CONTROL: with cuda_not_built, it was
built without CUDA; with cuda_available, it finds a GPU, and otherwise none.
torch.jit.load() gives a module that makes an output as DDML's
CaloCloudsTwoAngleModel reads it (30 layers, then (x, y, layer, energy) per
point).
"""

import os

__version__ = "2.99.0+fake"


def _control(name):
    return os.path.exists(os.path.join(os.environ.get("FAKE_CONTROL", "/nonexistent"), name))


class version(object):
    cuda = None if _control("cuda_not_built") else "12.9"


class Tensor(object):
    def __init__(self, values, device="cpu"):
        self.values = [float(v) for v in values]
        self.device = str(device)

    def to(self, device):
        return Tensor(self.values, device)

    def cpu(self):
        return self.to("cpu")

    def detach(self):
        return self

    def numpy(self):
        if self.device != "cpu":
            raise TypeError("can't convert %s device type tensor to numpy; use Tensor.cpu() first"
                            % self.device)
        import numpy
        return numpy.array(self.values, dtype=numpy.float32)


def is_tensor(value):
    return isinstance(value, Tensor)


def from_numpy(array):
    return Tensor(array.ravel().tolist())


def set_num_threads(number):
    pass


class _NoGrad(object):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __call__(self, function):
        return function


def inference_mode():
    return _NoGrad()


class nn(object):
    class Module(object):
        device = "cpu"

        def to(self, device):
            self.device = str(device)
            return self

        def eval(self):
            return self

        def __call__(self, *args, **kwargs):
            for value in list(args) + list(kwargs.values()):
                if is_tensor(value) and value.device != self.device:
                    raise RuntimeError("Expected all tensors to be on the same device, but found "
                                       "at least two devices, %s and %s!" % (self.device,
                                                                            value.device))
            return self.forward(*args, **kwargs)


class _Shower(nn.Module):
    """A network making one point in the first layer, of the photon's energy."""

    def forward(self, energy, theta, phi):
        layers = [1.0] + [0.0] * 29
        return Tensor(layers + [0.0, 0.0, 0.0, energy.values[0]], self.device)


class jit(object):
    @staticmethod
    def load(path):
        if not os.path.exists(path):
            raise RuntimeError("open file failed because of errno 2 on fopen: %s" % path)
        return _Shower()


class backends(object):
    class cuda(object):
        @staticmethod
        def is_built():
            return not _control("cuda_not_built")


class cuda(object):
    @staticmethod
    def is_available():
        return _control("cuda_available") and not _control("cuda_not_built")

    @staticmethod
    def current_device():
        return 0

    @staticmethod
    def get_device_name(index=0):
        return "Fake A100-PCIE-80GB"

    @staticmethod
    def synchronize():
        pass
