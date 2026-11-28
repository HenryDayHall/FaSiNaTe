"""A stand-in for PyTorch, for the tests: just what a model's own inference
code (models/*.py) uses as it is imported, so that the preflight can import it
(ddml_bench/preflight.py, step 4) without PyTorch: torch.jit.load() of any
file gives a model returning zeros."""

import os


def set_num_threads(threads):
    pass


def inference_mode():
    return lambda function: function


def from_numpy(array):
    return array


class _Model(object):
    def __init__(self, path):
        self.path = path

    def eval(self):
        return self

    def to(self, device):
        return self

    def __call__(self, *inputs):
        import numpy
        return numpy.zeros(30, dtype=numpy.float32)


class _Jit(object):
    @staticmethod
    def load(path, map_location=None):
        if not os.path.isfile(path):
            raise RuntimeError("open file failed because of errno 2 on fopen: %s" % path)
        return _Model(path)


jit = _Jit()
