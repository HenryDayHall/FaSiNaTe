"""A stand-in for DDML's embedded-Python CaloClouds3 module, for the tests."""


def run_inference(inputs):
    raise RuntimeError("the tests' fake ddsim never calls this")
