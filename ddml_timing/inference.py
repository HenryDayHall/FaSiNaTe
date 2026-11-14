"""Inside ddsim, for a timing run (ddsim_steer_timing.py runs prepare() before
the simulator's own steering file): the models' embedded-Python inference,
each call timed, and run on the GPU where the run asks for it.

DDML's embedded-Python plugins import a module (their PythonModule) in ddsim's
own interpreter and call its run_inference(inputs) for every shower, as
DDML's python/examples/cc3_sf_2a_wrapper.py and FaSiNaTe's
models/caloclouds3_rescaled.py do. Both keep their network as a torch module
at the top level of the module (_model), which run_inference calls with
tensors made on the CPU, then takes the result back to numpy. So prepare()
imports each module first, as the plugin will (it then gets the same module),
and puts in place of each of its top-level torch modules one that stands in
for it (Timed): on the GPU, it moves the network there, its tensor arguments
there for each call, and the result back, so that run_inference works as
before; and on either, it prints how long each call took, a line

  ddml-timing: inference <module>.<name> <seconds>

which the runner adds up event by event. A call on the GPU is timed until the
result is back on the CPU, so it includes the copies. Nothing in the module
changes but that name, and nothing about ddsim's other settings.

A module without a torch module at its top level can't be run on the GPU this
way: prepare() then stops the run, saying why. On the CPU its calls just aren't
timed, nor where PyTorch can't be imported (on the GPU, that stops the run).

The settings come from the runner, in the environment:

  FASINATE_TIMED_INFERENCE       cpu, or cuda for the GPU
  FASINATE_TIMED_PYTHON_MODULES  the modules, separated by commas
  FASINATE_TIMED_MODULE_PATH     a folder to put first on the Python path, as
                                 scripts/ddsim_steer_models.py puts models/
                                 (for a model with its own inference code)
"""

import importlib
import os
import sys
import time

PREFIX = "ddml-timing:"


def say(message):
    sys.stdout.write("%s %s\n" % (PREFIX, message))
    sys.stdout.flush()


def _moved(value, device, torch):
    """value with its tensors on device (tuples, lists and dicts of them too)."""
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, (tuple, list)):
        return type(value)(_moved(item, device, torch) for item in value)
    if isinstance(value, dict):
        return dict((key, _moved(item, device, torch)) for key, item in value.items())
    return value


class Timed(object):
    """Stands in for a model's torch module: runs it on device (moving it
    there, its tensor arguments there for each call, and its result back to
    the CPU), and prints how long each call takes."""

    def __init__(self, module, device, label, torch):
        self._torch = torch
        self._device = device
        self._move = device != "cpu"
        self._label = label
        self._module = module.to(device) if self._move else module

    def __call__(self, *args, **kwargs):
        torch = self._torch
        start = time.perf_counter()
        if self._move:
            args = _moved(args, self._device, torch)
            kwargs = _moved(kwargs, self._device, torch)
        result = self._module(*args, **kwargs)
        if self._move:
            # Back on the CPU, which waits for the GPU to finish
            result = _moved(result, "cpu", torch)
        elapsed = time.perf_counter() - start
        sys.stdout.write("%s inference %s %.9f\n" % (PREFIX, self._label, elapsed))
        sys.stdout.flush()
        return result

    def __getattr__(self, name):
        return getattr(self._module, name)


def cuda_problem(torch):
    """Why this torch can't run on a GPU here, or None."""
    backends = getattr(torch, "backends", None)
    cuda = getattr(backends, "cuda", None)
    if cuda is None or not cuda.is_built():
        return ("PyTorch %s was built without CUDA (torch.version.cuda is %s), so it can't run "
                "on a GPU" % (torch.__version__, getattr(torch.version, "cuda", None)))
    if not torch.cuda.is_available():
        return ("PyTorch %s (CUDA %s) finds no GPU it can use: CUDA_VISIBLE_DEVICES is %s; is "
                "this a GPU node, with the GPU passed into the container (--nv)?"
                % (torch.__version__, torch.version.cuda,
                   os.environ.get("CUDA_VISIBLE_DEVICES", "unset")))
    return None


def wrap(module, torch, device, name=None):
    """Put Timed in place of each torch module at the top level of module;
    returns the names it replaced."""
    name = name or module.__name__
    replaced = []
    for attr, value in sorted(vars(module).items()):
        if isinstance(value, Timed) or not isinstance(value, torch.nn.Module):
            continue
        setattr(module, attr, Timed(value, device, "%s.%s" % (name, attr), torch))
        replaced.append(attr)
    return replaced


def prepare(environ=None):
    """Set up the timing of the embedded-Python inference this run asks for
    (FASINATE_TIMED_*); raises RuntimeError, stopping ddsim, if it can't."""
    environ = os.environ if environ is None else environ
    inference = environ.get("FASINATE_TIMED_INFERENCE") or "cpu"
    # Each module once, in order, though several of a model's configurations
    # can run it (its barrel's and its endcap's): wrapped again, it would have
    # no torch module left to wrap, which stops ddsim on the GPU
    modules = list(dict.fromkeys(
        m for m in (environ.get("FASINATE_TIMED_PYTHON_MODULES") or "").split(",") if m))
    if inference not in ("cpu", "cuda"):
        raise RuntimeError("ddml-timing: FASINATE_TIMED_INFERENCE should be cpu or cuda, not %r"
                           % inference)
    if not modules:
        say("no embedded-Python inference in this run (inference %s)" % inference)
        return []
    path = environ.get("FASINATE_TIMED_MODULE_PATH")
    if path:
        while path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)
    try:
        import torch
    except Exception as error:  # ImportError, or anything torch raises as it loads
        if inference == "cuda":
            raise RuntimeError("ddml-timing: the inference of %s should run on the GPU, which "
                               "needs PyTorch, but it can't be imported here: %s"
                               % (", ".join(modules), error))
        say("PyTorch can't be imported here (%s), so the inference calls of %s aren't timed"
            % (error, ", ".join(modules)))
        return []
    device, detail = "cpu", "PyTorch %s" % torch.__version__
    if inference == "cuda":
        problem = cuda_problem(torch)
        if problem:
            raise RuntimeError("ddml-timing: the inference of %s should run on the GPU, but %s"
                               % (", ".join(modules), problem))
        device = "cuda:%d" % torch.cuda.current_device()
        detail = "%s, PyTorch %s, CUDA %s" % (torch.cuda.get_device_name(torch.cuda.current_device()),
                                             torch.__version__, torch.version.cuda)
    done = []
    for name in modules:
        module = importlib.import_module(name)
        replaced = wrap(module, torch, device, name)
        if not replaced:
            if device != "cpu":
                raise RuntimeError(
                    "ddml-timing: %s (%s) has no torch module at its top level, so its inference "
                    "can't be moved to the GPU" % (name, getattr(module, "__file__", "?")))
            say("module %s (%s) has no torch module at its top level, so its inference calls "
                "aren't timed" % (name, getattr(module, "__file__", "?")))
        for attr in replaced:
            say("model %s.%s on %s (%s)" % (name, attr, device, detail))
            done.append("%s.%s" % (name, attr))
    return done
