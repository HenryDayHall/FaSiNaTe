"""The models: their descriptions (models/*.yaml), with DDML's presets or DDML
model configurations written out (ddml_bench/ddml_configs.py), the particles
DDML gives them, their closure tests and the sha256 sums of what they are; the
steering file that runs the configurations in ddsim
(scripts/ddsim_steer_models.py), through the fake ddsim, which runs it as
ddsim does; the models' own inference code in models/, on a stand-in for
PyTorch; the models stage's check of them; and the closure tests in the
plots."""

import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import types

import numpy as np
import pytest
import yaml

from conftest import FAKES, REPO
from ddml_bench import ddml_configs, models

REPO_MODELS = sorted(name[:-len(".yaml")] for name in os.listdir(os.path.join(REPO, "models"))
                     if name.endswith(".yaml"))
# Models set aside for now (unused_models/), whose descriptions some tests still
# use: a compiled model that is the closure test of one DDML runs in Python
# (caloclouds3_libtorch), and a model trained on other particles than DDML gives
# it (par04_vae_onnx)
UNUSED_MODELS = os.path.join(REPO, "unused_models")
INPUTS = ["p22_th43-137_ph79-109_en20_seed0_ip_surface",
          "p22_th43-137_ph79-109_en80_seed1_ip_surface"]
DI_PHOTON_SIMULATIONS = ["diphoton_en20_sep90_x0_z70_r10_seed0_simseed%d" % seed
                         for seed in (1, 2)]


def read_csv(path):
    with open(path) as stream:
        return list(csv.DictReader(stream))


@pytest.fixture
def fake_ddml(monkeypatch):
    """DDML's Python package as the tests fake it (with its presets as DDML has
    them), and g4units, importable here; gone again afterwards."""
    monkeypatch.syspath_prepend(os.path.join(FAKES, "python"))
    monkeypatch.syspath_prepend(os.path.join(FAKES, "ddml", "python"))
    yield
    for name in list(sys.modules):
        if name == "ddml" or name.startswith("ddml.") or name in ("g4units", "DDSim") \
                or name.startswith("DDSim."):
            del sys.modules[name]


# ---------------------------------------------------------------------------
# The descriptions


def test_every_model_in_the_repository_is_described_whole():
    described = models.load_models()
    assert [model["name"] for model in described] == REPO_MODELS
    labels = [model["label"] for model in described]
    assert len(set(labels)) == len(labels)
    # Each curve can be told apart by its colour, and without colour
    for key in ("color", "linestyle"):
        styles = [json.dumps(model[key]) for model in described]
        assert len(set(styles)) == len(styles), key
    for model in described:
        assert bool(model["presets"]) != bool(model["model_configs"]), model["name"]
        assert model["weights"] and model["weights_md5"], model["name"]
        # Its weights are in DDML's build, or kept in this repository, where
        # they are what their md5 sum says
        assert model["weights_in"] in ("ddml", "fasinate"), model["name"]
        if model["weights_in"] == "fasinate":
            with open(os.path.join(REPO, model["weights"]), "rb") as stream:
                assert hashlib.md5(stream.read()).hexdigest() == model["weights_md5"], \
                    model["name"]
        assert model["compact_file"] == "ILD/compact/ILD_l5_o1_v02/ILD_l5_o1_v02.xml"
        assert model["particles"] == ["gamma"], model["name"]
        for entry in model["model_configs"]:
            if "plugin_properties" in entry and "ModelPath" in entry["plugin_properties"]:
                # Relative to DDML's scripts/ folder, where ddsim runs
                assert entry["plugin_properties"]["ModelPath"] == "../" + model["weights"]
        assert len(model["description_sha256"]) == 64, model["name"]
        # Its own inference code, if it has any: the module of models/ named
        # after it (caloclouds3_recompiled's)
        own = os.path.isfile(os.path.join(REPO, "models", model["name"] + ".py"))
        assert sorted(model["python"]) == ([model["name"]] if own else []), model["name"]
        for module, code in model["python"].items():
            assert code["path"] == "models/%s.py" % module
            with open(os.path.join(REPO, code["path"]), "rb") as stream:
                assert code["sha256"] == hashlib.sha256(stream.read()).hexdigest()
    # The models with closure tests (caloclouds3_libtorch, gan_onnx) are set
    # aside in unused_models/ for now
    closures = dict((m["name"], m["closure"]) for m in described if m["closure"])
    assert closures == {}


def describe(tmp_path, monkeypatch, ddml=None, **settings):
    """Load a model described as a small one with these settings changed
    (ddml replaces its ddml section), in a models/ folder of its own."""
    folder = tmp_path / "models"
    folder.mkdir(exist_ok=True)
    data = {"representation": "regular_grid",
            "fixed_at_training": {"detector": {"compact_file": "ILD.xml"},
                                  "particle": {"name": "gamma"}},
            "context": {"energy": {"unit": "GeV"}},
            "ddml": ddml if ddml is not None else {"presets": ["CC3_BARREL"]}}
    data.update(settings)
    (folder / "model.yaml").write_text(yaml.safe_dump(data))
    (folder / "other.yaml").write_text(yaml.safe_dump(dict(data, ddml={"presets": ["X"]})))
    monkeypatch.setattr(models, "MODELS_DIR", str(folder))
    return models.load_model("model")


GAN = {"plugin": "RegularGridGANPolyhedraBarrelONNXModel/BarrelModelONNX",
       "geometry": "ILD_BARREL", "plugin_properties": {"ModelPath": "../models/gan.onnx"},
       "applicable_particles": ["gamma"], "triggers": {"gamma": 10}, "correct_angles": True}


def test_a_model_runs_with_presets_or_model_configurations(tmp_path, monkeypatch):
    model = describe(tmp_path, monkeypatch)
    assert model["presets"] == ["CC3_BARREL"] and model["model_configs"] == []
    assert model["particles"] == ["gamma"] and model["closure"] is None
    model = describe(tmp_path, monkeypatch, ddml={"model_configs": [
        "CC3_BARREL", {"preset": "CC3_BARREL", "triggers": {"gamma": 5}}, GAN],
        "particles": ["gamma"]}, closure={"of": "other"})
    assert model["presets"] == []
    # A preset's name is an entry of just that preset
    assert model["model_configs"] == [{"preset": "CC3_BARREL"},
                                      {"preset": "CC3_BARREL", "triggers": {"gamma": 5}}, GAN]
    assert model["closure"] == {"of": "other", "identical": False}
    # Entries go to ddsim as JSON, the same every time
    assert json.loads(ddml_configs.option(GAN)) == GAN
    assert ddml_configs.option(dict(reversed(list(GAN.items())))) == ddml_configs.option(GAN)


@pytest.mark.parametrize("ddml, settings, message", [
    ({"weights": "models/x.pt"}, {},
     "ddml should have one of these: presets, DDML's own"),
    ({"presets": ["CC3_BARREL"], "model_configs": ["CC3_BARREL"]}, {},
     "one of these, not both"),
    ({"model_configs": [dict(GAN, triger={"gamma": 10})]}, {},
     "ddml.model_configs, entry 1 has triger, which DDML's ModelConfig doesn't have"),
    ({"model_configs": ["CC3_BARREL", {"plugin": GAN["plugin"]}]}, {},
     "entry 2 has no preset, so it needs geometry, applicable_particles, triggers too"),
    ({"model_configs": [dict(GAN, triggers={"gamma": "10 GeV"})]}, {},
     "triggers should be {particle: energy in GeV, at least 0}"),
    ({"model_configs": [dict(GAN, geometry={"ecal": {"detector": "EcalBarrel"}})]}, {},
     "geometry ecal.region should be the name of a detector region"),
    ({"model_configs": [dict(GAN, plugin="two words")]}, {},
     "plugin should be a DDML plugin"),
    ({"model_configs": [{"preset": 3}]}, {},
     "preset should name one of DDML's presets"),
    ({"model_configs": [dict(GAN, plugin_properties={"ModelPath": {"a": 1}})]}, {},
     "plugin_properties ModelPath should be a number, a string"),
    ({"model_configs": [dict(GAN, applicable_particles=["gamma", "gamma"])]}, {},
     "applicable_particles names a particle twice"),
    ({"presets": ["CC3_BARREL"], "particles": []}, {},
     "ddml.particles should be the particles DDML gives the model"),
    ({"presets": ["CC3_BARREL"]}, {"closure": {"of": "nothing"}},
     "closure.of should be another model in models/ (other), not 'nothing'"),
    ({"presets": ["CC3_BARREL"]}, {"closure": {"of": "model"}},
     "closure.of should be another model"),
    ({"presets": ["CC3_BARREL"]}, {"closure": {"of": "other", "identical": "yes"}},
     "closure.identical should be true or false"),
    ({"presets": ["CC3_BARREL"]}, {"closure": "other"},
     "closure should be {of: the model this is the same as"),
])
def test_what_is_wrong_with_a_description_is_said(tmp_path, monkeypatch, ddml, settings,
                                                  message):
    with pytest.raises(models.ModelError) as error:
        describe(tmp_path, monkeypatch, ddml=ddml, **settings)
    assert message in str(error.value)
    # Every message says which description it is about
    assert str(error.value).startswith(
        os.path.relpath(str(tmp_path / "models" / "model.yaml"), models.REPO_DIR) + ": ")


def test_what_a_model_is_by_the_sha256_of_its_description_and_own_code(tmp_path, monkeypatch):
    model = describe(tmp_path, monkeypatch, energy_rescale=1.5)
    path = tmp_path / "models" / "model.yaml"
    data = yaml.safe_load(path.read_text())
    # Not by the description's comments, layout, label or plot style...
    path.write_text("# Said otherwise\n" + yaml.safe_dump(
        dict(data, name="Another label", plot={"color": "#000000", "linestyle": ":"}),
        default_flow_style=True))
    assert models.load_model("model")["description_sha256"] == model["description_sha256"]
    assert models.description_sha256_now(model) == model["description_sha256"]
    # ... but by anything else, e.g. what the model's own code reads from it
    path.write_text(yaml.safe_dump(dict(data, energy_rescale=1.25)))
    assert models.load_model("model")["description_sha256"] != model["description_sha256"]
    assert models.description_sha256_now(dict(model, config="nowhere.yaml")) is None

    # Its own Python code: the modules of models/ its configurations run, not
    # DDML's
    assert model["python"] == {} and models.python_summary({}) == "none"
    code = tmp_path / "models" / "mine.py"
    code.write_text("def run_inference(inputs):\n    return inputs[0]\n")
    model = describe(tmp_path, monkeypatch, ddml={"model_configs": [
        {"preset": "CC3_BARREL_PY_INTERFACE", "plugin_properties": {"PythonModule": "mine"}},
        {"preset": "CC3_ENDCAP_PY_INTERFACE",
         "plugin_properties": {"PythonModule": "cc3_sf_2a_wrapper"}}]})
    sha256 = hashlib.sha256(code.read_bytes()).hexdigest()
    assert model["python"] == {"mine": {"path": os.path.relpath(str(code), models.REPO_DIR),
                                        "sha256": sha256}}
    assert models.python_summary(model["python"]) == "mine:" + sha256
    assert models.python_module_file("mine") == str(code)
    assert models.python_module_file("../mine") is None


# ---------------------------------------------------------------------------
# DDML's ModelConfigs, made with DDML's Python package (the fake one, with
# DDML's presets as DDML declares them)


def test_the_entries_make_ddmls_model_configurations(fake_ddml):
    from ddml import configs
    from ddml.geometry import DetectorGeometry, PluginGeometry
    changed, written, placed = ddml_configs.build([
        {"preset": "CC3_BARREL", "plugin_properties": {"ModelPath": "../models/CC3_SF_2A.pt"}},
        dict(GAN, plugin_properties={"ModelPath": "../models/gan.onnx", "OptimizeFlag": 1}),
        dict(GAN, geometry={"ecal": {"detector": "EcalBarrel", "region": "EcalBarrelRegion",
                                     "symmetry": 8}})])
    # A preset with its properties set one by one, the rest as it is
    assert changed.plugin == configs.CC3_BARREL.plugin
    assert changed.plugin_properties == {"OptimizeFlag": 1, "IntraOpNumThreads": 1,
                                         "ModelPath": "../models/CC3_SF_2A.pt"}
    assert changed.triggers == configs.CC3_BARREL.triggers
    assert configs.CC3_BARREL.plugin_properties["ModelPath"] == "../models/CC3_paper_checkpoint.pt"
    # One written out: the triggers in GeV become Geant4's units
    assert written.plugin == GAN["plugin"] and written.geometry is configs.ILD_BARREL
    assert written.applicable_particles == frozenset({"gamma"})
    assert written.triggers == {"gamma": 10000.0} and written.correct_angles
    assert not written.is_hadron
    assert placed.geometry == PluginGeometry(ecal=DetectorGeometry("EcalBarrel",
                                                                   "EcalBarrelRegion", 8))
    assert "gamma above 10 GeV; CorrectForAngles True" in ddml_configs.describe(written)
    with pytest.raises(ddml_configs.ModelConfigError, match="DDML has no preset NOPE"):
        ddml_configs.build([{"preset": "NOPE"}])
    with pytest.raises(ddml_configs.ModelConfigError, match="DDML has no geometry NOWHERE"):
        ddml_configs.build([dict(GAN, geometry="NOWHERE")])


def test_the_steering_files_options_come_out_of_the_command_line():
    argv = ["ddsim", "--steeringFile", "s.py", "--ml-config", ddml_configs.option(GAN),
            "--compactFile", "c.xml", "--ml-model", "CC3_BARREL",
            "--ml-config={\"preset\": \"CC3_ENDCAP\"}", "--random.seed=3", "--ml-conf", "x"]
    entries, record = ddml_configs.from_command_line(argv)
    assert entries == [GAN, {"preset": "CC3_BARREL"}, {"preset": "CC3_ENDCAP"}] and not record
    # ddsim's own are left, in their order; nothing is taken by an abbreviation
    assert argv == ["ddsim", "--steeringFile", "s.py", "--compactFile", "c.xml",
                    "--random.seed=3", "--ml-conf", "x"]
    assert ddml_configs.from_command_line(["ddsim", "--record-calo-entry"]) == ([], True)
    for wrong in ("{not json", json.dumps({"preset": "X", "triger": 1})):
        with pytest.raises(SystemExit):
            ddml_configs.from_command_line(["ddsim", "--ml-config", wrong])


def run_fake_ddsim(tmp_path, *options):
    """The fake ddsim, which runs the steering file as ddsim does, on an input
    of one event, with the fake DDML; (status, output, what it simulated)."""
    source = os.path.join(FAKES, "ddml")
    events = tmp_path / "in.slcio"
    events.write_text('{"run": 1}\n{"event": 0, "particles": []}\n')
    (tmp_path / "ILD.xml").write_text("<lccdd/>\n")
    out = tmp_path / ".out.root"
    if out.exists():
        out.unlink()
    env = dict(os.environ, DDML_SOURCE_DIR=source, PYTHONPATH=os.pathsep.join(
        [os.path.join(FAKES, "python"), os.path.join(source, "python"),
         os.path.join(source, "python", "examples")]))
    result = subprocess.run(
        [sys.executable, os.path.join(FAKES, "bin", "ddsim"), "--steeringFile",
         os.path.join(REPO, "scripts", "ddsim_steer_models.py"),
         "--compactFile", str(tmp_path / "ILD.xml")] + list(options) +
        ["--inputFiles", str(events), "--outputFile", str(out), "--numberOfEvents=-1"],
        cwd=os.path.join(source, "scripts"), env=env, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, universal_newlines=True)
    simulated = json.loads(out.read_text()) if out.exists() else None
    return result.returncode, result.stdout, simulated


def test_the_steering_file_sets_ddmls_up_with_the_configurations(tmp_path):
    status, output, simulated = run_fake_ddsim(
        tmp_path, "--ml-config", ddml_configs.option(
            {"preset": "CC3_BARREL", "plugin_properties": {"ModelPath": "../models/m.pt"}}),
        "--ml-config", ddml_configs.option(GAN), "--random.seed=7")
    assert status == 0, output
    assert simulated["simulator"] == "CC3_BARREL+RegularGridGANPolyhedraBarrelONNXModel"
    assert simulated["seed"] == 7 and len(simulated["events"]) == 1
    first, second = simulated["models"]
    assert first["plugin"] == "CaloCloudsTwoAngleModelPolyhedraBarrelTorchModel/BarrelModelTorch"
    assert first["plugin_properties"] == {"OptimizeFlag": 1, "IntraOpNumThreads": 1,
                                          "ModelPath": "../models/m.pt"}
    assert second["plugin"] == GAN["plugin"] and second["correct_angles"]
    assert second["triggers_gev"] == {"gamma": 10.0} and second["detector"] == "EcalBarrel"
    # The log says what it set up
    assert "ddsim_steer_models.py: model configuration 2: %s" % ddml_configs.option(GAN) \
        in output
    assert "RegularGridGANPolyhedraBarrelONNXModel/BarrelModelONNX, EcalBarrel in " \
        "EcalBarrelRegion (symmetry 8); gamma above 10 GeV; CorrectForAngles True" in output

    status, output, _ = run_fake_ddsim(tmp_path)
    assert status != 0 and "give the model's configurations with --ml-config" in output
    status, output, _ = run_fake_ddsim(tmp_path, "--ml-config", '{"preset": "NOPE"}')
    assert status != 0 and "DDML has no preset NOPE" in output


def test_the_steering_file_runs_the_models_own_code_from_models(tmp_path):
    """A configuration's embedded-Python plugin imports the module it names
    from models/, first on ddsim's Python path, and DDML's own from DDML."""
    status, output, simulated = run_fake_ddsim(
        tmp_path, "--ml-config", ddml_configs.option(
            {"preset": "CC3_BARREL_PY_INTERFACE",
             "plugin_properties": {"PythonModule": "caloclouds3_recompiled"}}),
        "--ml-config", ddml_configs.option({"preset": "CC3_ENDCAP_PY_INTERFACE"}))
    assert status == 0, output
    own, ddmls = simulated["models"]
    assert own["plugin_properties"] == {"PythonModule": "caloclouds3_recompiled",
                                        "EntryPoint": "run_inference"}
    assert os.path.realpath(own["python_module"]) == os.path.realpath(
        os.path.join(REPO, "models", "caloclouds3_recompiled.py"))
    assert os.path.realpath(ddmls["python_module"]) == os.path.realpath(
        os.path.join(FAKES, "ddml", "python", "examples", "cc3_sf_2a_wrapper.py"))
    # A module that is nowhere stops ddsim, as DDML's plugin does
    status, output, _ = run_fake_ddsim(tmp_path, "--ml-config", ddml_configs.option(
        {"preset": "CC3_BARREL_PY_INTERFACE", "plugin_properties": {"PythonModule": "nowhere"}}))
    assert status != 0 and "No module named 'nowhere'" in output, output


# ---------------------------------------------------------------------------
# The models' own inference code (models/caloclouds3_recompiled.py), as DDML's
# embedded-Python plugin runs it, on a stand-in for PyTorch whose model gives a
# made-up output


class FakeTensor(object):
    def __init__(self, array):
        self.array = array

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array


def fake_torch(output, seen):
    """A stand-in for PyTorch, whose TorchScript model gives a copy of output,
    noting in seen the weights it loads and what it is given."""
    torch = types.ModuleType("torch")
    torch.set_num_threads = lambda threads: None
    torch.from_numpy = FakeTensor
    torch.inference_mode = lambda: (lambda function: function)

    class Model(object):
        def eval(self):
            return self

        def to(self, device):
            seen["device"] = device
            return self

        def __call__(self, *tensors):
            seen["inputs"] = [tensor.array for tensor in tensors]
            return FakeTensor(output.copy())

    def load(path):
        seen["weights"] = path
        return Model()

    torch.jit = types.SimpleNamespace(load=load)
    return torch


def import_own_code(monkeypatch, module, output, seen, ddml="/sw/DDML"):
    """A module of models/, imported as the plugin imports it, with the fake
    PyTorch and DDML's build in ddml ($DDML_SOURCE_DIR; unset if None)."""
    monkeypatch.setitem(sys.modules, "torch", fake_torch(output, seen))
    if ddml is None:
        monkeypatch.delenv("DDML_SOURCE_DIR", raising=False)
    else:
        monkeypatch.setenv("DDML_SOURCE_DIR", ddml)
    spec = importlib.util.spec_from_file_location(
        module + "_under_test", os.path.join(REPO, "models", module + ".py"))
    imported = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(imported)
    return imported


def caloclouds_output():
    """An output as CaloClouds3 gives it, in a batch of one: the number of
    points in each of the 30 layers (two in the first, one in the fourth),
    then each point as (x, y, layer, energy), then what DDML doesn't read."""
    counts = np.zeros(30, dtype=np.float32)
    counts[0], counts[3] = 2, 1
    points = np.array([[1.5, -2.0, 0, 0.25], [0.5, 0.5, 0, 1.0], [-3.0, 4.0, 3, 2.5]],
                      dtype=np.float32)
    return np.concatenate([counts, points.ravel(), np.zeros(8, dtype=np.float32)])[None, :]


def test_caloclouds3_recompiled_runs_the_weights_kept_in_this_repository(monkeypatch):
    with open(os.path.join(REPO, "models", "caloclouds3_recompiled.yaml")) as stream:
        described = yaml.safe_load(stream)
    assert described["ddml"]["weights_in"] == "fasinate"
    output, seen = caloclouds_output(), {}
    # It needs no DDML build ($DDML_SOURCE_DIR unset)...
    recompiled = import_own_code(monkeypatch, "caloclouds3_recompiled", output, seen, ddml=None)
    # ... as the weights of its description are in this repository, whose md5
    # sum is checked
    assert seen["weights"] == os.path.join(REPO, described["ddml"]["weights"])
    assert seen["device"] == "cpu"

    inputs = [np.array([[value]], dtype=np.float32) for value in (50.0, 90.0, 85.0)]
    out = recompiled.run_inference(inputs)
    assert [a.tolist() for a in seen["inputs"]] == [[[50.0]], [[90.0]], [[85.0]]]
    # The model's output as it is, flat, as DDML reads it
    assert out.dtype == np.float32 and out.shape == (output.size,)
    assert out.tolist() == output.ravel().tolist()


# ---------------------------------------------------------------------------
# In the pipeline


def test_every_model_of_the_repository_runs_in_ddsim(bench):
    """Each of the repository's models, presets or configurations, set up in
    ddsim by the steering file it needs, as its description says, with its
    weights where it keeps them: in DDML's build, or in this repository."""
    for name in REPO_MODELS:
        bench.add_model(name)
    bench.run("generate_input.sh", "single_photon")
    status, output = bench.run("simulate_models.sh", "-n", "2", "-s", "single_photon")
    assert "made %d of %d" % (2 * len(REPO_MODELS), 2 * len(REPO_MODELS)) in output, output
    for model in REPO_MODELS:
        with open(os.path.join(bench.repo, "models", model + ".yaml")) as stream:
            described = yaml.safe_load(stream)
        ddml = described["ddml"]
        kept = bench.repo if ddml.get("weights_in") == "fasinate" else bench.ddml
        assert "%s: weights %s, md5 checked" % (model, os.path.join(kept, ddml["weights"])) \
            in output, output
        own = os.path.join(bench.repo, "models", model + ".py")
        for name in INPUTS:
            path = bench.path("sim_output", "single_photon", model, name + "_n2.root")
            with open(path) as stream:
                simulated = json.load(stream)
            log = open(path[:-len(".root")] + ".log").read()
            config, = simulated["models"]
            assert config["applicable_particles"] == ["gamma"]
            assert config["triggers_gev"] == {"gamma": 10.0}
            assert config["detector"] == "EcalBarrel"
            if "presets" in ddml:
                assert "--ml-model %s" % ddml["presets"][0] in log
                assert "/DDML/scripts/ddsim_steer.py" in log
                assert simulated["simulator"] == ddml["presets"][0]
            else:
                entry, = ddml["model_configs"]
                assert "scripts/ddsim_steer_models.py --compactFile" in log
                assert "--ml-config " in log and "model configuration 1:" in log
                assert config["plugin"] == entry.get("plugin", config["plugin"])
                for key, value in entry.get("plugin_properties", {}).items():
                    assert config["plugin_properties"][key] == value
                if "correct_angles" in entry:
                    assert config["correct_angles"] == entry["correct_angles"]
            # The plugin loads the weights whose md5 sum was checked, or its
            # Python module does: DDML's, or the model's own from models/
            if config["python_module"] is None:
                assert config["plugin_properties"]["ModelPath"] == "../" + ddml["weights"]
            elif os.path.isfile(own):
                assert os.path.realpath(config["python_module"]) == os.path.realpath(own)
            else:
                assert config["python_module"].startswith(os.path.join(bench.ddml, "python"))
            # The log says what the model is, as ddsim started
            assert "# model     %s, of models/%s.yaml as read" % (
                models.description_sha256(described), model) in log, log
            if os.path.isfile(own):
                with open(own, "rb") as stream:
                    sha256 = hashlib.sha256(stream.read()).hexdigest()
                assert "# python    %s:%s, from models/%s.py\n" % (model, sha256, model) in log
            else:
                assert "# python    none, the model runs no module of models/\n" in log


def test_the_environment_check_finds_what_a_model_gets_wrong(bench):
    text = open(os.path.join(UNUSED_MODELS, "caloclouds3_libtorch.yaml")).read()
    assert "    - preset: CC3_BARREL\n" in text and "ModelPath: ../models/CC3_SF_2A.pt" in text
    # without its closure test of caloclouds3, which is no more
    assert "\nclosure:\n  of: caloclouds3\n  identical: false\n" in text
    text = text.replace("\nclosure:\n  of: caloclouds3\n  identical: false\n", "\n")
    bench.add_model("other_weights", text.replace("ModelPath: ../models/CC3_SF_2A.pt",
                                                  "ModelPath: ../models/CC3_paper_checkpoint.pt"))
    bench.add_model("no_plugin", text.replace("    - preset: CC3_BARREL\n",
                                              "    - preset: CC3_BARREL\n"
                                              "      plugin: NoSuchModel/Barrel\n"))
    bench.add_model("electrons", text.replace("    - preset: CC3_BARREL\n",
                                              "    - preset: CC3_BARREL\n"
                                              "      applicable_particles: [e-]\n"))
    bench.add_model("no_preset", text.replace("    - preset: CC3_BARREL\n",
                                              "    - preset: CC3_NOPE\n"))
    # Python modules: one that is nowhere, and one of models/ that would stand
    # in for a preset's own without the description naming it
    own = open(os.path.join(REPO, "models", "caloclouds3_recompiled.yaml")).read()
    assert "        PythonModule: caloclouds3_recompiled\n" in own
    bench.add_model("no_module", own.replace("PythonModule: caloclouds3_recompiled",
                                             "PythonModule: nowhere"))
    bench.add_model("unnamed_module", own.replace(
        "      plugin_properties:\n        PythonModule: caloclouds3_recompiled\n", ""))
    with open(os.path.join(bench.repo, "models", "cc3_sf_2a_wrapper.py"), "w") as stream:
        stream.write("def run_inference(inputs):\n    return inputs[0]\n")
    # and a model of presets, which runs DDML's own even so: here a preset of
    # DDML's embedded-Python plugin
    paper = open(os.path.join(REPO, "models", "caloclouds3_paper.yaml")).read()
    assert "  presets: [CC3_BARREL]\n" in paper
    bench.add_model("python_preset", paper.replace("  presets: [CC3_BARREL]\n",
                                                   "  presets: [CC3_BARREL_PY_INTERFACE]\n"))
    # Weights kept in this repository (ddml.weights_in: fasinate), checked
    # there: ones that aren't there, and ones that aren't what their md5 sum says
    assert "\n  weights_in: fasinate\n" in own
    assert "\n  weights: models/weights/CC3_recompiled.pt\n" in own
    kept_here = os.path.join(bench.repo, "models", "weights")
    for name in ("missing_weights", "changed_weights"):
        bench.add_model(name, own.replace("models/weights/CC3_recompiled.pt",
                                          "models/weights/%s.pt" % name))
    os.remove(os.path.join(kept_here, "missing_weights.pt"))
    with open(os.path.join(kept_here, "changed_weights.pt"), "rb") as stream:
        described_md5 = hashlib.md5(stream.read()).hexdigest()
    with open(os.path.join(kept_here, "changed_weights.pt"), "wb") as stream:
        stream.write(b"other weights\n")
    # Given other particles on purpose, and saying so
    bench.add_model("par04_vae_onnx", folder="unused_models")
    bench.run("generate_input.sh", "single_photon")
    status, output = bench.run("simulate_models.sh", "--dry-run", "-n", "1", "-s",
                               "single_photon", check=False)
    assert status == 1, output
    assert "no_module: model configuration 1 runs the Python module nowhere, which isn't in " \
        "models/ or on PYTHONPATH" in output, output
    assert ("unnamed_module: model configuration 1 runs the Python module cc3_sf_2a_wrapper, "
            "which ddsim would take from %s, ahead of DDML's" % os.path.join(
                bench.repo, "models", "cc3_sf_2a_wrapper.py")) in output, output
    # but a model with presets runs DDML's own, as DDML's steering file does
    assert "python_preset: preset CC3_BARREL_PY_INTERFACE runs the Python module %s" % (
        os.path.join(bench.ddml, "python", "examples", "cc3_sf_2a_wrapper.py")) in output, output
    assert "missing_weights: weights not found: %s " % os.path.join(
        kept_here, "missing_weights.pt") in output, output
    assert ("changed_weights: the md5 sum of %s is %s, not %s as in models/changed_weights.yaml"
            % (os.path.join(kept_here, "changed_weights.pt"),
               hashlib.md5(b"other weights\n").hexdigest(), described_md5)) in output, output
    # and those that are fine are checked there too
    assert "unnamed_module: weights %s, md5 checked" % os.path.join(
        kept_here, "CC3_recompiled.pt") in output, output
    weights = os.path.join(bench.ddml, "models", "CC3_SF_2A.pt")
    assert ("other_weights: model configuration 1 loads %s, but the weights "
            "models/other_weights.yaml checks are %s" % (
                os.path.join(bench.ddml, "models", "CC3_paper_checkpoint.pt"), weights)) in output
    assert "no_plugin: model configuration 1 needs the plugin NoSuchModel, which no " \
        ".components file on LD_LIBRARY_PATH declares" in output
    assert "electrons: model configuration 1 gives e- to the model, but " \
        "models/electrons.yaml says it was trained on gamma" in output
    assert "no_preset: can't make its DDML model configurations (models/no_preset.yaml): " \
        "DDML has no preset CC3_NOPE" in output
    assert "par04_vae_onnx: trained on e-, but DDML gives it gamma (ddml.particles in " \
        "models/par04_vae_onnx.yaml)" in output
    # which is no problem
    assert not [line for line in output.splitlines()
                if line.startswith("ddml-bench: par04_vae_onnx")], output


def test_a_compiled_model_is_its_python_twins_closure_test(bench):
    # Set aside in unused_models/ for now, but its closure test still runs. Its
    # twin, caloclouds3 (the same weights through DDML's embedded-Python
    # plugin), is no more, so caloclouds3_recompiled, which that plugin runs
    # too, stands in for it
    text = open(os.path.join(UNUSED_MODELS, "caloclouds3_libtorch.yaml")).read()
    assert "\nclosure:\n  of: caloclouds3\n" in text
    bench.add_model("caloclouds3_recompiled")
    bench.add_model("caloclouds3_libtorch", text.replace(
        "\nclosure:\n  of: caloclouds3\n", "\nclosure:\n  of: caloclouds3_recompiled\n"))
    status, output = bench.run("pipeline.sh")
    # It ran with its configuration, from the steering file for them
    with open(bench.path("sim_output", "di_photon", "caloclouds3_libtorch",
                         DI_PHOTON_SIMULATIONS[0] + ".root")) as stream:
        simulated = json.load(stream)
    assert simulated["simulator"] == "CC3_BARREL" and simulated["seed"] == 1
    config, = simulated["models"]
    assert config["plugin"] == "CaloCloudsTwoAngleModelPolyhedraBarrelTorchModel/BarrelModelTorch"
    assert config["plugin_properties"]["ModelPath"] == "../models/CC3_SF_2A.pt"
    with open(bench.path("sim_output", "di_photon", "caloclouds3_recompiled",
                         DI_PHOTON_SIMULATIONS[0] + ".root")) as stream:
        twin = json.load(stream)
    assert twin["models"][0]["plugin"].startswith(
        "CaloCloudsTwoAngleModelPolyhedraBarrelPyEmbeddedModel")

    # Both are compared with Geant4 (as the bench's model is), and with each
    # other in closure/
    plots = bench.path("observables", "plots")
    assert {row["simulator"] for row in read_csv(os.path.join(plots, "metrics.csv"))} == {
        "caloclouds3_paper", "caloclouds3_recompiled", "caloclouds3_libtorch"}
    closure = os.path.join(plots, "closure", "caloclouds3_libtorch")
    rows = read_csv(os.path.join(closure, "metrics.csv"))
    assert rows and {row["simulator"] for row in rows} == {"caloclouds3_libtorch"}
    # The fakes simulate and reconstruct every model alike, so each is the other's
    assert all(float(row["value"]) == 0 for row in rows if row["metric"] == "JS")
    assert os.path.isfile(os.path.join(closure, "radial_profile_r30__single_photon.png"))
    # A comparison of the distributions, even with the same seeds
    assert read_csv(os.path.join(plots, "closure.csv")) == [
        {"model": "caloclouds3_libtorch", "reference": "caloclouds3_recompiled",
         "scenario": "di_photon", "seeds": "same", "identical": "yes",
         "expected": "statistical", "passed": ""},
        {"model": "caloclouds3_libtorch", "reference": "caloclouds3_recompiled",
         "scenario": "single_photon", "seeds": "ddsim's own", "identical": "yes",
         "expected": "statistical", "passed": ""}]
    assert "closure test: caloclouds3_libtorch on di_photon: the same as " \
        "caloclouds3_recompiled" in output, output

    # A pair that should give the same events with the same seeds passes...
    path = os.path.join(bench.repo, "models", "caloclouds3_libtorch.yaml")
    with open(path) as stream:
        text = stream.read()
    with open(path, "w") as stream:
        stream.write(text.replace("  identical: false\n", "  identical: true\n"))
    status, output = bench.run("plot_observables.sh")
    assert [(row["scenario"], row["expected"], row["passed"])
            for row in read_csv(os.path.join(plots, "closure.csv"))] == [
        ("di_photon", "identical", "yes"), ("single_photon", "statistical", "")]

    # ... and measurements that differ fail it
    path = bench.path("observables", "caloclouds3_libtorch.npz")
    with np.load(path) as data:
        arrays = dict((key, data[key]) for key in data.files)
    arrays["photon_separation__di_photon__n_photons"] = \
        arrays["photon_separation__di_photon__n_photons"] + 1
    np.savez_compressed(path, **arrays)
    status, output = bench.run("plot_observables.sh")
    assert "closure test FAILED: caloclouds3_libtorch and caloclouds3_recompiled were " \
        "simulated with the same seeds on di_photon, but their measurements differ" in output, \
        output
    rows = read_csv(os.path.join(plots, "closure.csv"))
    assert [(row["scenario"], row["identical"], row["passed"]) for row in rows] == [
        ("di_photon", "no", "no"), ("single_photon", "yes", "")]
    # and only the closure tests asked for are drawn
    status, output = bench.run("plot_observables.sh", "--no-closure", "-o",
                               os.path.join(bench.tmp, "elsewhere"))
    assert not os.path.exists(os.path.join(bench.tmp, "elsewhere", "closure.csv"))
    assert "closure test" not in output


def test_models_ddmls_build_leaves_out_are_fetched_by_their_md5_sums(tmp_path):
    """scripts/fetch_ddml_models.py, which build_ddml.sh runs for
    ddml.extra_models: as DDML's build fetches its own, by the md5 sum in
    DDML's models/<file>.md5, checked, and only when they aren't there."""
    import hashlib
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    try:
        import fetch_ddml_models
    finally:
        sys.path.pop(0)
    server, models_dir = tmp_path / "eos", tmp_path / "DDML" / "models"
    server.mkdir()
    models_dir.mkdir(parents=True)
    weights = b"the weights of a flow\n"
    md5 = hashlib.md5(weights).hexdigest()
    (server / md5).write_bytes(weights)
    (models_dir / "L2LFlowsx9.pt.md5").write_text(md5 + "\n")
    url = "file://%s/{md5}" % server

    assert fetch_ddml_models.main(["--url", url, "--models-dir", str(models_dir), "--dry-run",
                                   "L2LFlowsx9.pt"]) == 0
    assert not (models_dir / "L2LFlowsx9.pt").exists()
    assert fetch_ddml_models.fetch(["L2LFlowsx9.pt"], str(models_dir), url) == []
    assert (models_dir / "L2LFlowsx9.pt").read_bytes() == weights
    # There already, so not fetched again
    (server / md5).unlink()
    assert fetch_ddml_models.fetch(["L2LFlowsx9.pt"], str(models_dir), url) == []

    # A file that isn't what its md5 sum says isn't put in place
    (models_dir / "L2LFlowsx9.pt").write_bytes(b"something else\n")
    (server / md5).write_bytes(b"not the weights\n")
    problem, = fetch_ddml_models.fetch(["L2LFlowsx9.pt"], str(models_dir), url)
    assert "isn't the published file: its md5 is" in problem
    assert (models_dir / "L2LFlowsx9.pt").read_bytes() == b"something else\n"
    # and a model DDML has no md5 sum for is an error
    problem, = fetch_ddml_models.fetch(["Nothing.pt"], str(models_dir), url)
    assert "DDML has no model Nothing.pt to fetch" in problem
    with pytest.raises(SystemExit):
        fetch_ddml_models.main(["--url", "file:///x", "--models-dir", str(models_dir)])


def test_the_default_configuration_fetches_l2lflows():
    from ddml_bench.config import load_config
    ddml = load_config()["ddml"]
    assert "L2LFlowsx9.pt" in ddml["extra_models"] and "{md5}" in ddml["models_url"]
    with open(os.path.join(REPO, "models", "l2lflows.yaml")) as stream:
        described = yaml.safe_load(stream)["ddml"]
    assert described["weights"] == "models/L2LFlowsx9.pt"
