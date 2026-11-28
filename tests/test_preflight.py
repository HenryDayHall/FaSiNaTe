"""The preflight of the models (ddml_bench/preflight.py, ./preflight_models.sh):
each step, what it writes to preflight_output, and how the simulations of a
model wait for it."""

import json
import os

import pytest
import yaml

from conftest import REPO
from ddml_bench import preflight
from ddml_bench.graph import DONE, TODO, UNNEEDED, Graph, PlanError, make_plan

MODEL = "caloclouds3_paper"
# A model's own inference code, as Black writes it, and as it doesn't
FORMATTED = "def run_inference(inputs):\n    return inputs[0]\n"
UNFORMATTED = "def run_inference( inputs ):\n  return inputs[0]\n"


def own_model(bench, name, code):
    """A model running its own module, models/<name>.py, with this code, as
    caloclouds3_recompiled runs its own."""
    text = open(os.path.join(REPO, "models", "caloclouds3_recompiled.yaml")).read()
    bench.add_model(name, text.replace("PythonModule: caloclouds3_recompiled",
                                       "PythonModule: %s" % name))
    with open(os.path.join(bench.repo, "models", name + ".py"), "w") as stream:
        stream.write(code)


def read(bench, *parts):
    with open(bench.path("preflight_output", *parts)) as stream:
        return stream.read()


def control(bench, name, on=True):
    """Switch one of the fake ddsim's behaviours (tests/fakes/bin/ddsim)."""
    path = os.path.join(bench.tmp, "fake-control", name)
    if on:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "w").close()
    elif os.path.exists(path):
        os.remove(path)


# ---------------------------------------------------------------------------
# The settings, and what a model is checked as


SETTINGS = {"black": "26", "import_timeout": 600,
            "barrel": {"detector": "EcalBarrel", "energy": 100, "position": [2, 0, 0],
                       "directions": {"up": [0, 1, 0], "tilted": [0, 1, 0.5]},
                       "hit_collections": ["ECalBarrelSiHitsEven"], "seed": 1}}


def test_the_default_settings_are_the_shots_asked_for():
    with open(os.path.join(REPO, "configs", "default.yaml")) as stream:
        found = preflight.settings(yaml.safe_load(stream))
    barrel = found["barrel"]
    assert barrel["energy"] == 100.0 and barrel["position"] == [2.0, 0.0, 0.0]
    assert barrel["directions"] == {"up": [0.0, 1.0, 0.0], "tilted": [0.0, 1.0, 0.5]}
    assert barrel["detector"] == "EcalBarrel" and found["black"] == "26"


def test_the_settings_are_checked():
    assert preflight.settings({"preflight": SETTINGS})["black"] == "26"
    assert preflight.settings({"preflight": dict(SETTINGS, black=None)})["black"] is None
    for change, message in (({"black": "latest"}, "preflight.black"),
                            ({"import_timeout": 0}, "import_timeout"),
                            ({"barrel": dict(SETTINGS["barrel"], energy=-1)}, "energy"),
                            ({"barrel": dict(SETTINGS["barrel"], position=[0, 0])}, "position"),
                            ({"barrel": dict(SETTINGS["barrel"], directions={"x": [0, 0, 0]})},
                             "no direction"),
                            ({"barrel": dict(SETTINGS["barrel"], hit_collections=[])},
                             "hit_collections")):
        with pytest.raises(preflight.PreflightError, match=message):
            preflight.settings({"preflight": dict(SETTINGS, **change)})
    with pytest.raises(preflight.PreflightError, match="no preflight section"):
        preflight.settings({})


def test_what_a_model_is_checked_as(config_env):
    from ddml_bench.models import load_model
    model = load_model(MODEL)
    pins = {"key4hep": "2026-04-08", "ddml": None, "ildconfig": None}
    first = preflight.digest(model, SETTINGS, pins)
    assert first == preflight.digest(dict(model), SETTINGS, pins)
    # Its description, weights, settings and software make it another check
    assert preflight.digest(dict(model, description_sha256="0" * 64), SETTINGS, pins) != first
    assert preflight.digest(dict(model, weights_md5="0" * 32), SETTINGS, pins) != first
    assert preflight.digest(model, dict(SETTINGS, black=None), pins) != first
    assert preflight.digest(model, SETTINGS, dict(pins, ddml="abc")) != first
    # and its own code
    assert preflight.digest(dict(model, python={"m": {"path": "models/m.py", "sha256": "1"}}),
                            SETTINGS, pins) != first


# ---------------------------------------------------------------------------
# Step 1, before any plan


def test_a_description_must_be_well_formed_and_complete(config_env):
    folder = os.path.join(config_env.repo, "models")
    text = open(os.path.join(folder, MODEL + ".yaml")).read()
    model, details = preflight.check_description(MODEL)
    assert model["name"] == MODEL and "well formed" in details[0]
    # A key twice, which YAML readers otherwise take, keeping the last
    with open(os.path.join(folder, "twice.yaml"), "w") as stream:
        stream.write(text + "\nname: Again\n")
    with pytest.raises(preflight.StepFailed, match="the key 'name' is in it twice"):
        preflight.check_description("twice")
    # Without the md5 sum of its weights
    with open(os.path.join(folder, "no_md5.yaml"), "w") as stream:
        stream.write("\n".join(line for line in text.splitlines()
                               if not line.strip().startswith("weights_md5")))
    with pytest.raises(preflight.StepFailed, match="ddml.weights_md5 is missing"):
        preflight.check_description("no_md5")
    # Weights somewhere unknown
    with open(os.path.join(folder, "elsewhere.yaml"), "w") as stream:
        stream.write(text.replace("ddml:\n", "ddml:\n  weights_in: somewhere\n", 1))
    with pytest.raises(preflight.StepFailed, match="weights_in should be ddml or fasinate"):
        preflight.check_description("elsewhere")
    # What the pipeline needs of every model
    with open(os.path.join(folder, "no_context.yaml"), "w") as stream:
        stream.write(text.replace("\ncontext:", "\nsomething_else:"))
    with pytest.raises(preflight.StepFailed, match="context is missing"):
        preflight.check_description("no_context")


def test_a_description_that_isnt_yaml_is_summed_up_and_stops_the_run(bench):
    with open(os.path.join(bench.repo, "models", "broken.yaml"), "w") as stream:
        stream.write("name: [unclosed\n")
    status, output = bench.run("preflight_models.sh", check=False)
    assert status == 1 and "the preflight of broken failed at step 1, description" in output, \
        output
    summary = read(bench, "broken", "summary.txt")
    assert "FAILED at step 1, description" in summary and "2. weights      not run" in summary
    assert "broken" in read(bench, "summary.txt")


# ---------------------------------------------------------------------------
# The whole preflight, on the fakes


def test_a_model_passes_and_is_checked_once(bench):
    status, output = bench.run("preflight_models.sh", "--dry-run")
    assert "would run 1 task(s)" in output and "preflight    %s" % MODEL in output, output
    status, output = bench.run("preflight_models.sh")
    assert "made 1 of 1" in output, output
    assert "the preflight of the models is summed up in %s" % bench.path(
        "preflight_output", "summary.txt") in output, output
    files = bench.files("preflight_output", MODEL)
    marker, = [name for name in files if name.startswith("passed_")]
    assert {"summary.txt", "summary.json", "preflight.log", "barrel_gamma.png", "barrel"} \
        <= set(files)
    summary = read(bench, MODEL, "summary.txt")
    assert "Preflight of %s (models/%s.yaml): passed" % (MODEL, MODEL) in summary, summary
    for line in ("1. description  passed", "2. weights      passed", "3. Python code  skipped",
                 "4. import       skipped", "5. barrel       passed",
                 "gamma up (0, 1, 0): 1 shower(s) placed by the model",
                 "gamma tilted (0, 1, 0.5): 1 shower(s) placed by the model",
                 bench.path("preflight_output", MODEL, "barrel_gamma.png"),
                 "Full logs:", bench.path("preflight_output", MODEL, "preflight.log"),
                 bench.path("preflight_output", MODEL, "barrel", "ddsim.log")):
        assert line in summary, (line, summary)
    assert "%s  passed" % MODEL.ljust(31) in read(bench, "summary.txt")
    # Each particle the model takes in the barrel, shot from (2, 0, 0) mm at
    # 100 GeV, up and tilted towards the beam, an event each
    shots = json.loads(read(bench, MODEL, "barrel", "shots.json"))
    assert [(s["particle"], s["direction"]) for s in shots] == [("gamma", "up"),
                                                                ("gamma", "tilted")]
    assert all(s["position"] == [2.0, 0.0, 0.0] for s in shots)
    assert shots[0]["momentum"] == pytest.approx([0, 100, 0])
    assert shots[1]["momentum"] == pytest.approx([0, 100 / 1.25 ** 0.5, 50 / 1.25 ** 0.5])
    # simulated as the models stage simulates, with the model's presets
    log = read(bench, MODEL, "barrel", "ddsim.log")
    assert "--ml-model CC3_BARREL --ml-model CC3_ENDCAP" in log and "--random.seed=1" in log
    # and the record that it passed, as it is now
    record = json.loads(read(bench, MODEL, marker))
    assert record["status"] == "passed" and marker == "passed_%s.json" % record["digest"]

    # Done, until it is forced
    status, output = bench.run("preflight_models.sh")
    assert "everything selected is done" in output, output
    status, output = bench.run("preflight_models.sh", "-f", MODEL)
    assert "made 1 of 1" in output, output


def test_python_code_must_conform_to_black_and_import(bench):
    pytest.importorskip("black")
    own_model(bench, "unformatted", UNFORMATTED)
    own_model(bench, "unimportable", FORMATTED + '\n\nraise RuntimeError("no weights here")\n')
    own_model(bench, "no_entry", "def something_else(inputs):\n    return inputs[0]\n")
    own_model(bench, "fine", FORMATTED)
    status, output = bench.run("preflight_models.sh", check=False)
    assert status == 1, output
    summary = read(bench, "summary.txt")
    for name, step in (("unformatted", "3. Python code"), ("unimportable", "4. import"),
                       ("no_entry", "4. import")):
        assert "%s FAILED at %s" % (name.ljust(32), step) in summary, summary
    for name in (MODEL, "fine"):
        assert "%s passed" % name.ljust(32) in summary, summary
    # What Black would change is in the full log
    assert "models/unformatted.py would be reformatted by Black" in read(
        bench, "unformatted", "summary.txt")
    assert "+def run_inference(inputs):" in read(bench, "unformatted", "preflight.log")
    # and the traceback of the import
    assert "RuntimeError: no weights here" in read(bench, "unimportable", "preflight.log")
    assert "no_entry has no function run_inference, which DDML calls" in read(
        bench, "no_entry", "summary.txt")
    # Steps after the one failing aren't run
    assert "4. import       not run" in read(bench, "unformatted", "summary.txt")
    assert "barrel" not in bench.files("preflight_output", "unformatted")


def test_the_model_must_place_showers_that_deposit_energy(bench):
    control(bench, "ddsim_no_showers")
    status, output = bench.run("preflight_models.sh", check=False)
    assert status == 1, output
    summary = read(bench, MODEL, "summary.txt")
    assert "FAILED at step 5, barrel" in summary
    assert "gamma up: DDML placed no shower of the model (Geant4 simulated it)" in summary
    # The plots are there to look at, whatever happened
    assert "barrel_gamma.png" in bench.files("preflight_output", MODEL)
    assert not [name for name in bench.files("preflight_output", MODEL)
                if name.startswith("passed_")]

    control(bench, "ddsim_no_showers", False)
    control(bench, "ddsim_no_hits")
    status, output = bench.run("preflight_models.sh", check=False)
    assert status == 1, output
    assert "gamma tilted: no energy was deposited in" in read(bench, MODEL, "summary.txt")

    control(bench, "ddsim_no_hits", False)
    status, output = bench.run("preflight_models.sh")
    assert "made 1 of 1" in output, output


def test_a_model_outside_the_barrel_has_no_shots(bench):
    text = open(os.path.join(REPO, "models", MODEL + ".yaml")).read()
    bench.add_model("endcap", text.replace("presets: [CC3_BARREL, CC3_ENDCAP]",
                                           "presets: [CC3_ENDCAP]"))
    status, output = bench.run("preflight_models.sh", "endcap")
    summary = read(bench, "endcap", "summary.txt")
    assert "5. barrel       skipped" in summary and "no showers in the ECal barrel" in summary
    assert "barrel_gamma.png" not in bench.files("preflight_output", "endcap")


# ---------------------------------------------------------------------------
# Its place in the pipeline


def test_a_models_simulations_wait_for_its_preflight(bench):
    pytest.importorskip("black")
    own_model(bench, "unformatted", UNFORMATTED)
    bench.run("generate_input.sh", "single_photon")
    status, output = bench.run("simulate_models.sh", "-n", "1", "-s", "single_photon",
                               check=False)
    assert status == 1, output
    # The model that failed is skipped, the other simulated
    assert "skipped, as it needs the preflight of unformatted" in output, output
    assert bench.files("sim_output", "single_photon") == [MODEL]
    assert "the preflight of the models is summed up in" in output, output

    # Fixed, it passes and is simulated, and the other isn't checked again
    with open(os.path.join(bench.repo, "models", "unformatted.py"), "w") as stream:
        stream.write(FORMATTED)
    status, output = bench.run("simulate_models.sh", "-n", "1", "-s", "single_photon")
    assert "made 3 of 3" in output, output
    assert bench.files("sim_output", "single_photon") == [MODEL, "unformatted"]

    # Only the models stage, still with the preflight; -f simulates again, but
    # doesn't check again
    status, output = bench.run("pipeline.sh", "--stages", "models", "-n", "1", "-s",
                               "single_photon", "--dry-run", "-f", "models")
    assert "would run 4 task(s)" in output and "preflight    " not in output.split(
        "would run")[1], output


def test_a_model_changed_is_checked_again_without_simulating_again(config_env):
    from ddml_bench.layout import Layout
    graph = Graph(make_plan())
    layout = Layout(graph.plan["data_dir"])
    simulations = [t["output"] for t in graph.list if t["stage"] == "models"]
    inputs = [t["output"] for t in graph.list if t["stage"] == "generate"]
    for path in simulations + inputs:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "w").close()
    # A preflight the model passed before it changed
    os.makedirs(layout.preflight_model_dir(MODEL))
    open(layout.preflight_passed(MODEL, "0" * 16), "w").close()
    graph = Graph(make_plan()).find_status()
    preflight_task, = [t for t in graph.list if t["stage"] == "preflight"]
    assert preflight_task["output"] != layout.preflight_passed(MODEL, "0" * 16)
    # What was simulated is done, so it needn't be checked again for that
    assert all(t["status"] == DONE for t in graph.list if t["stage"] == "models")
    assert preflight_task["status"] == UNNEEDED
    # but run on its own, it checks the model
    graph = Graph(make_plan(stages=["preflight"])).find_status()
    task, = graph.list[:1]
    assert task["stage"] == "preflight" and task["status"] == TODO
    # and forced, it makes nothing after it again
    graph = Graph(make_plan(force=["preflight"])).find_status()
    assert [t["status"] for t in graph.list if t["stage"] == "preflight"] == [TODO]
    assert all(t["status"] == DONE for t in graph.list if t["stage"] == "models")


def test_the_settings_are_checked_when_planning(config_env, monkeypatch):
    override = os.path.join(config_env.tmp, "bad.yaml")
    with open(override, "w") as stream:
        stream.write("preflight:\n  barrel:\n    energy: lots\n")
    monkeypatch.setenv("FASINATE_CONFIG", config_env.config + ":" + override)
    with pytest.raises(PlanError, match="preflight.barrel.energy"):
        make_plan()
