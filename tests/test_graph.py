"""The task graph: its tasks, what each needs, and their status, worked out
in this process with the tests' configuration (no environment needed)."""

import collections
import os

import pytest

from ddml_bench import config
from ddml_bench.config import ConfigError
from ddml_bench.graph import BLOCKED, DONE, TODO, UNSELECTED, Graph, PlanError, make_plan
from ddml_bench.stages import BY_NAME


@pytest.fixture
def configured(config_env, monkeypatch):
    # Not the configs/local.yaml of whoever runs the tests
    monkeypatch.setattr(config, "LOCAL_FILE", os.path.join(config_env.tmp, "no-local.yaml"))
    return config_env


def touch(*paths):
    for path in paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "w").close()


def outputs(graph, stage):
    return [task["output"] for task in graph.list if task["stage"] == stage]


def status(graph, stage):
    return collections.Counter(task["status"] for task in graph.list if task["stage"] == stage)


def test_a_task_per_file_each_after_what_it_needs(configured):
    graph = Graph(make_plan())
    # single_photon's 2 inputs, and di_photon's 1, simulated with 2 seeds
    assert collections.Counter(task["stage"] for task in graph.list) == {
        "generate": 3, "geant4": 4, "models": 4, "reconstruct": 8, "observables": 2, "plot": 1}
    seen = set()
    for task in graph.list:
        assert set(task["needs"]) <= seen, "%s comes before what it needs" % task["key"]
        for key in task["needs"]:
            assert graph.by_key[key]["output"] in task["inputs"]
        seen.add(task["key"])
    assert len(set(task["output"] for task in graph.list)) == len(graph.list)


def test_with_nothing_on_disk_everything_is_to_do(configured):
    graph = Graph(make_plan()).find_status()
    assert all(task["status"] == TODO for task in graph.list)
    plot = graph.list[-1]
    assert plot["after"] == [task["key"] for task in graph.list if task["stage"] == "observables"]


def test_test_samples_are_named_alike_at_every_stage(configured):
    graph = Graph(make_plan(events=5))
    for task in graph.list:
        if task["stage"] != "generate":
            assert "_n5" in os.path.basename(task["output"]), task["output"]
        assert "_n5_n5" not in task["output"]
    for task in graph.list:
        if task["stage"] == "reconstruct":
            simulation = graph.by_key[task["needs"][0]]
            assert task["output"].endswith(simulation["name"] + "_REC.edm4hep.root")
            assert task["params"]["events"] == -1


def test_what_is_on_disk_is_done(configured):
    graph = Graph(make_plan())
    touch(*outputs(graph, "generate") + outputs(graph, "geant4"))
    graph.find_status()
    assert status(graph, "generate") == {DONE: 3}
    assert status(graph, "geant4") == {DONE: 4}
    assert status(graph, "models") == {TODO: 4}
    assert status(graph, "reconstruct") == {TODO: 8}


def test_forcing_a_stage_remakes_what_is_made_from_it(configured):
    graph = Graph(make_plan())
    touch(*[path for stage in ("generate", "geant4", "models", "reconstruct")
            for path in outputs(graph, stage)])
    graph = Graph(make_plan(force=["geant4"])).find_status()
    assert status(graph, "geant4") == {TODO: 4}
    assert all(task["redo"] for task in graph.list if task["stage"] == "geant4")
    remade = [task for task in graph.list if task["stage"] == "reconstruct" and task["redo"]]
    assert sorted(task["simulator"] for task in remade) == ["geant4"] * 4
    assert all("is made again" in task["why"] for task in remade)
    assert status(graph, "models") == {DONE: 4}
    measured = dict((task["simulator"], task) for task in graph.list
                    if task["stage"] == "observables")
    assert measured["geant4"]["why"].endswith("is made again")
    assert measured["caloclouds3"]["why"] == "no caloclouds3.npz yet"


def test_what_isnt_selected_isnt_run(configured):
    graph = Graph(make_plan(simulators=["geant4"])).find_status()
    assert status(graph, "models") == {UNSELECTED: 4}
    assert status(graph, "geant4") == {TODO: 4}
    # The plots are of what there is, so they don't wait for the models
    assert graph.list[-1]["status"] == TODO

    graph = Graph(make_plan(stages=["geant4"])).find_status()
    assert status(graph, "geant4") == {BLOCKED: 4}
    assert all("isn't selected to run" in task["why"] for task in graph.list
               if task["stage"] == "geant4")


def test_only_the_inputs_named(configured):
    graph = Graph(make_plan())
    inputs = outputs(graph, "generate")
    touch(*inputs)
    graph = Graph(make_plan(stages=["geant4"], files=[inputs[1]])).find_status()
    selected = [task["input_file"] for task in graph.list
                if task["stage"] == "geant4" and task["status"] == TODO]
    assert selected == [inputs[1]]
    with pytest.raises(PlanError, match="isn't an input of any scenario"):
        make_plan(files=[os.path.join(configured.tmp, "test.yaml")])


def test_each_seed_of_a_scenario_is_a_simulation_of_its_own(configured, monkeypatch):
    """di_photon's input is simulated once with each of its simulation_seeds,
    and everything made from each simulation carries its seed on."""
    monkeypatch.setenv("DDML_SOURCE_DIR", "/ddml")  # where the models' steering file is
    graph = Graph(make_plan())
    simulations = [task for task in graph.list
                   if task["stage"] in ("geant4", "models") and task["scenario"] == "di_photon"]
    assert sorted((task["simulator"], task["seed"]) for task in simulations) == [
        ("caloclouds3", 1), ("caloclouds3", 2), ("geant4", 1), ("geant4", 2)]
    for task in simulations:
        assert task["name"].endswith("_seed0_simseed%d" % task["seed"])
        assert os.path.basename(task["output"]).startswith(task["name"] + ".")
        assert task["params"]["log"].endswith(task["name"] + ".log")
        command = BY_NAME[task["stage"]].command(task, "/detector.xml", "/out/.partial.root")
        assert command[-1] == "--random.seed=%d" % task["seed"]
    # single_photon's inputs are simulated once, with ddsim's own seed
    for task in graph.list:
        if task["stage"] == "geant4" and task["scenario"] == "single_photon":
            assert task["seed"] is None and "_simseed" not in task["output"]
            command = BY_NAME["geant4"].command(task, "/detector.xml", "/out/.partial.root")
            assert not any(arg.startswith("--random.seed") for arg in command)
    for task in graph.list:
        if task["stage"] == "reconstruct":
            simulation = graph.by_key[task["needs"][0]]
            assert task["seed"] == simulation["seed"]
            assert task["output"].endswith(simulation["name"] + "_REC.edm4hep.root")
    measuring = [task for task in graph.list if task["stage"] == "observables"]
    for task in measuring:
        assert task["params"]["seeds"] == {"single_photon": [None, None], "di_photon": [1, 2]}
        assert [os.path.basename(f) for f in task["params"]["files"]["di_photon"]] == [
            "diphoton_en20_sep90_x0_z70_r10_seed0_simseed%d_REC.edm4hep.root" % seed
            for seed in (1, 2)]


@pytest.mark.parametrize("seeds, message", [
    ("[0, 1]", "simulation_seeds should be a list of ddsim's random seeds"),
    ("3", "simulation_seeds should be a list of ddsim's random seeds"),
    ("[]", "simulation_seeds should be a list of ddsim's random seeds"),
    ("[1, 2, 1]", "simulation_seeds has a seed twice: 1, 2, 1"),
])
def test_the_simulation_seeds_are_checked(configured, monkeypatch, seeds, message):
    override = os.path.join(configured.tmp, "seeds.yaml")
    with open(override, "w") as stream:
        stream.write("scenarios:\n  di_photon:\n    simulation_seeds: %s\n" % seeds)
    monkeypatch.setenv("FASINATE_CONFIG", configured.config + ":" + override)
    with pytest.raises(ConfigError, match=message):
        make_plan()


def test_a_simulation_given_is_the_graph_task_making_it(configured):
    graph = Graph(make_plan())
    simulation = outputs(graph, "geant4")[0]
    touch(simulation)
    graph = Graph(make_plan(stages=["reconstruct"], simulations=[simulation])).find_status()
    todo = [task for task in graph.list if task["status"] == TODO]
    assert len(todo) == 1 and todo[0]["inputs"] == [simulation] and todo[0]["key"] in graph.given


@pytest.mark.parametrize("options, message", [
    ({"stages": ["simulate"]}, "there's no stage 'simulate'"),
    ({"simulators": ["fastsim"]}, "there's no simulator 'fastsim'"),
    ({"scenarios": ["tau"]}, "there is no scenario 'tau'"),
    ({"stages": ["geant4"], "force": ["plot"]}, "isn't one of the --stages"),
    ({"events": 0}, "at least 1"),
])
def test_bad_options_are_caught_before_anything_runs(configured, options, message):
    with pytest.raises(PlanError, match=message):
        make_plan(**options)
