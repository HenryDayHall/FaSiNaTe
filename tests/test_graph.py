"""The task graph: its tasks, what each needs, and their status, worked out
in this process with the tests' configuration (no environment needed)."""

import collections
import os
import shutil

import pytest

from ddml_bench import config, measure, models
from ddml_bench.config import ConfigError
from ddml_bench.graph import (BLOCKED, DONE, TODO, UNNEEDED, UNSELECTED, Graph, PlanError,
                              make_plan)
from ddml_bench.layout import Layout
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


def status(graph, stage, simulator=None):
    return collections.Counter(task["status"] for task in graph.list if task["stage"] == stage
                               and simulator in (None, task["simulator"]))


def measure_all(graph, monkeypatch):
    """Write each simulator's .npz file as the observables stage does, up to
    date for its task, without reading any reconstructions (as if fetched):
    every key it needs, with nothing measured in them."""
    import numpy as np

    def nothing(path, observables, profiles, simulator, cache):
        sums = dict((key, [np.zeros(len(edges) - 1), np.zeros(len(edges) - 1)])
                    for key, edges in profiles.items())
        events = dict((name, dict((what, np.zeros(0))
                                  for what in measure.event_items(name, settings)))
                      for name, settings in observables.items() if name in measure.PER_EVENT)
        return sums, events, 0, 0, float("nan")

    with monkeypatch.context() as patched:
        patched.setattr(measure, "measure_file", nothing)
        for task in graph.list:
            if task["stage"] == "observables":
                params = task["params"]
                measure.compute_simulator(params["observables"], params["simulator"],
                                          params["files"], measure.wanted(params["observables"]),
                                          task["output"], Layout(params["data_dir"]).relative,
                                          seeds=params["seeds"])


@pytest.fixture
def fetched(configured, monkeypatch):
    """The data folder with only the inputs and each simulator's .npz file, as
    fetched from where they are published."""
    graph = Graph(make_plan())
    touch(*outputs(graph, "generate"))
    measure_all(graph, monkeypatch)
    return configured


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


def test_up_to_date_measurements_need_no_simulations_or_reconstructions(fetched):
    """The inputs and the .npz files, as fetched: nothing to simulate,
    reconstruct or measure, only the plots."""
    graph = Graph(make_plan()).find_status()
    assert status(graph, "generate") == {DONE: 3}
    assert status(graph, "geant4") == status(graph, "models") == {UNNEEDED: 4}
    assert status(graph, "reconstruct") == {UNNEEDED: 8}
    assert status(graph, "observables") == {DONE: 2}
    plot, = [task for task in graph.list if task["stage"] == "plot"]
    assert plot["status"] == TODO and plot["after"] == []
    for task in graph.list:
        if task["status"] == UNNEEDED:
            assert task["why"] == "what it is for is done: the measurements of %s" \
                % task["simulator"]
            assert task["after"] == [] and not task["redo"]


def test_a_model_added_runs_only_its_own_tasks(fetched, monkeypatch):
    folder = os.path.join(fetched.tmp, "models")
    os.makedirs(folder)
    for name in ("caloclouds3", "newmodel"):
        shutil.copy(os.path.join(models.MODELS_DIR, "caloclouds3.yaml"),
                    os.path.join(folder, name + ".yaml"))
    monkeypatch.setattr(models, "MODELS_DIR", folder)
    graph = Graph(make_plan()).find_status()
    assert status(graph, "geant4") == {UNNEEDED: 4}
    assert status(graph, "models", "caloclouds3") == {UNNEEDED: 4}
    assert status(graph, "models", "newmodel") == {TODO: 4}
    assert status(graph, "reconstruct", "newmodel") == {TODO: 4}
    assert status(graph, "reconstruct") == {UNNEEDED: 8, TODO: 4}
    assert status(graph, "observables") == {DONE: 2, TODO: 1}
    plot, = [task for task in graph.list if task["stage"] == "plot"]
    assert plot["status"] == TODO and plot["after"] == [
        task["key"] for task in graph.list
        if task["stage"] == "observables" and task["simulator"] == "newmodel"]


def test_forcing_a_stage_makes_what_isnt_there_and_what_is_made_from_it(fetched):
    graph = Graph(make_plan(force=["geant4"])).find_status()
    assert status(graph, "geant4") == {TODO: 4}
    assert not any(task["redo"] for task in graph.list if task["stage"] == "geant4")
    assert status(graph, "reconstruct", "geant4") == {TODO: 4}
    assert all(task["why"].endswith("is made again") for task in graph.list
               if task["stage"] == "reconstruct" and task["simulator"] == "geant4")
    measured = dict((task["simulator"], task) for task in graph.list
                    if task["stage"] == "observables")
    assert measured["geant4"]["status"] == TODO and measured["geant4"]["redo"]
    # CaloClouds3's measurements are up to date, so its tasks aren't needed
    assert measured["caloclouds3"]["status"] == DONE
    assert status(graph, "models") == {UNNEEDED: 4}
    assert status(graph, "reconstruct", "caloclouds3") == {UNNEEDED: 4}


def test_measurements_out_of_date_need_their_reconstructions(fetched, monkeypatch):
    override = os.path.join(fetched.tmp, "photons.yaml")
    with open(override, "w") as stream:
        stream.write("observables:\n  energy:\n    pdg: [22]\n")
    monkeypatch.setenv("FASINATE_CONFIG", fetched.config + ":" + override)
    graph = Graph(make_plan()).find_status()
    assert [task["why"] for task in graph.list if task["stage"] == "observables"] == [
        "the settings of energy changed"] * 2
    assert status(graph, "geant4") == status(graph, "models") == {TODO: 4}
    assert status(graph, "reconstruct") == {TODO: 8}


def test_whichever_stages_run_what_isnt_needed_isnt_made(fetched):
    """The stage scripts (e.g. simulate_geant4.sh) make only what is needed
    too, unless forced."""
    graph = Graph(make_plan(stages=["geant4"])).find_status()
    assert status(graph, "geant4") == {UNNEEDED: 4}
    assert status(graph, "reconstruct") == {UNSELECTED: 8}
    # Not blocked, though their simulations aren't there
    graph = Graph(make_plan(stages=["reconstruct"])).find_status()
    assert status(graph, "reconstruct") == {UNNEEDED: 8}
    graph = Graph(make_plan(stages=["geant4"], force=["geant4"])).find_status()
    assert status(graph, "geant4") == {TODO: 4}


def test_the_inputs_and_a_simulation_given_are_made_whatever_needs_them(fetched):
    """Only the simulations and reconstructions are intermediate: the inputs
    are the benchmark's events, and a simulation given is to be reconstructed."""
    graph = Graph(make_plan())
    for path in outputs(graph, "generate"):
        os.remove(path)
    graph.find_status()
    assert status(graph, "generate") == {TODO: 3}
    assert status(graph, "geant4") == {UNNEEDED: 4}
    simulation = outputs(graph, "geant4")[0]
    touch(simulation)
    graph = Graph(make_plan(stages=["reconstruct"], simulations=[simulation])).find_status()
    assert [task["inputs"] for task in graph.tasks(TODO)] == [[simulation]]


def test_with_nothing_measured_a_stage_run_on_its_own_makes_all_of_it(configured):
    """The measurements still to be made need every simulation."""
    graph = Graph(make_plan())
    touch(*outputs(graph, "generate"))
    graph = Graph(make_plan(stages=["geant4"])).find_status()
    assert status(graph, "geant4") == {TODO: 4}


def test_simulations_deleted_once_reconstructed_arent_made_again(configured, monkeypatch):
    graph = Graph(make_plan())
    touch(*[path for stage in ("generate", "geant4", "models", "reconstruct")
            for path in outputs(graph, stage)])
    measure_all(graph, monkeypatch)
    for path in outputs(graph, "geant4"):
        os.remove(path)
    graph = Graph(make_plan()).find_status()
    assert status(graph, "geant4") == {UNNEEDED: 4}
    assert status(graph, "models") == {DONE: 4}
    assert status(graph, "reconstruct") == {DONE: 8}
    assert status(graph, "observables") == {DONE: 2}
    for task in graph.list:
        if task["stage"] == "geant4":
            assert task["why"] == "what it is for is done: the reconstruction of the geant4 " \
                "simulation of %s" % task["name"]


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
