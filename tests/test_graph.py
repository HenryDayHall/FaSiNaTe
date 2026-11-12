"""The task graph: its tasks, what each needs, and their status, worked out
in this process with the tests' configuration (no environment needed)."""

import collections
import os
import re
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
    # single_photon's 2 inputs, and di_photon's 1, simulated with 2 seeds, by
    # Geant4 twice (its second sample) and by the model
    assert collections.Counter(task["stage"] for task in graph.list) == {
        "generate": 3, "geant4": 8, "models": 4, "reconstruct": 12, "observables": 3, "plot": 1}
    assert collections.Counter(task["simulator"] for task in graph.list
                               if task["stage"] == "geant4") == {"geant4": 4, "geant4_2": 4}
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
    assert status(graph, "geant4") == {DONE: 8}
    assert status(graph, "models") == {TODO: 4}
    assert status(graph, "reconstruct") == {TODO: 12}


def test_forcing_a_stage_remakes_what_is_made_from_it(configured):
    graph = Graph(make_plan())
    touch(*[path for stage in ("generate", "geant4", "models", "reconstruct")
            for path in outputs(graph, stage)])
    graph = Graph(make_plan(force=["geant4"])).find_status()
    assert status(graph, "geant4") == {TODO: 8}
    assert all(task["redo"] for task in graph.list if task["stage"] == "geant4")
    remade = [task for task in graph.list if task["stage"] == "reconstruct" and task["redo"]]
    assert sorted(task["simulator"] for task in remade) == ["geant4"] * 4 + ["geant4_2"] * 4
    assert all("is made again" in task["why"] for task in remade)
    assert status(graph, "models") == {DONE: 4}
    measured = dict((task["simulator"], task) for task in graph.list
                    if task["stage"] == "observables")
    assert measured["geant4"]["why"].endswith("is made again")
    assert measured["geant4_2"]["why"].endswith("is made again")
    assert measured["caloclouds3_paper"]["why"] == "no caloclouds3_paper.npz yet"


def test_what_isnt_selected_isnt_run(configured):
    graph = Graph(make_plan(simulators=["geant4"])).find_status()
    assert status(graph, "models") == {UNSELECTED: 4}
    assert status(graph, "geant4", "geant4") == {TODO: 4}
    # Its second sample is a simulator of its own
    assert status(graph, "geant4", "geant4_2") == {UNSELECTED: 4}
    # The plots are of what there is, so they don't wait for the models
    assert graph.list[-1]["status"] == TODO
    graph = Graph(make_plan(simulators=["geant4_2"])).find_status()
    assert status(graph, "geant4", "geant4_2") == {TODO: 4}
    assert status(graph, "geant4", "geant4") == {UNSELECTED: 4}

    graph = Graph(make_plan(stages=["geant4"])).find_status()
    assert status(graph, "geant4") == {BLOCKED: 8}
    assert all("isn't selected to run" in task["why"] for task in graph.list
               if task["stage"] == "geant4")


def test_up_to_date_measurements_need_no_simulations_or_reconstructions(fetched):
    """The inputs and the .npz files, as fetched: nothing to simulate,
    reconstruct or measure, only the plots."""
    graph = Graph(make_plan()).find_status()
    assert status(graph, "generate") == {DONE: 3}
    assert status(graph, "geant4") == {UNNEEDED: 8}
    assert status(graph, "models") == {UNNEEDED: 4}
    assert status(graph, "reconstruct") == {UNNEEDED: 12}
    assert status(graph, "observables") == {DONE: 3}
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
    for name in ("caloclouds3_paper", "newmodel"):
        shutil.copy(os.path.join(models.MODELS_DIR, "caloclouds3_paper.yaml"),
                    os.path.join(folder, name + ".yaml"))
    monkeypatch.setattr(models, "MODELS_DIR", folder)
    graph = Graph(make_plan()).find_status()
    assert status(graph, "geant4") == {UNNEEDED: 8}
    assert status(graph, "models", "caloclouds3_paper") == {UNNEEDED: 4}
    assert status(graph, "models", "newmodel") == {TODO: 4}
    assert status(graph, "reconstruct", "newmodel") == {TODO: 4}
    assert status(graph, "reconstruct") == {UNNEEDED: 12, TODO: 4}
    assert status(graph, "observables") == {DONE: 3, TODO: 1}
    plot, = [task for task in graph.list if task["stage"] == "plot"]
    assert plot["status"] == TODO and plot["after"] == [
        task["key"] for task in graph.list
        if task["stage"] == "observables" and task["simulator"] == "newmodel"]


def test_forcing_a_stage_makes_what_isnt_there_and_what_is_made_from_it(fetched):
    graph = Graph(make_plan(force=["geant4"])).find_status()
    assert status(graph, "geant4") == {TODO: 8}
    assert not any(task["redo"] for task in graph.list if task["stage"] == "geant4")
    assert status(graph, "reconstruct", "geant4") == {TODO: 4}
    assert status(graph, "reconstruct", "geant4_2") == {TODO: 4}
    assert all(task["why"].endswith("is made again") for task in graph.list
               if task["stage"] == "reconstruct" and task["simulator"] in ("geant4", "geant4_2"))
    measured = dict((task["simulator"], task) for task in graph.list
                    if task["stage"] == "observables")
    assert measured["geant4"]["status"] == TODO and measured["geant4"]["redo"]
    assert measured["geant4_2"]["status"] == TODO and measured["geant4_2"]["redo"]
    # CaloClouds3's measurements are up to date, so its tasks aren't needed
    assert measured["caloclouds3_paper"]["status"] == DONE
    assert status(graph, "models") == {UNNEEDED: 4}
    assert status(graph, "reconstruct", "caloclouds3_paper") == {UNNEEDED: 4}


def test_measurements_out_of_date_need_their_reconstructions(fetched, monkeypatch):
    override = os.path.join(fetched.tmp, "photons.yaml")
    with open(override, "w") as stream:
        stream.write("observables:\n  energy:\n    pdg: [22]\n")
    monkeypatch.setenv("FASINATE_CONFIG", fetched.config + ":" + override)
    graph = Graph(make_plan()).find_status()
    assert [task["why"] for task in graph.list if task["stage"] == "observables"] == [
        "the settings of energy changed"] * 3
    assert status(graph, "geant4") == {TODO: 8}
    assert status(graph, "models") == {TODO: 4}
    assert status(graph, "reconstruct") == {TODO: 12}


def test_whichever_stages_run_what_isnt_needed_isnt_made(fetched):
    """The stage scripts (e.g. simulate_geant4.sh) make only what is needed
    too, unless forced."""
    graph = Graph(make_plan(stages=["geant4"])).find_status()
    assert status(graph, "geant4") == {UNNEEDED: 8}
    assert status(graph, "reconstruct") == {UNSELECTED: 12}
    # Not blocked, though their simulations aren't there
    graph = Graph(make_plan(stages=["reconstruct"])).find_status()
    assert status(graph, "reconstruct") == {UNNEEDED: 12}
    graph = Graph(make_plan(stages=["geant4"], force=["geant4"])).find_status()
    assert status(graph, "geant4") == {TODO: 8}


def test_the_inputs_and_a_simulation_given_are_made_whatever_needs_them(fetched):
    """Only the simulations and reconstructions are intermediate: the inputs
    are the benchmark's events, and a simulation given is to be reconstructed."""
    graph = Graph(make_plan())
    for path in outputs(graph, "generate"):
        os.remove(path)
    graph.find_status()
    assert status(graph, "generate") == {TODO: 3}
    assert status(graph, "geant4") == {UNNEEDED: 8}
    simulation = outputs(graph, "geant4")[0]
    touch(simulation)
    graph = Graph(make_plan(stages=["reconstruct"], simulations=[simulation])).find_status()
    assert [task["inputs"] for task in graph.tasks(TODO)] == [[simulation]]


def test_with_nothing_measured_a_stage_run_on_its_own_makes_all_of_it(configured):
    """The measurements still to be made need every simulation."""
    graph = Graph(make_plan())
    touch(*outputs(graph, "generate"))
    graph = Graph(make_plan(stages=["geant4"])).find_status()
    assert status(graph, "geant4") == {TODO: 8}


def test_simulations_deleted_once_reconstructed_arent_made_again(configured, monkeypatch):
    graph = Graph(make_plan())
    touch(*[path for stage in ("generate", "geant4", "models", "reconstruct")
            for path in outputs(graph, stage)])
    measure_all(graph, monkeypatch)
    for path in outputs(graph, "geant4"):
        os.remove(path)
    graph = Graph(make_plan()).find_status()
    assert status(graph, "geant4") == {UNNEEDED: 8}
    assert status(graph, "models") == {DONE: 4}
    assert status(graph, "reconstruct") == {DONE: 12}
    assert status(graph, "observables") == {DONE: 3}
    for task in graph.list:
        if task["stage"] == "geant4":
            assert task["why"] == "what it is for is done: the reconstruction of the %s " \
                "simulation of %s" % (task["simulator"], task["name"])


def test_only_the_inputs_named(configured):
    graph = Graph(make_plan())
    inputs = outputs(graph, "generate")
    touch(*inputs)
    graph = Graph(make_plan(stages=["geant4"], files=[inputs[1]])).find_status()
    selected = [(task["simulator"], task["input_file"]) for task in graph.list
                if task["stage"] == "geant4" and task["status"] == TODO]
    assert selected == [("geant4", inputs[1]), ("geant4_2", inputs[1])]
    with pytest.raises(PlanError, match="isn't an input of any scenario"):
        make_plan(files=[os.path.join(configured.tmp, "test.yaml")])


def test_inputs_given_directly_are_simulated_once(configured):
    """Simulated as the scenario custom, which nothing measures, so not as
    Geant4's second sample."""
    given = os.path.join(configured.tmp, "mine.slcio")
    touch(given)
    graph = Graph(make_plan(stages=["geant4"], inputs=[given])).find_status()
    assert [(task["simulator"], task["seed"]) for task in graph.list
            if task["stage"] == "geant4" and task["scenario"] == "custom"] == [("geant4", None)]


def test_each_seed_of_a_scenario_is_a_simulation_of_its_own(configured, monkeypatch):
    """di_photon's input is simulated once with each of its simulation_seeds,
    and everything made from each simulation carries its seed on; the second
    Geant4 sample has seeds of its own, the first's plus 1000000."""
    monkeypatch.setenv("DDML_SOURCE_DIR", "/ddml")  # where the models' steering file is
    graph = Graph(make_plan())
    simulations = [task for task in graph.list
                   if task["stage"] in ("geant4", "models") and task["scenario"] == "di_photon"]
    assert sorted((task["simulator"], task["seed"]) for task in simulations) == [
        ("caloclouds3_paper", 1), ("caloclouds3_paper", 2), ("geant4", 1), ("geant4", 2),
        ("geant4_2", 1000001), ("geant4_2", 1000002)]
    for task in simulations:
        assert task["name"].endswith("_seed0_simseed%d" % task["seed"])
        assert os.path.basename(task["output"]).startswith(task["name"] + ".")
        assert task["params"]["log"].endswith(task["name"] + ".log")
        command = BY_NAME[task["stage"]].command(task, "/detector.xml", "/out/.partial.root")
        assert command[-1] == "--random.seed=%d" % task["seed"]
    # single_photon's inputs are simulated once, with ddsim's own seed, and
    # again for the second sample, with the seed 1000000
    for task in graph.list:
        if task["stage"] == "geant4" and task["scenario"] == "single_photon":
            command = BY_NAME["geant4"].command(task, "/detector.xml", "/out/.partial.root")
            if task["simulator"] == "geant4":
                assert task["seed"] is None and "_simseed" not in task["output"]
                assert not any(arg.startswith("--random.seed") for arg in command)
            else:
                assert task["seed"] == 1000000
                assert task["output"].endswith("_simseed1000000.edm4hep.root")
                assert command[-1] == "--random.seed=1000000"
    for task in graph.list:
        if task["stage"] == "reconstruct":
            simulation = graph.by_key[task["needs"][0]]
            assert task["seed"] == simulation["seed"]
            assert task["output"].endswith(simulation["name"] + "_REC.edm4hep.root")
    measuring = dict((task["simulator"], task) for task in graph.list
                     if task["stage"] == "observables")
    for name, seeds in (("geant4", {"single_photon": [None, None], "di_photon": [1, 2]}),
                        ("caloclouds3_paper", {"single_photon": [None, None],
                                               "di_photon": [1, 2]}),
                        ("geant4_2", {"single_photon": [1000000, 1000000],
                                      "di_photon": [1000001, 1000002]})):
        task = measuring[name]
        assert task["params"]["seeds"] == seeds
        assert [os.path.basename(f) for f in task["params"]["files"]["di_photon"]] == [
            "diphoton_en20_sep90_x0_z70_r10_seed0_simseed%d_REC.edm4hep.root" % seed
            for seed in seeds["di_photon"]]
    assert measuring["geant4_2"]["params"]["simulator"] == {
        "name": "geant4_2", "label": "Geant4 (2nd sample)", "color": "#1a1a1a",
        "linestyle": "--", "compact_file": measuring["geant4"]["params"]["simulator"][
            "compact_file"]}
    # Its simulations are the geant4 stage's, run as the first's are, which
    # say which they are
    second = next(task for task in simulations if task["simulator"] == "geant4_2")
    first = next(task for task in simulations if task["simulator"] == "geant4"
                 and task["seed"] == second["seed"] - 1000000)
    stage = BY_NAME["geant4"]
    assert stage.command(second, "/d.xml", "/o.root")[:-1] == \
        stage.command(first, "/d.xml", "/o.root")[:-1]
    assert stage.describe(second) == "the second Geant4 simulation of %s" % second["name"]
    assert stage.header(second) == [
        "# ddml-bench geant4: Geant4's second sample (geant4_2) on di_photon"]
    assert (stage.job_name(second), stage.job_name(first)) == ("geant4_2", "geant4")


@pytest.mark.parametrize("seeds, message", [
    ("[0, 1]", "simulation_seeds should be a list of ddsim's random seeds"),
    # (the second Geant4 sample takes each seed plus 1000000)
    ("[1, 1000000]", "below 1000000 .* the second Geant4 sample is simulated with each plus "
                     "1000000"),
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


# ---------------------------------------------------------------------------
# Inputs made elsewhere (input type external): tau_pairs, enabled


@pytest.fixture
def taus(configured, monkeypatch):
    """tau_pairs enabled, with its two inputs there."""
    configured.with_tau_pairs()
    monkeypatch.setenv("FASINATE_CONFIG", configured.env["FASINATE_CONFIG"])
    return configured


@pytest.fixture
def no_taus(configured, monkeypatch):
    """tau_pairs enabled, without its inputs."""
    configured.with_tau_pairs(write=False)
    monkeypatch.setenv("FASINATE_CONFIG", configured.env["FASINATE_CONFIG"])
    return configured


def test_inputs_made_elsewhere_are_done_and_each_simulated_with_its_own_seed(taus,
                                                                            monkeypatch):
    monkeypatch.setenv("DDML_SOURCE_DIR", "/ddml")
    graph = Graph(make_plan()).find_status()
    assert graph.left_out == {}
    inputs = [task for task in graph.list
              if task["stage"] == "generate" and task["scenario"] == "tau_pairs"]
    assert [os.path.relpath(task["output"], taus.data) for task in inputs] == [
        "input_physics_samples/taus_a.slcio", "input_physics_samples/taus_b.slcio"]
    assert all(task["status"] == DONE and task["params"]["command"] is None for task in inputs)
    simulations = [task for task in graph.list
                   if task["stage"] in ("geant4", "models") and task["scenario"] == "tau_pairs"]
    assert sorted((task["simulator"], task["name"], task["seed"]) for task in simulations) == [
        (simulator, name, seed) for simulator in ("caloclouds3_paper", "geant4")
        for name, seed in (("taus_a_simseed42", 42), ("taus_b_simseed1764", 1764))] + [
        ("geant4_2", "taus_a_simseed1000042", 1000042),
        ("geant4_2", "taus_b_simseed1001764", 1001764)]
    for task in simulations:
        command = BY_NAME[task["stage"]].command(task, "/detector.xml", "/out/.partial.root")
        assert command[-1] == "--random.seed=%d" % task["seed"]
    for task in graph.list:
        if task["stage"] == "reconstruct" and task["scenario"] == "tau_pairs":
            assert "--usingParticleGun" not in task["params"]["args"]
            assert "--noBeamCalReco" in task["params"]["args"]
    measuring = [task for task in graph.list if task["stage"] == "observables"]
    for task in measuring:
        assert task["params"]["observables"]["pi0"]["scenarios"] == ["tau_pairs"]
        assert task["params"]["seeds"]["tau_pairs"] == (
            [1000042, 1001764] if task["simulator"] == "geant4_2" else [42, 1764])
    # Forcing the generate stage remakes the guns' inputs, not these
    graph = Graph(make_plan(force=["generate"])).find_status()
    assert collections.Counter((task["scenario"], task["status"]) for task in graph.list
                               if task["stage"] == "generate") == {
        ("single_photon", TODO): 2, ("di_photon", TODO): 1, ("tau_pairs", DONE): 2}


def test_a_scenario_without_all_its_inputs_is_left_out(no_taus):
    plan = make_plan()
    paths = [os.path.join(no_taus.data, "input_physics_samples", name)
             for name in ("taus_a.slcio", "taus_b.slcio")]
    graph = Graph(plan)
    assert graph.left_out == {"tau_pairs": paths}
    assert not [task for task in graph.list if task["scenario"] == "tau_pairs"]
    assert graph.measured == ["single_photon", "di_photon"] and "pi0" not in graph.observables
    for task in graph.list:
        if task["stage"] == "observables":
            assert "pi0" not in task["params"]["observables"]
            assert sorted(task["params"]["files"]) == ["di_photon", "single_photon"]
    assert graph.left_out_reasons() == [
        "the scenario tau_pairs is left out of this run, with the observables measured on it "
        "alone: its 2 inputs, made outside this repository, aren't in %s (taus_a.slcio, "
        "taus_b.slcio); put them there, or publish them in a record of `published`, to run it"
        % os.path.dirname(paths[0])]

    # Kept, as when the inputs to fetch are chosen: an input that isn't there
    # can't be made here
    graph = Graph(plan, keep_unavailable=True).find_status()
    blocked = [task for task in graph.list
               if task["stage"] == "generate" and task["scenario"] == "tau_pairs"]
    assert [task["status"] for task in blocked] == [BLOCKED] * 2
    assert "it is made outside this repository, so it can't be made here" in blocked[0]["why"]
    # Not left out with its inputs about to be fetched, or to be fetched in a dry run
    assert Graph(plan, coming=paths).left_out == {}
    assert Graph(dict(plan, to_fetch=dict((p, "Zenodo record 1") for p in paths))).left_out \
        == {}
    assert Graph(plan, coming=paths[:1]).left_out == {"tau_pairs": paths[1:]}
    # Asking for it, or for one of its inputs, is an error
    with pytest.raises(PlanError, match="the scenario tau_pairs can't run: its 2 inputs"):
        Graph(make_plan(scenarios=["tau_pairs"]))
    touch(paths[0])
    with pytest.raises(PlanError, match="can't run: 1 of its 2 inputs"):
        Graph(make_plan(stages=["geant4"], files=[paths[0]]))


def test_a_disabled_scenario_isnt_in_the_plan(configured):
    """The tests' configuration disables tau_pairs."""
    plan = make_plan()
    assert [group["scenario"] for group in plan["scenarios"]] == ["single_photon", "di_photon"]
    assert plan["disabled"] == ["tau_pairs"]
    assert "pi0" not in plan["observables"]["observables"]
    with pytest.raises(PlanError, match=r"the scenario tau_pairs is disabled \(scenarios"):
        make_plan(scenarios=["tau_pairs"])


@pytest.mark.parametrize("settings, message", [
    ("    input:\n      files: taus.slcio\n", "input.files should be a list of the names"),
    ("    input:\n      files: [sub/taus.slcio]\n", "not 'sub/taus.slcio'"),
    ("    input:\n      files: [taus.hepmc]\n", "taus.hepmc isn't an LCIO file (.slcio)"),
    ("    input:\n      files: [a.slcio, a.slcio]\n", "input.files has a.slcio twice"),
    ("    input:\n      folder: sim_output\n", "can't be sim_output, which the pipeline"),
    ("    input:\n      folder: input/physics\n", "not 'input/physics'"),
    ("    simulation_seeds: [42]\n", "has 3 inputs but 1 simulation_seeds; with "
                                     "one_seed_per_input it needs one seed for each input"),
    ("    simulation_seeds: null\n", "one_seed_per_input needs simulation_seeds"),
    ("    one_seed_per_input: 1\n", "one_seed_per_input should be true or false"),
    ("    enabled: maybe\n", "enabled should be true or false"),
])
def test_a_scenario_of_inputs_made_elsewhere_is_checked(configured, monkeypatch, settings,
                                                        message):
    override = os.path.join(configured.tmp, "taus.yaml")
    with open(override, "w") as stream:
        stream.write("scenarios:\n  tau_pairs:\n" + settings)
    monkeypatch.setenv("FASINATE_CONFIG", configured.config + ":" + override)
    with pytest.raises(ConfigError, match=re.escape(message)):
        make_plan()


def test_inputs_made_elsewhere_take_only_their_files():
    from ddml_bench.scenarios import scenario_inputs
    config = {"scenarios": {"taus": {"input": {"type": "external", "files": ["a.slcio"],
                                               "n_events": 6791}}}}
    with pytest.raises(ConfigError, match="input has n_events, which inputs made elsewhere"):
        scenario_inputs(config, Layout("/data"))
    del config["scenarios"]["taus"]["input"]["n_events"]
    group, = scenario_inputs(config, Layout("/data"))
    assert group["folder"] == "input_physics_samples" and group["enabled"]
    assert group["inputs"] == [{"file": "/data/input_physics_samples/a.slcio", "command": None,
                                "seeds": [None]}]


# ---------------------------------------------------------------------------
# Inputs in segments (input.segments)


@pytest.fixture
def segmented(configured, monkeypatch):
    """di_photon and tau_pairs with their inputs in three segments, without
    tau_pairs' inputs."""
    configured.with_tau_pairs(write=False)
    configured.with_segments(3)
    monkeypatch.setenv("FASINATE_CONFIG", configured.env["FASINATE_CONFIG"])
    return configured


def test_each_segment_is_an_input_simulated_with_the_seeds_of_its_whole(segmented,
                                                                        monkeypatch):
    data = segmented.data
    taus = [os.path.join(data, "input_physics_samples", "taus_%s_seg%dof3.slcio" % (name, i))
            for name in "ab" for i in (1, 2, 3)]
    # Without them, tau_pairs is left out, saying what makes them
    graph = Graph(make_plan())
    assert graph.left_out == {"tau_pairs": taus}
    reason, = graph.left_out_reasons()
    assert "its 6 inputs, made outside this repository, aren't in" in reason, reason
    assert "put them there (or the whole files taus_a.slcio, taus_b.slcio, and split them " \
        "into segments with scripts/split_data.py)" in reason, reason
    touch(*taus)
    monkeypatch.setenv("DDML_SOURCE_DIR", "/ddml")
    graph = Graph(make_plan()).find_status()
    assert graph.left_out == {}
    # The gun makes each segment of di_photon's input, with --segment
    inputs = [task for task in graph.list
              if task["stage"] == "generate" and task["scenario"] == "di_photon"]
    whole = "diphoton_en20_sep90_x0_z70_r10_seed0"
    assert [task["name"] for task in inputs] == [whole + "_seg%dof3" % i for i in (1, 2, 3)]
    for index, task in enumerate(inputs, 1):
        command = task["params"]["command"]
        assert command[-4:] == ["--segment", str(index), "--segments", "3"], command
        assert task["params"]["whole"] == os.path.join(data, "input_single_particle_gun",
                                                       whole + ".slcio")
    # Each segment simulated with the seeds of its whole input: di_photon's
    # each with every seed, tau_pairs' each with its file's (and by Geant4's
    # second sample with those plus 1000000)
    names = sorted((task["simulator"], task["scenario"], task["name"], task["seed"])
                   for task in graph.list
                   if task["stage"] == "geant4" and task["scenario"] != "single_photon")
    samples = (("geant4", 0), ("geant4_2", 1000000))
    assert names == sorted(
        [(simulator, "di_photon", "%s_seg%dof3_simseed%d" % (whole, i, s + plus), s + plus)
         for simulator, plus in samples for i in (1, 2, 3) for s in (1, 2)]
        + [(simulator, "tau_pairs", "taus_%s_seg%dof3_simseed%d" % (n, i, s + plus), s + plus)
           for simulator, plus in samples for n, s in (("a", 42), ("b", 1764))
           for i in (1, 2, 3)])
    for task in graph.list:
        if task["stage"] == "observables":
            plus = 1000000 if task["simulator"] == "geant4_2" else 0
            assert task["params"]["seeds"]["di_photon"] == [1 + plus, 2 + plus] * 3
            assert task["params"]["seeds"]["tau_pairs"] == [42 + plus] * 3 + [1764 + plus] * 3
    # Forcing the generate stage remakes the gun's segments, not those made elsewhere
    graph = Graph(make_plan(force=["generate"])).find_status()
    assert collections.Counter((task["scenario"], task["status"]) for task in graph.list
                               if task["stage"] == "generate") == {
        ("single_photon", TODO): 2, ("di_photon", TODO): 3, ("tau_pairs", DONE): 6}


def test_whole_files_left_unsplit_stop_the_plan(segmented):
    """What was made from whole inputs before they were in segments would be
    made again for the segments: the plan says to split it first."""
    data = segmented.data
    segments = [os.path.join(data, "input_physics_samples", "taus_%s_seg%dof3.slcio" % (name, i))
                for name in "ab" for i in (1, 2, 3)]
    touch(*segments)
    make_plan()
    # A whole input whose segments are all there costs nothing
    whole = os.path.join(data, "input_physics_samples", "taus_a.slcio")
    touch(whole)
    make_plan()
    # but without them, it would be made anew (of any sample, -n too)
    os.remove(segments[1])
    with pytest.raises(PlanError, match=re.escape(
            "the inputs of tau_pairs are in segments now (input.segments), but 1 file(s) made "
            "from whole inputs before aren't split into them yet, e.g. "
            "input_physics_samples/taus_a.slcio: split them with scripts/split_data.py first")):
        make_plan(events=10)
    os.remove(whole)
    touch(segments[1])
    # a simulation of a whole input, or a reconstruction, as well
    name = "diphoton_en20_sep90_x0_z70_r10_seed0"
    simulation = os.path.join(data, "sim_output", "di_photon", "geant4",
                              name + "_simseed2.edm4hep.root")
    touch(simulation)
    with pytest.raises(PlanError, match="di_photon are in segments now .* e.g. sim_output/"
                                        "di_photon/geant4/%s_simseed2.edm4hep.root" % name):
        make_plan()
    touch(*[os.path.join(data, "sim_output", "di_photon", "geant4",
                         name + "_seg%dof3_simseed2.edm4hep.root" % i) for i in (1, 2, 3)])
    make_plan()
    reconstruction = os.path.join(data, "rec_output", "di_photon", "caloclouds3_paper",
                                  name + "_simseed1_REC.edm4hep.root")
    touch(reconstruction)
    with pytest.raises(PlanError, match="1 file\\(s\\) made from whole inputs"):
        make_plan()
    # and Geant4's second sample's, with its own seeds
    os.remove(reconstruction)
    second = os.path.join(data, "sim_output", "di_photon", "geant4_2",
                          name + "_simseed1000002.edm4hep.root")
    touch(second)
    with pytest.raises(PlanError, match="e.g. sim_output/di_photon/geant4_2/"
                                        "%s_simseed1000002.edm4hep.root" % name):
        make_plan()


@pytest.mark.parametrize("settings, message", [
    ("  di_photon:\n    input:\n      segments: 0\n", "di_photon.input.segments should be a "
                                                     "whole number, at least 1"),
    ("  di_photon:\n    input:\n      segments: 31\n", "can't put 30 events in 31 segments"),
    ("  tau_pairs:\n    simulation_seeds: [42]\n", "tau_pairs has 2 inputs but 1 "
                                                   "simulation_seeds"),
])
def test_segments_are_checked(segmented, monkeypatch, settings, message):
    override = os.path.join(segmented.tmp, "more.yaml")
    with open(override, "w") as stream:
        stream.write("scenarios:\n" + settings)
    monkeypatch.setenv("FASINATE_CONFIG", segmented.env["FASINATE_CONFIG"] + ":" + override)
    with pytest.raises(ConfigError, match=re.escape(message)):
        make_plan()
