"""The low-cost mode (ddml_bench/low_cost.py, low_cost.sh): a model tested on
the first events of each input against the full samples of the others, as
published. Its plans, in this process; then end to end on the fakes, against
a record published from a run of the pipeline (as tests/test_fetch.py
publishes one)."""

import collections
import csv
import os
import shutil

import pytest

from conftest import Bench
from ddml_bench import config, low_cost
from ddml_bench.config import ConfigError
from ddml_bench.graph import TODO, UNSELECTED, Graph, PlanError, make_plan
from test_fetch import fetching, publish
from test_graph import measure_all, outputs, touch

MODEL = "caloclouds3_paper"


@pytest.fixture
def configured(config_env, monkeypatch):
    # Not the configs/local.yaml of whoever runs the tests
    monkeypatch.setattr(config, "LOCAL_FILE", os.path.join(config_env.tmp, "no-local.yaml"))
    return config_env


@pytest.fixture
def full(configured, monkeypatch):
    """The data folder with the inputs and every simulator's full
    measurements, as fetched."""
    graph = Graph(make_plan())
    touch(*outputs(graph, "generate"))
    measure_all(graph, monkeypatch)
    return configured


def read_csv(path):
    with open(path) as stream:
        return list(csv.DictReader(stream))


def test_a_low_cost_plan_runs_only_the_models_sample_and_its_plots(full):
    plan = make_plan(simulators=[MODEL], events={"single_photon": 3, "di_photon": 2},
                     low_cost=True)
    graph = Graph(plan).find_status()
    todo = collections.Counter(task["stage"] for task in graph.list if task["status"] == TODO)
    # The model's 2 single-photon inputs and its di-photon input with 2
    # seeds, reconstructed, measured, and plotted against the full samples
    assert todo == {"models": 4, "reconstruct": 4, "observables": 1, "plot": 1}
    assert all(task["status"] == UNSELECTED for task in graph.list
               if task["stage"] == "geant4" or task["simulator"] in ("geant4", "geant4_2"))
    simulations = dict((task["name"], task) for task in graph.list
                       if task["stage"] == "models" and task["status"] == TODO)
    # Each scenario's simulations of its own number of events
    assert sorted(simulations) == [
        "diphoton_en20_sep90_x0_z70_r10_seed0_n2_simseed1",
        "diphoton_en20_sep90_x0_z70_r10_seed0_n2_simseed2",
        "p22_th43-137_ph79-109_en20_seed0_ip_surface_n3",
        "p22_th43-137_ph79-109_en80_seed1_ip_surface_n3"]
    assert sorted(task["params"]["events"] for task in simulations.values()) == [2, 2, 3, 3]
    measured, = [task for task in graph.list
                 if task["stage"] == "observables" and task["status"] == TODO]
    assert measured["output"] == full.path("observables", MODEL + "_n3_di_photon2.npz")
    plot = graph.list[-1]
    assert plot["output"] == full.path("observables", "plots", "low_cost_n3_di_photon2",
                                       "metrics.csv")
    assert plot["params"]["samples"] == [MODEL] and plot["after"] == [measured["key"]]
    assert plot["params"]["suffix"] == "_n3_di_photon2"


def test_the_numbers_of_events_of_the_scenarios_are_checked(full):
    with pytest.raises(PlanError, match="no number of events for the scenario di_photon"):
        make_plan(simulators=[MODEL], events={"single_photon": 3}, low_cost=True)
    with pytest.raises(PlanError, match="the scenario elsewhere, which the configuration"):
        make_plan(simulators=[MODEL], events={"single_photon": 3, "di_photon": 3,
                                              "elsewhere": 3}, low_cost=True)
    # A disabled scenario's is left out
    plan = make_plan(simulators=[MODEL], events={"single_photon": 3, "di_photon": 3,
                                                 "tau_pairs": 1}, low_cost=True)
    assert plan["events"] == {"single_photon": 3, "di_photon": 3}
    with pytest.raises(PlanError, match="needs a number of them"):
        make_plan(simulators=[MODEL], low_cost=True)


def test_the_configuration_has_a_number_of_events_for_each_scenario(full):
    settings = config.load_config()
    assert low_cost.configured_events(settings) == {"single_photon": 1000, "di_photon": 1000}
    assert low_cost.configured_events(settings, 7) == {"single_photon": 7, "di_photon": 7}
    assert low_cost.configured_suffix(settings) == "_n1000"
    settings["low_cost"]["events"]["di_photon"] = 500
    assert low_cost.configured_suffix(settings) == "_n1000_di_photon500"
    assert low_cost.events_text({"single_photon": 1000, "di_photon": 500}) == \
        "the first 1000 events of each input, and 500 of di_photon's"
    del settings["low_cost"]["events"]["single_photon"]
    with pytest.raises(ConfigError, match="no number of events for the scenario single_photon"):
        low_cost.configured_events(settings)


def test_the_estimate_counts_each_scenarios_simulations_and_events(full):
    plan = make_plan(simulators=[MODEL], events={"single_photon": 3, "di_photon": 2},
                     low_cost=True)
    settings = config.load_config()
    rows = low_cost.estimate(plan, settings)
    assert [row[:4] for row in rows] == [("single_photon", 2, 3, 6), ("di_photon", 2, 2, 4)]
    scenario, simulations, each, events, simulation, source, reconstruction, hours = rows[0]
    assert (simulation, source, reconstruction) == (0.104, "configured", 0.32)
    assert hours == pytest.approx(2 * (3 * (0.104 + 0.32) + 180) / 3600)
    lines = low_cost.estimate_lines(rows, settings, [MODEL])
    assert lines[-1].split()[0] == "all" and lines[-1].split()[-1] == "0.2"


def test_what_to_pin_is_written_as_yaml():
    assert low_cost.pins_yaml([("ddml.commit", "abc", None),
                               ("detector_configs.ildconfig.commit", "def", "xyz")]) == (
        'ddml:\n  commit: "abc"\ndetector_configs:\n  ildconfig:\n    commit: "def"\n')


# ---------------------------------------------------------------------------
# End to end, on the fakes


@pytest.fixture(scope="module")
def published(tmp_path_factory):
    """A record of the essential files of a run of the pipeline."""
    publisher = Bench(str(tmp_path_factory.mktemp("publisher")))
    publisher.run("pipeline.sh")
    return publish(publisher, "321")


def another_model(bench, name="another"):
    """A model of its own, CaloClouds3 (paper) drawn otherwise."""
    with open(os.path.join(bench.repo, "models", MODEL + ".yaml")) as stream:
        text = stream.read().replace("name: CaloClouds3 (paper)", "name: Another")
    with open(os.path.join(bench.repo, "models", name + ".yaml"), "w") as stream:
        stream.write(text.replace('linestyle: "-."', 'linestyle: "--"'))


def pinned(bench, published):
    """Pin the software the published measurements were made with, and stamp
    the fake DDML build as built for that, as scripts/build_ddml.sh would."""
    import json
    from conftest import ENV_FILE, FAKES
    with open(os.path.join(published["folder"], "manifest.json")) as stream:
        stack = json.load(stream)["stack"]
    override = os.path.join(bench.tmp, "pinned.yaml")
    with open(override, "w") as stream:
        stream.write('ddml:\n  commit: "%s"\ndetector_configs:\n  ildconfig:\n    commit: "%s"\n'
                     % (stack["ddml"], stack["ildconfig"]))
    bench.env["FASINATE_CONFIG"] += ":" + override
    # (each bench's ILDConfig is a repository of its own: the publisher's has
    # its commit)
    clone = os.path.join(bench.tmp, "ILDConfig.git")
    shutil.rmtree(clone)
    shutil.copytree(os.path.join(os.path.dirname(published["data"]), "ILDConfig.git"), clone)
    with open(os.path.join(bench.software, "env", "ddml.sh"), "w") as stream:
        stream.write(ENV_FILE.format(stamp=bench.setting("FASINATE_DDML_BUILD_STAMP"),
                                     ddml=bench.ddml, tmp=bench.tmp, bin=bench.bin,
                                     python=os.path.join(FAKES, "python")))


def test_a_new_model_is_tested_on_a_part_of_the_events_against_the_published_ones(
        bench, published):
    fetching(bench, published)
    another_model(bench)
    # The software isn't the published measurements', so it stops, saying what
    # to pin
    status, output = bench.run("low_cost.sh", "another", "-n", "3", "--dry-run", check=False)
    assert status == 1 and "wouldn't be compared like with like" in output, output
    assert 'ddml:\n  commit: "0123456789abcdef0123456789abcdef01234567"' in output, output
    assert not os.path.exists(bench.path("observables"))

    pinned(bench, published)
    status, output = bench.run("low_cost.sh", "another", "-n", "3", "--dry-run")
    assert "would fetch 6 published file(s)" in output, output
    assert "low-cost run of another: the first 3 events of each input, against the full " \
        "samples of geant4 (to fetch), geant4_2 (to fetch), caloclouds3_paper (to fetch)" \
        in output, output
    assert "  all                        4                      12                                        0.2" in output, output
    assert "would run 13 task(s)" in output, output
    assert not os.path.exists(bench.path("observables"))

    status, output = bench.run("low_cost.sh", "another", "-n", "3")
    assert "fetching 6 published file(s)" in output and "made 10 of 10" in output, output
    # Only the model's sample was made, and nothing of the others
    assert bench.files("sim_output", "single_photon") == ["another"]
    assert all("_n3" in name for name in bench.files("sim_output", "single_photon", "another"))
    assert sorted(bench.files("observables")) == [
        "another_n3.npz", "caloclouds3_paper.npz", "geant4.npz", "geant4_2.npz", "plots"]
    # Its plots, against the full samples, with the floors and projections
    assert bench.files("observables", "plots") == ["low_cost_n3"]
    plots = bench.path("observables", "plots", "low_cost_n3")
    assert {"metrics.csv", "fits.csv", "dashboard.html",
            "radial_profile_r100__single_photon.png"} <= set(os.listdir(plots))
    rows = read_csv(os.path.join(plots, "metrics.csv"))
    assert sorted({row["simulator"] for row in rows}) == ["another", "caloclouds3_paper",
                                                         "geant4_2"]
    events = dict((row["simulator"], int(row["events"])) for row in rows
                  if row["profile"] == "r100" and row["metric"] == "JS")
    # 2 energies of 6 events each, 3 of each for the sample
    assert events == {"another": 6, "caloclouds3_paper": 12, "geant4_2": 12}
    for row in rows:
        if row["simulator"] != "another":
            assert row["projected"] == ""
            continue
        if row["metric"] in ("JS", "W1", "L1"):
            assert float(row["floor"]) > 0 and row["floor_uncertainty"], row
        if row["metric"] in ("JS", "W1", "L1", "chi2"):
            assert row["projected"], row
    # The second sample's profiles have their floors at the full size, the
    # check of how they are drawn
    assert all(row["floor"] for row in rows if row["simulator"] == "geant4_2"
               and row["observable"].endswith("_profile"))
    with open(os.path.join(plots, "dashboard.html")) as stream:
        page = stream.read()
    assert "Another (n=3)" in page and "its sample of the first events of each input" in page
    # Done, so a run again has nothing to do
    status, output = bench.run("low_cost.sh", "another", "-n", "3")
    assert "everything selected is done" in output and "fetching" not in output, output

    # The scenarios can have numbers of their own
    override = os.path.join(bench.tmp, "events.yaml")
    with open(override, "w") as stream:
        stream.write("low_cost:\n  events:\n    single_photon: 3\n    di_photon: 2\n")
    bench.env["FASINATE_CONFIG"] += ":" + override
    status, output = bench.run("low_cost.sh", "another")
    # Its single photons are those of -n 3 already, so only the di-photons
    # are simulated, then all measured
    assert "made 6 of 6" in output, output
    assert "another_n3_di_photon2.npz" in bench.files("observables")
    assert sorted(bench.files("observables", "plots")) == ["low_cost_n3", "low_cost_n3_di_photon2"]
    rows = read_csv(bench.path("observables", "plots", "low_cost_n3_di_photon2", "metrics.csv"))
    assert sorted({int(row["events"]) for row in rows if row["simulator"] == "another"}) == [4, 6]


def test_the_plots_of_a_sample_against_the_full_ones(bench, published):
    """plot_observables.sh --sample, on what a low-cost run measured."""
    fetching(bench, published)
    pinned(bench, published)
    another_model(bench)
    status, output = bench.run("low_cost.sh", "another", "-n", "2")
    assert "made 10 of 10" in output, output
    shutil.rmtree(bench.path("observables", "plots"))
    status, output = bench.run("plot_observables.sh", "--sample", "another:2", "--format", "png")
    assert "floor" in output, output
    assert bench.files("observables", "plots") == ["low_cost_n2"]
    # A sample that isn't measured is said, and nothing drawn
    status, output = bench.run("plot_observables.sh", "--sample", "another:5", check=False)
    assert status == 1 and "another_n5.npz" in output, output
    status, output = bench.run("plot_observables.sh", "--sample", "another:x", check=False)
    assert status == 2 and "give the number of events of each input" in output, output


def test_without_published_measurements_it_compares_with_what_is_here(bench):
    """--no-fetch (or nothing published): the full measurements here."""
    another_model(bench)
    status, output = bench.run("low_cost.sh", "another", "-n", "2", check=False)
    assert status == 1 and "no full measurements of Geant4" in output, output
    bench.run("pipeline.sh")
    status, output = bench.run("low_cost.sh", "another", "-n", "2", "--no-fetch")
    assert "against the full samples of geant4 (here), geant4_2 (here), caloclouds3_paper " \
        "(here)" in output and "made 10 of 10" in output, output
    # The model's full measurements are there too, but its sample is drawn
    rows = read_csv(bench.path("observables", "plots", "low_cost_n2", "metrics.csv"))
    assert {int(row["events"]) for row in rows if row["simulator"] == "another"} == {4}


def test_a_low_cost_run_as_slurm_jobs(bench, published):
    fetching(bench, published)
    pinned(bench, published)
    another_model(bench)
    status, output = bench.run("low_cost.sh", "another", "-n", "3", "--executor", "slurm")
    assert "submitting 10 job(s)" in output, output
    jobs = bench.slurm_jobs()
    assert all(job["state"] == "COMPLETED" for job in jobs), jobs
    assert "metrics.csv" in bench.files("observables", "plots", "low_cost_n3")


def test_only_models_are_tested(bench):
    status, output = bench.run("low_cost.sh", "geant4", check=False)
    assert status == 1 and "so geant4 can't be one of them" in output, output
    status, output = bench.run("low_cost.sh", "nonesuch", check=False)
    assert status == 1 and "there is no model 'nonesuch'" in output, output
    status, output = bench.run("low_cost.sh", check=False)
    assert status == 1 and "give the models to test" in output, output


def test_a_samples_measurements_are_never_published_as_full_ones(configured):
    from ddml_bench import manifest, models
    for name in ("another_n3.npz", "another_n3_di_photon2.npz"):
        record, role, _, _ = manifest.classify("observables/" + name)
        assert (record, role) != (manifest.ESSENTIAL, manifest.REQUIRED), name
    assert manifest.classify("observables/another.npz")[:2] == (manifest.ESSENTIAL,
                                                                 manifest.REQUIRED)
    # nor can a model be named as a sample's measurements are
    source = os.path.join(configured.repo, "models", MODEL + ".yaml")
    shutil.copy(source, os.path.join(configured.repo, "models", "mine_n5_tau_pairs2.yaml"))
    with pytest.raises(models.ModelError, match="can't be a model name"):
        models.load_model("mine_n5_tau_pairs2")
