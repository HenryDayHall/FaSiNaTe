"""The pipeline end to end with the local executor: pipeline.sh and the stage
scripts, through scripts/run_in_environment.sh and setup.sh, with the fakes of
tests/fakes at the ends."""

import csv
import json
import os

import numpy as np

INPUTS = ["p22_th43-137_ph79-109_en20_seed0_ip_surface",
          "p22_th43-137_ph79-109_en80_seed1_ip_surface"]
SIMULATORS = ["caloclouds3", "geant4"]


def read_metrics(path):
    with open(path) as stream:
        return list(csv.DictReader(stream))


def test_pipeline_runs_every_stage(bench):
    status, output = bench.run("pipeline.sh", "--dry-run")
    assert "would run 13 task(s)" in output, output

    status, output = bench.run("pipeline.sh")
    assert "made 13 of 13" in output, output
    assert bench.files("input_single_particle_gun") == sorted(
        name + ending for name in INPUTS for ending in (".h5", ".slcio"))
    assert bench.files("sim_output", "single_photon", "geant4") == sorted(
        name + ending for name in INPUTS for ending in (".edm4hep.root", ".log"))
    assert bench.files("sim_output", "single_photon", "caloclouds3") == sorted(
        name + ending for name in INPUTS for ending in (".root", ".log"))
    for simulator in SIMULATORS:
        assert bench.files("rec_output", "single_photon", simulator) == sorted(
            name + ending for name in INPUTS
            for ending in ("_AIDA.root", "_PfoAnalysis.root", "_REC.edm4hep.root", ".log"))
    assert bench.files("observables") == ["caloclouds3.npz", "geant4.npz", "plots"]
    rows = read_metrics(bench.path("observables", "plots", "metrics.csv"))
    assert sorted({row["simulator"] for row in rows}) == ["caloclouds3"]
    assert {(row["observable"], row["profile"]) for row in rows} == {
        ("radial_profile", "r100"), ("radial_profile", "r30"),
        ("longitudinal_profile", "layers")}
    assert all(float(row["value"]) > 0 for row in rows)
    assert os.path.isfile(bench.path("observables", "plots",
                                     "radial_profile_r30__single_photon.png"))

    # The measurements record the reconstructed files relative to the data folder
    with np.load(bench.path("observables", "geant4.npz")) as data:
        assert [str(f) for f in data["single_photon__files"]] == [
            "rec_output/single_photon/geant4/%s_REC.edm4hep.root" % name for name in INPUTS]
        assert list(data["single_photon__n_events"]) == [6, 6]

    # The simulations ran as the stages run ddsim
    with open(bench.path("sim_output", "single_photon", "caloclouds3",
                         INPUTS[0] + ".root")) as stream:
        simulation = json.load(stream)
    assert simulation["simulator"] == "CC3_BARREL_PY_INTERFACE"
    assert len(simulation["events"]) == 6
    log = open(bench.path("sim_output", "single_photon", "geant4", INPUTS[0] + ".log")).read()
    assert "ddsim_steer_geant4.py" in log and "# finished" in log

    # Everything is done, so a second run has nothing to do
    status, output = bench.run("pipeline.sh")
    assert "everything selected is done" in output, output

    # Measurements made before the records were relative, with full paths, still count
    path = bench.path("observables", "geant4.npz")
    with np.load(path) as data:
        arrays = dict((key, data[key]) for key in data.files)
    arrays["single_photon__files"] = np.array(
        [bench.path(str(name)) for name in arrays["single_photon__files"]])
    np.savez_compressed(path, **arrays)
    status, output = bench.run("observables_from_reconstruction.sh", "--dry-run")
    assert "everything selected is done" in output, output

    # Remaking the reconstructions remakes what is made from them, and only that
    status, output = bench.run("pipeline.sh", "--dry-run", "-f", "reconstruct")
    assert "would run 7 task(s)" in output, output
    assert "(--force reconstruct)" in output
    assert "(the reconstruction of the geant4 simulation of %s is made again)" % INPUTS[0] \
        in output


def test_stage_scripts_keep_one_sample_through_every_stage(bench):
    """-n N names the same sample at every stage (it used to give _nN_nN_REC)."""
    bench.run("generate_input.sh")
    bench.run("simulate_geant4.sh", "-n", "3")
    bench.run("simulate_models.sh", "-n", "3")
    bench.run("reconstruct_simulation.sh", "-n", "3")
    bench.run("observables_from_reconstruction.sh", "-n", "3")
    bench.run("plot_observables.sh", "-n", "3")

    names = [name + "_n3" for name in INPUTS]
    assert bench.files("sim_output", "single_photon", "geant4") == sorted(
        name + ending for name in names for ending in (".edm4hep.root", ".log"))
    for simulator in SIMULATORS:
        assert bench.files("rec_output", "single_photon", simulator) == sorted(
            name + ending for name in names
            for ending in ("_AIDA.root", "_PfoAnalysis.root", "_REC.edm4hep.root", ".log"))
    assert bench.files("observables") == ["caloclouds3_n3.npz", "geant4_n3.npz", "plots"]
    assert "metrics_n3.csv" in bench.files("observables", "plots")
    with np.load(bench.path("observables", "caloclouds3_n3.npz")) as data:
        assert list(data["single_photon__n_events"]) == [3, 3]
    # The reconstruction ran on the whole test simulation, of 3 events
    log = open(bench.path("rec_output", "single_photon", "geant4", names[0] + ".log")).read()
    assert "--num-events=-1" in log and "--usingParticleGun" in log

    # The full sample is still to do, and the test sample is done
    status, output = bench.run("pipeline.sh", "--dry-run", "-n", "3")
    assert "everything selected is done" in output, output
    status, output = bench.run("pipeline.sh", "--dry-run")
    assert "would run 11 task(s)" in output, output


def test_a_failure_stops_only_what_needs_it(bench):
    bench.fail_ddsim("geant4 *en80*")
    status, output = bench.run("pipeline.sh", check=False)
    assert status == 1, output
    assert "made 9 of 13" in output, output
    log = bench.path("sim_output", "single_photon", "geant4", INPUTS[1] + ".log")
    assert "# FAILED" in open(log).read()
    # Nothing half made is left, and only the Geant4 simulation of en80 is missing
    assert not any(name.startswith(".") for name in os.listdir(
        bench.path("sim_output", "single_photon", "geant4")))
    assert INPUTS[1] + ".edm4hep.root" not in bench.files("sim_output", "single_photon", "geant4")
    assert bench.files("observables") == ["caloclouds3.npz"]

    bench.fail_ddsim(None)
    status, output = bench.run("pipeline.sh")
    assert "made 4 of 4" in output, output
    assert os.path.isfile(bench.path("observables", "plots", "metrics.csv"))


def test_stages_skip_what_exists_unless_forced(bench):
    bench.run("generate_input.sh")
    path = bench.path("input_single_particle_gun", INPUTS[0] + ".slcio")
    os.utime(path, (0, 0))
    status, output = bench.run("generate_input.sh")
    assert "everything selected is done" in output, output
    assert os.path.getmtime(path) == 0
    status, output = bench.run("generate_input.sh", "-f", "single_photon")
    assert "made 2 of 2" in output, output
    assert os.path.getmtime(path) > 0


def test_simulating_inputs_not_made_is_an_error(bench):
    status, output = bench.run("simulate_geant4.sh", check=False)
    assert status == 1, output
    assert "it needs the input %s, which isn't selected to run" % INPUTS[0] in output


def test_files_given_directly(bench):
    bench.run("generate_input.sh")
    elsewhere = os.path.join(bench.tmp, "elsewhere")
    os.makedirs(elsewhere)
    given = os.path.join(elsewhere, "mine.slcio")
    with open(bench.path("input_single_particle_gun", INPUTS[0] + ".slcio")) as source:
        open(given, "w").write(source.read())

    # An input of no scenario is simulated as the scenario custom, and nothing else
    bench.run("simulate_geant4.sh", "-i", given)
    assert bench.files("sim_output") == ["custom"]
    assert bench.files("sim_output", "custom", "geant4") == ["mine.edm4hep.root", "mine.log"]
    bench.run("reconstruct_simulation.sh", "-s", "custom")
    assert "mine_REC.edm4hep.root" in bench.files("rec_output", "custom", "geant4")
    log = open(bench.path("rec_output", "custom", "geant4", "mine.log")).read()
    assert "--usingParticleGun" not in log

    # A simulation from outside sim_output is reconstructed to custom/files
    other = os.path.join(elsewhere, "other.edm4hep.root")
    with open(bench.path("sim_output", "custom", "geant4", "mine.edm4hep.root")) as source:
        open(other, "w").write(source.read())
    bench.run("reconstruct_simulation.sh", other)
    assert "other_REC.edm4hep.root" in bench.files("rec_output", "custom", "files")

    status, output = bench.run("reconstruct_simulation.sh", "-n", "2", other, check=False)
    assert status == 2 and "give simulation files without it" in output, output


def test_help_needs_no_configuration(bench):
    for script in ("pipeline.sh", "slurm_pipeline.sh", "generate_input.sh", "simulate_geant4.sh",
                   "simulate_models.sh", "reconstruct_simulation.sh",
                   "observables_from_reconstruction.sh", "plot_observables.sh",
                   "slurm_simulate_geant4.sh"):
        status, output = bench.run(script, "--help", env={"FASINATE_CONFIG": "/nonexistent"})
        assert output.startswith(("Run", "Make", "Simulate", "Reconstruct", "Measure", "Plot")), \
            output
