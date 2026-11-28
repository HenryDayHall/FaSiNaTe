"""The pipeline end to end with the local executor: pipeline.sh and the stage
scripts, through scripts/run_in_environment.sh and setup.sh, with the fakes of
tests/fakes at the ends."""

import csv
import json
import os
import shutil

import h5py
import numpy as np

from conftest import TAU_INPUTS, TAU_SEEDS, write_tau_input

INPUTS = ["p22_th43-137_ph79-109_en20_seed0_ip_surface",
          "p22_th43-137_ph79-109_en80_seed1_ip_surface"]
SIMULATORS = ["caloclouds3_paper", "geant4"]
# Geant4's second sample, simulated with seeds of its own: the first's plus
# 1000000, or 1000000 where the first has ddsim's own
SECOND = "geant4_2"
SECOND_INPUTS = [name + "_simseed1000000" for name in INPUTS]
# The di_photon scenario's input, simulated with the seeds 1 and 2
DI_PHOTON_INPUT = "diphoton_en20_sep90_x0_z70_r10_seed0"
DI_PHOTON_SIMULATIONS = [DI_PHOTON_INPUT + "_simseed1", DI_PHOTON_INPUT + "_simseed2"]
SECOND_DI_PHOTON = [DI_PHOTON_INPUT + "_simseed1000001", DI_PHOTON_INPUT + "_simseed1000002"]
# The tau_pairs scenario's inputs (Bench.with_tau_pairs()), each simulated with
# its own seed
TAU_SIMULATIONS = ["%s_simseed%d" % (name[:-len(".slcio")], seed)
                   for name, seed in zip(TAU_INPUTS, TAU_SEEDS)]


def stage_line(stage, *counts):
    """A line of the table of what each stage has done and has to do."""
    return "  %-12s" % stage + "".join(" %6d" % count for count in counts)


def read_metrics(path):
    with open(path) as stream:
        return list(csv.DictReader(stream))


def test_pipeline_runs_every_stage(bench):
    status, output = bench.run("pipeline.sh", "--dry-run")
    assert "would run 32 task(s)" in output, output

    status, output = bench.run("pipeline.sh")
    assert "made 32 of 32" in output, output
    assert bench.files("input_single_particle_gun") == sorted(
        name + ending for name in INPUTS + [DI_PHOTON_INPUT] for ending in (".h5", ".slcio"))
    assert bench.files("sim_output", "single_photon", "geant4") == sorted(
        name + ending for name in INPUTS for ending in (".edm4hep.root", ".log"))
    assert bench.files("sim_output", "single_photon", "caloclouds3_paper") == sorted(
        name + ending for name in INPUTS for ending in (".root", ".log"))
    for simulator in SIMULATORS:
        assert bench.files("rec_output", "single_photon", simulator) == sorted(
            name + ending for name in INPUTS
            for ending in ("_AIDA.root", "_PfoAnalysis.root", "_REC.edm4hep.root", ".log"))
    # di_photon's input once with each of its seeds
    assert bench.files("sim_output", "di_photon", "geant4") == sorted(
        name + ending for name in DI_PHOTON_SIMULATIONS for ending in (".edm4hep.root", ".log"))
    assert bench.files("sim_output", "di_photon", "caloclouds3_paper") == sorted(
        name + ending for name in DI_PHOTON_SIMULATIONS for ending in (".root", ".log"))
    # and Geant4's second sample, with seeds of its own, reconstructed too
    for scenario, names in (("single_photon", SECOND_INPUTS), ("di_photon", SECOND_DI_PHOTON)):
        assert bench.files("sim_output", scenario, SECOND) == sorted(
            name + ending for name in names for ending in (".edm4hep.root", ".log"))
        assert bench.files("rec_output", scenario, SECOND) == sorted(
            name + ending for name in names
            for ending in ("_AIDA.root", "_PfoAnalysis.root", "_REC.edm4hep.root", ".log"))
    assert bench.files("observables") == ["caloclouds3_paper.npz", "geant4.npz", "geant4_2.npz",
                                          "plots"]
    rows = read_metrics(bench.path("observables", "plots", "metrics.csv"))
    # The second sample is compared with Geant4 as the model is
    assert sorted({row["simulator"] for row in rows}) == ["caloclouds3_paper", SECOND]
    assert {(row["observable"], row["profile"]) for row in rows if row["simulator"] == SECOND} \
        == {(row["observable"], row["profile"]) for row in rows
            if row["simulator"] == "caloclouds3_paper"}
    assert {(row["observable"], row["profile"]) for row in rows} == {
        ("radial_profile", "r100"), ("radial_profile", "r30"),
        ("longitudinal_profile", "layers"), ("energy", "linearity"), ("energy", "resolution"),
        ("intrinsic_angle", "theta_all"), ("intrinsic_angle", "phi_all"),
        ("intrinsic_angle", "theta_top4"), ("intrinsic_angle", "phi_top4"),
        ("photon_separation", "20GeV")}
    assert all(float(row["value"]) > 0 for row in rows)
    # Each distribution's Wasserstein distance as well as its metric, and each
    # energy figure's chi^2
    assert sorted((row["observable"], row["profile"]) for row in rows
                  if row["metric"] == "W1") == sorted(
        (row["observable"], row["profile"]) for row in rows if row["metric"] == "JS")
    assert sorted(row["profile"] for row in rows if row["metric"] == "chi2") == [
        "linearity", "linearity", "resolution", "resolution"]
    # Only the separation's metrics have an uncertainty, from its two seeds
    for row in rows:
        if row["observable"] == "photon_separation":
            assert row["metric"] in ("JS", "W1") and float(row["uncertainty"]) > 0, row
        else:
            assert row["uncertainty"] == "", row
    for figure in ("radial_profile_r30__single_photon", "energy_linearity__single_photon",
                   "energy_resolution__single_photon", "intrinsic_angle_theta_all__single_photon",
                   "intrinsic_angle_phi_top4__single_photon", "photon_separation_20GeV__di_photon"):
        assert os.path.isfile(bench.path("observables", "plots", figure + ".png")), figure
    # and every figure in the dashboard, with a menu entry for the model
    with open(bench.path("observables", "plots", "dashboard.html"), encoding="utf-8") as stream:
        page = stream.read()
    assert page.count('<figure class="fsn-card"') == len(
        [name for name in bench.files("observables", "plots") if name.endswith(".png")])
    assert '<label class="fsn-model" title="caloclouds3_paper"' in page
    assert '<label class="fsn-model" title="geant4_2"' in page
    # The two energies fix the straight line of the linearity, but not the
    # three parameters of the resolution's fit
    fits = read_metrics(bench.path("observables", "plots", "fits.csv"))
    assert sorted((row["profile"], row["simulator"], row["parameter"]) for row in fits) == sorted(
        ("linearity", simulator, parameter) for simulator in SIMULATORS + [SECOND]
        for parameter in ("slope", "intercept"))
    assert all(row["ndf"] == "0" for row in fits)

    # The measurements record the reconstructed files relative to the data folder
    with np.load(bench.path("observables", "geant4.npz")) as data:
        assert [str(f) for f in data["single_photon__files"]] == [
            "rec_output/single_photon/geant4/%s_REC.edm4hep.root" % name for name in INPUTS]
        assert list(data["single_photon__n_events"]) == [6, 6]
        # and the reconstructed energy of each event, the sum of its
        # PandoraPFOs', with its true energy
        np.testing.assert_allclose(data["energy__single_photon__true"], [20] * 6 + [80] * 6,
                                   rtol=1e-5)
        ratio = data["energy__single_photon__reconstructed"] / data["energy__single_photon__true"]
        assert ((ratio > 0.8) & (ratio < 1.2)).all(), ratio
        # and the residuals of the intrinsic angles: the axes of the fake
        # showers, from all their hits and from their most energetic 4%, are
        # near the photons' lines of flight
        for angle in ("theta", "phi"):
            for hits in ("all", "top4"):
                residuals = data["intrinsic_angle__single_photon__d%s_%s" % (angle, hits)]
                assert residuals.shape == (12,) and (np.abs(residuals) < 10).all(), residuals
        # against the photons' own theta and phi
        momenta = np.array([event["particles"][0]["momentum"] for name in INPUTS
                            for event in json.load(open(bench.path(
                                "sim_output", "single_photon", "geant4",
                                name + ".edm4hep.root")))["events"]])
        np.testing.assert_allclose(
            data["intrinsic_angle__single_photon__theta_true"],
            np.degrees(np.arccos(momenta[:, 2] / np.linalg.norm(momenta, axis=1))), atol=1e-4)
        np.testing.assert_allclose(data["intrinsic_angle__single_photon__phi_true"],
                                   np.degrees(np.arctan2(momenta[:, 1], momenta[:, 0])), atol=1e-4)

    # The separation of di_photon's pairs, from the MC truth of the
    # reconstructed events, against the number of reconstructed photons
    with h5py.File(bench.path("input_single_particle_gun", DI_PHOTON_INPUT + ".h5"), "r") as h5:
        generated = h5["separation"][:]
    assert generated.shape == (30,)
    for simulator in SIMULATORS:
        with np.load(bench.path("observables", simulator + ".npz")) as data:
            assert [str(f) for f in data["di_photon__files"]] == [
                "rec_output/di_photon/%s/%s_REC.edm4hep.root" % (simulator, name)
                for name in DI_PHOTON_SIMULATIONS]
            assert list(data["di_photon__seeds"]) == [1, 2]
            assert list(data["single_photon__seeds"]) == [-1, -1]
            assert list(data["di_photon__n_events"]) == [30, 30]
            # The photons start on the ECAL face, so they are separated on it
            # as the gun placed them, the same events for each seed
            np.testing.assert_allclose(data["photon_separation__di_photon__separation"],
                                       np.tile(generated, 2), rtol=1e-9)
            np.testing.assert_allclose(data["photon_separation__di_photon__energy"], 20,
                                       rtol=1e-5)
            photons = data["photon_separation__di_photon__n_photons"]
            separation = data["photon_separation__di_photon__separation"]
            assert set(photons) <= {1, 2, 3}
            # (the seeded input has 3 pairs closer than 15 mm, and 19 further than 40)
            assert photons[separation > 40].mean() > photons[separation < 15].mean()

    with np.load(bench.path("observables", SECOND + ".npz")) as data:
        assert str(data["meta__label"]) == "Geant4 (2nd sample)"
        assert [str(f) for f in data["di_photon__files"]] == [
            "rec_output/di_photon/%s/%s_REC.edm4hep.root" % (SECOND, name)
            for name in SECOND_DI_PHOTON]
        assert list(data["di_photon__seeds"]) == [1000001, 1000002]
        assert list(data["single_photon__seeds"]) == [1000000, 1000000]
        assert list(data["single_photon__n_events"]) == [6, 6]

    # The simulations ran as the stages run ddsim, di_photon's with their seeds
    with open(bench.path("sim_output", "single_photon", "caloclouds3_paper",
                         INPUTS[0] + ".root")) as stream:
        simulation = json.load(stream)
    # with the presets of its description (the fake ddsim names them, joined by +)
    assert simulation["simulator"] == "+".join(
        bench.description("caloclouds3_paper")["ddml"]["presets"])
    assert len(simulation["events"]) == 6 and simulation["seed"] is None
    log = open(bench.path("sim_output", "single_photon", "geant4", INPUTS[0] + ".log")).read()
    assert "ddsim_steer_geant4.py" in log and "# finished" in log
    assert "--random.seed" not in log and "# seed      ddsim's own" in log
    for seed, name in zip((1, 2), DI_PHOTON_SIMULATIONS):
        with open(bench.path("sim_output", "di_photon", "caloclouds3_paper",
                             name + ".root")) as stream:
            simulation = json.load(stream)
        assert simulation["seed"] == seed and len(simulation["events"]) == 30
        assert all(len(event["particles"]) == 2 for event in simulation["events"])
        log = open(bench.path("sim_output", "di_photon", "geant4", name + ".log")).read()
        assert "--random.seed=%d" % seed in log and "# seed      %d\n" % seed in log
    # The second sample's as the first's, with its own seeds
    for seed, name in zip((1000000, 1000000, 1000001, 1000002), SECOND_INPUTS + SECOND_DI_PHOTON):
        scenario = "di_photon" if name.startswith("diphoton") else "single_photon"
        log = open(bench.path("sim_output", scenario, SECOND, name + ".log")).read()
        assert "ddsim_steer_geant4.py" in log and "# finished" in log
        assert "--random.seed=%d" % seed in log and "# seed      %d\n" % seed in log
        assert "# ddml-bench geant4: Geant4's second sample (geant4_2) on %s" % scenario in log
    # and their reconstruction is told they are a particle gun's
    log = open(bench.path("rec_output", "di_photon", "geant4",
                          DI_PHOTON_SIMULATIONS[0] + ".log")).read()
    assert "--usingParticleGun" in log

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
    assert "would run 16 task(s)" in output, output
    assert "(--force reconstruct)" in output
    assert "(the reconstruction of the geant4 simulation of %s is made again)" % INPUTS[0] \
        in output


def test_a_model_added_to_fetched_measurements_runs_only_its_own_tasks(bench):
    """With only the inputs and the .npz files of the simulators measured so far,
    as fetched from where they are published, into a data folder of its own:
    adding a model simulates, reconstructs and measures only that model, and
    plots it against the others."""
    bench.run("pipeline.sh")
    measured = dict((name, open(bench.path("observables", name), "rb").read())
                    for name in ("caloclouds3_paper.npz", "geant4.npz", "geant4_2.npz"))
    fetched = os.path.join(bench.tmp, "fetched")
    for folder in ("input_single_particle_gun", "observables"):
        os.makedirs(os.path.join(fetched, folder))
        for name in bench.files(folder):
            if name != "plots":
                shutil.copy(bench.path(folder, name), os.path.join(fetched, folder, name))
    override = os.path.join(bench.tmp, "fetched.yaml")
    with open(override, "w") as stream:
        stream.write("paths:\n  data_dir: %s\n" % fetched)
    bench.env["FASINATE_CONFIG"] = bench.config + ":" + override
    bench.data = fetched

    status, output = bench.run("pipeline.sh", "--dry-run")
    assert "would run 1 task(s)" in output and "plot         every .npz file" in output, output
    # (nor the model's preflight, as nothing of it is to be simulated)
    assert "  preflight         0      0       0          1            0" in output, output
    assert "not needed, so not made, as what they are for is done: the measurements of " \
        "caloclouds3_paper, the measurements of geant4, the measurements of geant4_2" \
        in output, output
    status, output = bench.run("simulate_geant4.sh", "--dry-run")
    assert "everything selected is done or not needed, so there is nothing to run" in output, \
        output

    with open(os.path.join(bench.repo, "models", "caloclouds3_paper.yaml")) as stream:
        model = stream.read().replace("name: CaloClouds3 (paper)", "name: Another model")
    with open(os.path.join(bench.repo, "models", "another.yaml"), "w") as stream:
        stream.write(model.replace('linestyle: "-."', 'linestyle: "--"'))
    status, output = bench.run("pipeline.sh", "--dry-run")
    # Its preflight, 4 simulations and reconstructions, its measurements and
    # the plots
    assert "would run 11 task(s)" in output, output
    status, output = bench.run("simulate_models.sh", "--dry-run")
    assert "would run 5 task(s)" in output, output
    assert "models       caloclouds3_paper" not in output, output
    status, output = bench.run("slurm_pipeline.sh", "--dry-run")
    assert "would submit 11 job(s)" in output, output
    status, output = bench.run("pipeline.sh")
    assert "made 11 of 11" in output, output

    for scenario in ("single_photon", "di_photon"):
        assert bench.files("sim_output", scenario) == ["another"]
        assert bench.files("rec_output", scenario) == ["another"]
    assert bench.files("observables") == ["another.npz", "caloclouds3_paper.npz", "geant4.npz",
                                          "geant4_2.npz", "plots"]
    for name, data in measured.items():
        assert open(bench.path("observables", name), "rb").read() == data, name
    rows = read_metrics(bench.path("observables", "plots", "metrics.csv"))
    assert sorted({row["simulator"] for row in rows}) == ["another", "caloclouds3_paper",
                                                          SECOND]
    status, output = bench.run("pipeline.sh")
    assert "everything selected is done" in output, output

    # Forcing Geant4 simulates both its samples again, and everything made
    # from them
    status, output = bench.run("pipeline.sh", "--dry-run", "-f", "geant4")
    assert "would run 19 task(s)" in output, output
    assert "(the Geant4 simulation of %s is made again)" % INPUTS[0] in output, output


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
    assert bench.files("sim_output", "di_photon", "caloclouds3_paper") == sorted(
        "%s_n3_simseed%d%s" % (DI_PHOTON_INPUT, seed, ending) for seed in (1, 2)
        for ending in (".root", ".log"))
    assert bench.files("observables") == ["caloclouds3_paper_n3.npz", "geant4_2_n3.npz",
                                          "geant4_n3.npz", "plots"]
    assert bench.files("sim_output", "single_photon", SECOND) == sorted(
        name + "_n3_simseed1000000" + ending for name in INPUTS
        for ending in (".edm4hep.root", ".log"))
    for name in ("metrics_n3.csv", "fits_n3.csv", "energy_resolution__single_photon_n3.png"):
        assert name in bench.files("observables", "plots"), name
    with np.load(bench.path("observables", "caloclouds3_paper_n3.npz")) as data:
        assert list(data["single_photon__n_events"]) == [3, 3]
        assert list(data["di_photon__n_events"]) == [3, 3]
    # The reconstruction ran on the whole test simulation, of 3 events
    log = open(bench.path("rec_output", "single_photon", "geant4", names[0] + ".log")).read()
    assert "--num-events=-1" in log and "--usingParticleGun" in log

    # The full sample is still to do, and the test sample is done
    status, output = bench.run("pipeline.sh", "--dry-run", "-n", "3")
    assert "everything selected is done" in output, output
    status, output = bench.run("pipeline.sh", "--dry-run")
    assert "would run 28 task(s)" in output, output


def test_a_failure_stops_only_what_needs_it(bench):
    # (both of Geant4's samples, which the fake ddsim calls geant4 alike)
    bench.fail_ddsim("geant4 *en80*")
    status, output = bench.run("pipeline.sh", check=False)
    assert status == 1, output
    assert "made 25 of 32" in output, output
    log = bench.path("sim_output", "single_photon", "geant4", INPUTS[1] + ".log")
    assert "# FAILED" in open(log).read()
    # Nothing half made is left, and only the Geant4 simulation of en80 is missing
    assert not any(name.startswith(".") for name in os.listdir(
        bench.path("sim_output", "single_photon", "geant4")))
    assert INPUTS[1] + ".edm4hep.root" not in bench.files("sim_output", "single_photon", "geant4")
    assert bench.files("observables") == ["caloclouds3_paper.npz"]

    bench.fail_ddsim(None)
    status, output = bench.run("pipeline.sh")
    assert "made 7 of 7" in output, output
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


def test_inputs_made_elsewhere_are_simulated_each_with_its_own_seed(bench):
    """tau_pairs' inputs are made elsewhere: there, they are done, and each is
    simulated once, with the seed it goes with; their reconstruction isn't a
    particle gun's; and the pi0s of the paper's figures 10 to 12 are measured
    and plotted."""
    bench.with_tau_pairs()
    status, output = bench.run("pipeline.sh", "--dry-run")
    assert "left out" not in output, output
    # The two inputs made elsewhere are done, the guns' three are to make
    assert stage_line("generate", 2, 3) in output, output
    assert "would run 44 task(s)" in output, output
    status, output = bench.run("pipeline.sh")
    assert "made 44 of 44" in output, output
    for simulator, ending in (("geant4", ".edm4hep.root"), ("caloclouds3_paper", ".root")):
        assert bench.files("sim_output", "tau_pairs", simulator) == sorted(
            name + e for name in TAU_SIMULATIONS for e in (ending, ".log"))
    for seed, name in zip(TAU_SEEDS, TAU_SIMULATIONS):
        log = open(bench.path("sim_output", "tau_pairs", "caloclouds3_paper",
                              name + ".log")).read()
        assert "--random.seed=%d" % seed in log and "# seed      %d\n" % seed in log
        log = open(bench.path("rec_output", "tau_pairs", "geant4", name + ".log")).read()
        assert "--noBeamCalReco" in log and "--usingParticleGun" not in log
    # and by Geant4's second sample, each with a seed of its own
    second = ["%s_simseed%d" % (name[:-len(".slcio")], seed + 1000000)
              for name, seed in zip(TAU_INPUTS, TAU_SEEDS)]
    assert bench.files("sim_output", "tau_pairs", SECOND) == sorted(
        name + e for name in second for e in (".edm4hep.root", ".log"))
    with np.load(bench.path("observables", SECOND + ".npz")) as data:
        assert list(data["tau_pairs__seeds"]) == [seed + 1000000 for seed in TAU_SEEDS]

    for simulator in SIMULATORS:
        with np.load(bench.path("observables", simulator + ".npz")) as data:
            assert [str(f) for f in data["tau_pairs__files"]] == [
                "rec_output/tau_pairs/%s/%s_REC.edm4hep.root" % (simulator, name)
                for name in TAU_SIMULATIONS]
            assert list(data["tau_pairs__seeds"]) == TAU_SEEDS
            assert list(data["tau_pairs__n_events"]) == [12, 12]
            # Every event has a pi0 from the tau-, every third one from the tau+ too
            assert list(data["pi0__tau_pairs__n_true"]) == [2, 1, 1] * 8
            counts = data["pi0__tau_pairs__n_candidates"]
            assert counts.shape == (24,) and counts.sum() == len(data["pi0__tau_pairs__mass"])
            match = data["pi0__tau_pairs__match"]
            good = match == 2
            assert good.sum() > 10 and (match == 1).any(), match
            # The good ones are of their MC pi0s, whose two photons are of the
            # pi0's mass, reconstructed a little off
            mass = data["pi0__tau_pairs__mass"][good]
            assert ((mass > 0.095) & (mass < 0.175)).all() and abs(np.median(mass) - 0.135) < 0.01
            true = data["pi0__tau_pairs__true_energy"]
            assert np.isfinite(true[good]).all() and np.isnan(true[match == 0]).all()
            ratio = data["pi0__tau_pairs__energy"][good] / true[good]
            assert ((ratio > 0.8) & (ratio < 1.2)).all(), ratio
            assert (np.abs(data["pi0__tau_pairs__theta"][good]
                           - data["pi0__tau_pairs__true_theta"][good]) < 1).all()
    rows = [row for row in read_metrics(bench.path("observables", "plots", "metrics.csv"))
            if row["observable"] == "pi0"]
    js = [row for row in rows if row["metric"] == "JS"]
    assert sorted((row["scenario"], row["profile"], row["simulator"], row["metric"])
                  for row in js) == sorted(
        ("tau_pairs", profile, simulator, "JS") for simulator in ("caloclouds3_paper", SECOND)
        for profile in ("energy", "mass", "phi", "theta"))
    # from the two seeds, paired, so with their spread
    assert all(float(row["uncertainty"]) > 0 for row in js), js
    # and the rest, the model's relative deviation from Geant4 in each category
    # of the reconstruction (the paper's figure 10)
    categories = {"categories_" + key for key in ("n_reco", "n_fake", "n_good", "n_confused",
                                                  "n_missed")}
    distances = [row for row in rows if row["metric"] == "W1"]
    chi2 = [row for row in rows if row["metric"] == "chi2"]
    assert rows != js and all(row["metric"] == "dev%" and row["profile"] in categories
                              for row in rows if row not in js + distances + chi2), rows
    # and their Wasserstein distances, seed by seed as the metric
    assert sorted(row["profile"] for row in distances) == sorted(
        ["energy", "mass", "phi", "theta"] * 2)
    assert all(float(row["uncertainty"]) > 0 for row in distances), distances
    # and the categories' chi^2
    assert [(row["profile"], row["uncertainty"]) for row in chi2] == [("categories", "")] * 2
    assert all(float(row["value"]) >= 0 for row in chi2)
    for figure in ("pi0_mass__tau_pairs", "pi0_energy__tau_pairs", "pi0_theta__tau_pairs",
                   "pi0_phi__tau_pairs", "pi0_categories__tau_pairs"):
        assert os.path.isfile(bench.path("observables", "plots", figure + ".png")), figure

    # Nothing makes the inputs made elsewhere, even forced
    path = bench.path("input_physics_samples", TAU_INPUTS[0])
    os.utime(path, (0, 0))
    status, output = bench.run("generate_input.sh", "-f", "tau_pairs")
    assert "everything selected is done" in output, output
    assert os.path.getmtime(path) == 0
    status, output = bench.run("pipeline.sh")
    assert "everything selected is done" in output, output


def test_a_scenario_without_its_inputs_is_left_out_and_the_rest_runs(bench):
    bench.with_tau_pairs(write=False)
    status, output = bench.run("pipeline.sh")
    assert ("the scenario tau_pairs is left out of this run, with the observables measured on "
            "it alone: its 2 inputs, made outside this repository, aren't in %s (taus_a.slcio, "
            "taus_b.slcio); put them there, or publish them in a record of `published`, to run "
            "it" % bench.path("input_physics_samples")) in output, output
    assert "made 32 of 32" in output, output
    with np.load(bench.path("observables", "geant4.npz")) as data:
        assert not [key for key in data.files if key.startswith(("pi0", "tau_pairs"))]
    # Asking for it is an error
    status, output = bench.run("simulate_geant4.sh", "-s", "tau_pairs", check=False)
    assert status == 1 and "the scenario tau_pairs can't run: its 2 inputs" in output, output

    # It runs with all of its inputs, or not at all
    write_tau_input(bench.path("input_physics_samples", TAU_INPUTS[0]), 12, 0)
    status, output = bench.run("pipeline.sh", "--dry-run")
    assert "1 of its 2 inputs, made outside this repository, isn't in %s (taus_b.slcio)" \
        % bench.path("input_physics_samples") in output, output
    assert "everything selected is done" in output, output
    write_tau_input(bench.path("input_physics_samples", TAU_INPUTS[1]), 12, 1)
    status, output = bench.run("pipeline.sh", "--dry-run")
    assert "left out" not in output, output
    # Its simulations and reconstructions, and every simulator measured again
    # with it, then the plots
    assert "would run 16 task(s)" in output, output
    assert "(geant4.npz has no pi0__tau_pairs__n_candidates)" in output, output
