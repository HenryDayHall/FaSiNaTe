"""The manifests of a data folder's records (ddml-bench manifest), from a run
of the pipeline on the fakes, and the reading of the logs' headers."""

import hashlib
import json
import os
import shutil
import tarfile

import yaml

from ddml_bench import ddml_configs, manifest, models, process

from conftest import TAU_INPUTS
from test_pipeline_local import DI_PHOTON_INPUT, DI_PHOTON_SIMULATIONS, INPUTS


def digest(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def load(out, record):
    with open(os.path.join(out, record, "manifest.json")) as stream:
        return json.load(stream)


def test_the_records_of_a_data_folder(bench):
    bench.run("pipeline.sh")
    # A test sample's measurements (as -n 2 would make)
    shutil.copy(bench.path("observables", "geant4.npz"), bench.path("observables", "geant4_n2.npz"))
    out = os.path.join(bench.tmp, "zenodo")
    status, output = bench.run("scripts/ddml-bench", "manifest", "--dry-run")
    assert "essential: 5 files" in output and "input_truth__di_photon: 1 files" in output, output

    status, output = bench.run("scripts/ddml-bench", "manifest", "--out", out, "--stage")
    assert "warning" not in output, output
    simulations = ["%s__%s__%s" % (stage, scenario, simulator)
                   for stage in ("sim_output", "rec_output")
                   for scenario in ("single_photon", "di_photon")
                   for simulator in ("geant4", "caloclouds3_paper")]
    assert sorted(os.listdir(out)) == sorted(
        ["essential", "input_truth__single_photon", "input_truth__di_photon", "other"]
        + simulations)

    # The essential record: the inputs and the full samples' measurements,
    # all required, with the scenarios and options that made the inputs
    essential = load(out, "essential")
    assert essential["format"] == "fasinate-manifest/2" and essential["record"] == "essential"
    assert essential["title"] == "FaSiNaTe benchmark data: inputs and measurements, all the " \
        "pipeline needs"
    by_name = dict((entry["name"], entry) for entry in essential["files"])
    assert sorted(by_name) == sorted([name + ".slcio" for name in INPUTS + [DI_PHOTON_INPUT]]
                                     + ["caloclouds3_paper.npz", "geant4.npz"])
    for name, entry in by_name.items():
        source = bench.path(entry["path"])
        assert entry["sha256"] == digest(source) and entry["size"] == os.path.getsize(source)
        assert entry["role"] == "required"
        # staged as links, under the same flat names
        assert os.path.realpath(os.path.join(out, "essential", name)) == os.path.realpath(source)
    gun = by_name[INPUTS[0] + ".slcio"]
    assert gun["path"] == "input_single_particle_gun/" + INPUTS[0] + ".slcio"
    assert gun["scenarios"] == ["single_photon"] and gun["generator"] == "ILD_Particle_Gun.py"
    assert "--save_dir" not in gun["options"] and "--energy" in gun["options"]
    assert essential["configuration"]["scenarios"]["di_photon"]["simulation_seeds"] == [1, 2]
    assert essential["configuration"]["pins"]["key4hep.release"] == "2026-04-08"

    # The measurements, what they hold, and what made them
    geant4 = by_name["geant4.npz"]
    assert geant4["path"] == "observables/geant4.npz" and geant4["up_to_date"] is True
    assert geant4["simulator"] == "geant4" and geant4["scenarios"] == ["di_photon",
                                                                       "single_photon"]
    assert geant4["observables"]["radial_profile"]["profiles"] == ["r100", "r30"]
    assert geant4["observables"]["photon_separation"]["scenarios"] == ["di_photon"]
    assert geant4["observables"]["energy"]["settings"]["particle_collections"] == ["PandoraPFOs"]
    assert [(r["path"], r["seed"], r["events"]) for r in geant4["reconstructions"]["di_photon"]] \
        == [("rec_output/di_photon/geant4/%s_REC.edm4hep.root" % name, seed, 30)
            for seed, name in zip((1, 2), DI_PHOTON_SIMULATIONS)]
    assert sorted(d["path"] for d in geant4["derived_from"]) == sorted(
        "input_single_particle_gun/%s.slcio" % name for name in INPUTS + [DI_PHOTON_INPUT])
    assert all(d["sha256"] == by_name[d["path"].split("/")[-1]]["sha256"]
               for d in geant4["derived_from"])
    assert geant4["stack"] == {
        "compact": "ILD/compact/ILD_l5_o1_v02/ILD_l5_o1_v02.xml",
        "ddml": "0123456789abcdef0123456789abcdef01234567",
        "ildconfig": geant4["stack"]["ildconfig"], "key4hep": "2026-04-08", "patches": "none",
        "reconstruction_args": "--noBeamCalReco",
        "steering": "ddsim_steer_geant4.py"}
    assert len(geant4["stack"]["ildconfig"]) == 40
    assert essential["stack"] == dict((k, v) for k, v in geant4["stack"].items()
                                      if k != "steering")
    caloclouds3 = by_name["caloclouds3_paper.npz"]
    # The presets its description gives, as its logs have them
    assert caloclouds3["stack"]["presets"] == " ".join(
        bench.description("caloclouds3_paper")["ddml"]["presets"])
    # What the model is, by its logs and by its description now: DDML's own
    # plugin, and no module of models/
    with open(os.path.join(bench.repo, "models", "caloclouds3_paper.yaml")) as stream:
        described = models.description_sha256(yaml.safe_load(stream))
    assert caloclouds3["stack"]["description_sha256"] == described
    assert caloclouds3["stack"]["python"] == "none"
    assert caloclouds3["model"]["description_sha256"] == described
    assert caloclouds3["model"]["python"] == {}
    assert "description_sha256" not in essential["stack"]

    # The catalogue says what every record holds
    catalogue = essential["catalogue"]
    assert sorted(catalogue) == sorted(os.listdir(out))
    assert catalogue["rec_output__di_photon__geant4"]["title"] == \
        "FaSiNaTe benchmark data: Geant4 reconstructions, di_photon"
    assert catalogue["input_truth__di_photon"]["paths"] == [
        "input_single_particle_gun/%s.h5" % DI_PHOTON_INPUT]

    # The simulations and reconstructions, file by file, by stage, scenario
    # and simulator, never fetched
    record = load(out, "rec_output__di_photon__caloclouds3_paper")
    assert record["title"] == \
        "FaSiNaTe benchmark data: caloclouds3_paper reconstructions, di_photon"
    assert sorted(entry["name"] for entry in record["files"]) == sorted(
        name + ending for name in DI_PHOTON_SIMULATIONS
        for ending in (".log", "_AIDA.root", "_PfoAnalysis.root", "_REC.edm4hep.root"))
    for entry in record["files"]:
        assert entry["role"] == "archive"
        assert entry["path"] == "rec_output/di_photon/caloclouds3_paper/" + entry["name"]
        assert entry["sha256"] == digest(bench.path(entry["path"]))
        assert os.path.islink(os.path.join(out, record["record"], entry["name"]))
    truth = load(out, "input_truth__single_photon")
    assert sorted(entry["name"] for entry in truth["files"]) == sorted(
        name + ".h5" for name in INPUTS)
    assert all(entry["role"] == "optional" for entry in truth["files"])

    # The rest in tar archives of their folders: the plots, the test sample's
    # measurements, which unpacked in another data folder go in place
    other = load(out, "other")
    archive, = other["files"]
    assert archive["name"] == "observables.tar" and archive["role"] == "archive"
    assert archive["sha256"] == digest(os.path.join(out, "other", "observables.tar"))
    names = [m["path"] for m in archive["members"]]
    assert "observables/geant4_n2.npz" in names and "observables/plots/metrics.csv" in names
    elsewhere = os.path.join(bench.tmp, "elsewhere")
    with tarfile.open(os.path.join(out, "other", "observables.tar")) as tar:
        if hasattr(tarfile, "data_filter"):
            tar.extractall(elsewhere, filter="data")
        else:
            tar.extractall(elsewhere)
    for member in archive["members"]:
        unpacked = os.path.join(elsewhere, member["path"])
        assert digest(unpacked) == member["sha256"] == digest(bench.path(member["path"]))
        assert abs(os.path.getmtime(unpacked)
                   - os.path.getmtime(bench.path(member["path"]))) < 1e-3

    # Records bigger than --max-record-gb are split into parts, each
    # reconstruction's files together
    status, output = bench.run("scripts/ddml-bench", "manifest", "--dry-run",
                               "--max-record-gb", "0.0003")
    assert "rec_output__di_photon__geant4__part1: 4 files" in output, output
    assert "rec_output__di_photon__geant4__part2: 4 files" in output, output
    assert "(FaSiNaTe benchmark data: Geant4 reconstructions, di_photon (part 2))" in output
    assert "sim_output__di_photon__geant4: 4 files" in output, output

    # Only some records
    again = os.path.join(bench.tmp, "again")
    bench.run("scripts/ddml-bench", "manifest", "--out", again, "--records", "*__di_photon__*")
    assert sorted(os.listdir(again)) == sorted(r for r in simulations if "di_photon" in r)
    assert not os.path.islink(os.path.join(again, "sim_output__di_photon__geant4",
                                           DI_PHOTON_SIMULATIONS[0] + ".edm4hep.root"))


def test_measurements_from_a_data_folder_without_logs_say_so(bench):
    bench.run("pipeline.sh")
    for folder in ("sim_output", "rec_output"):
        shutil.rmtree(bench.path(folder))
    out = os.path.join(bench.tmp, "zenodo")
    status, output = bench.run("scripts/ddml-bench", "manifest", "--out", out,
                               "--records", "essential")
    assert "logs of its simulations and reconstructions aren't there" in output, output
    entry = next(e for e in load(out, "essential")["files"] if e["name"] == "geant4.npz")
    assert entry["stack"] == {} and entry["up_to_date"] is True
    assert sorted(os.listdir(out)) == ["essential"]


def test_the_manifest_records_what_a_model_with_its_own_code_is(bench):
    """caloclouds3_recompiled, whose inference code is a module of models/: the
    logs of its simulations record the sha256 of its description and of that
    code as ddsim started, and its measurements' entry in the manifest has
    those (stack), and the description's as it is now (model)."""
    bench.add_model("caloclouds3_recompiled")
    bench.run("pipeline.sh")
    code = os.path.join(bench.repo, "models", "caloclouds3_recompiled.py")
    description = os.path.join(bench.repo, "models", "caloclouds3_recompiled.yaml")
    with open(code, "rb") as stream:
        sha256 = hashlib.sha256(stream.read()).hexdigest()
    with open(description) as stream:
        text = stream.read()
    described = models.description_sha256(yaml.safe_load(text))
    out = os.path.join(bench.tmp, "zenodo")

    def entry():
        bench.run("scripts/ddml-bench", "manifest", "--out", out, "--records", "essential")
        return dict((e["name"], e) for e in load(out, "essential")["files"])[
            "caloclouds3_recompiled.npz"]

    measured = entry()
    assert measured["stack"]["description_sha256"] == described
    assert measured["stack"]["python"] == "caloclouds3_recompiled:" + sha256
    # Its DDML model configurations, each running that code (the barrel's, the
    # endcap's, ...), as its description gives them
    configs = yaml.safe_load(text)["ddml"]["model_configs"]
    assert configs and all(entry["plugin_properties"]["PythonModule"] == "caloclouds3_recompiled"
                           for entry in configs)
    assert measured["stack"]["model_configs"] == ddml_configs.canonical(configs)
    assert measured["model"]["description_sha256"] == described
    assert measured["model"]["python"] == {"caloclouds3_recompiled": {
        "path": "models/caloclouds3_recompiled.py", "sha256": sha256}}

    # Both changed since: the logs still say what the simulations were made
    # with, and the description what the model is now
    with open(code, "a") as stream:
        stream.write("# changed\n")
    assert text.count("training_range: [1, 127]") == 1
    with open(description, "w") as stream:
        stream.write(text.replace("training_range: [1, 127]", "training_range: [1, 126]"))
    measured = entry()
    assert measured["stack"]["description_sha256"] == described
    assert measured["stack"]["python"] == "caloclouds3_recompiled:" + sha256
    assert measured["model"]["description_sha256"] not in (None, described)
    assert measured["model"]["python"]["caloclouds3_recompiled"]["sha256"] != sha256


def test_records_are_split_with_each_samples_files_together():
    files = [{"name": "%s%s" % (sample, ending), "size": size}
             for sample, size in (("a", 30), ("b", 30), ("c", 10))
             for ending in ("_REC.edm4hep.root", ".log")]
    warnings = []
    parts = manifest.split("rec_output__s__geant4", files, 85, warnings)
    assert [(name, [e["name"] for e in part]) for name, part in parts] == [
        ("rec_output__s__geant4__part1", ["a_REC.edm4hep.root", "a.log"]),
        ("rec_output__s__geant4__part2", ["b_REC.edm4hep.root", "b.log", "c_REC.edm4hep.root",
                                          "c.log"])]
    assert warnings == []
    assert manifest.split("rec_output__s__geant4", files, 200, warnings) == [
        ("rec_output__s__geant4", files)]
    # A sample bigger than a record makes one of its own, and says so
    parts = manifest.split("sim_output__s__geant4", files, 50, warnings)
    assert len(parts) == 3 and "the files of a are 0.0 GB" in warnings[0]
    # The essential record isn't split
    assert manifest.split("essential", files, 1, []) == [("essential", files)]


def test_the_headers_of_the_logs(tmp_path):
    log = tmp_path / "sim.log"
    log.write_text(
        "# simulate_models.sh: caloclouds3 on single_photon\n"
        "# input     /data/input_single_particle_gun/p22.slcio\n"
        "# seed      7\n"
        "# Key4hep   2026-04-08\n"
        "# DDML      263256cca31b37335cd2aae34b81e899bd2a5851, installed in /sw/DDML/install\n"
        "# command   (in /sw/DDML/scripts) ddsim --steeringFile /sw/DDML/scripts/ddsim_steer.py "
        "--compactFile /cvmfs/k4geo/ILD/compact/ILD_l5_o1_v02/ILD_l5_o1_v02.xml --ml-model "
        "CC3_BARREL_PY_INTERFACE --inputFiles /data/p22.slcio --outputFile /data/.x.root "
        "--numberOfEvents=-1\n"
        "ddsim output\n")
    assert manifest.read_log(str(log)) == {
        "key4hep": "2026-04-08", "ddml": "263256cca31b37335cd2aae34b81e899bd2a5851", "seed": 7,
        "steering": "ddsim_steer.py", "compact": "ILD/compact/ILD_l5_o1_v02/ILD_l5_o1_v02.xml",
        "presets": "CC3_BARREL_PY_INTERFACE"}
    # A model with DDML model configurations written out: them, as one string
    configs = [{"preset": "CC3_BARREL", "plugin_properties": {"ModelPath": "../models/a.pt"}},
               {"plugin": "P/B", "geometry": "ILD_BARREL", "applicable_particles": ["gamma"],
                "triggers": {"gamma": 10}}]
    log.write_text(
        "# ddml-bench models: caloclouds3_libtorch on di_photon\n"
        "# model     %s, of models/caloclouds3_libtorch.yaml as read, without what only the "
        "plots use\n"
        "# python    own:%s, from models/own.py\n" % ("a" * 64, "b" * 64) +
        "# DDML      263256cca31b37335cd2aae34b81e899bd2a5851, installed in /sw/DDML/install\n"
        "# command   (in /sw/DDML/scripts) " + process.quote(
            ["ddsim", "--steeringFile", "/repo/scripts/ddsim_steer_models.py", "--compactFile",
             "/k/ILD/compact/ILD_l5_o1_v02/ILD_l5_o1_v02.xml"]
            + [arg for c in configs for arg in ("--ml-config", ddml_configs.option(c))]
            + ["--inputFiles", "/d/p22.slcio", "--outputFile", "/d/.x.root",
               "--numberOfEvents=-1", "--random.seed=2"]) + "\n")
    assert manifest.read_log(str(log)) == {
        "ddml": "263256cca31b37335cd2aae34b81e899bd2a5851", "steering": "ddsim_steer_models.py",
        "compact": "ILD/compact/ILD_l5_o1_v02/ILD_l5_o1_v02.xml",
        "model_configs": ddml_configs.canonical(configs),
        "description_sha256": "a" * 64, "python": "own:" + "b" * 64}
    log.write_text(
        "# reconstruct_simulation.sh: geant4 on single_photon\n"
        "# Key4hep   2026-04-08\n"
        "# ILDConfig 6922e3bbd7e7d221557a2aa62c945a23cf318467, in /sw/ILDConfig/StandardConfig\n"
        "# patches   marlintrkprocessors (patches in the configuration)\n"
        "# command   (in a folder linking to /sw/x) k4run /w/ILDReconstruction.py "
        "--inputFiles=/d/a.edm4hep.root --outputFileBase=/d/.a --compactFile=/k/ILD/compact/"
        "ILD_l5_o1_v02/ILD_l5_o1_v02.xml --num-events=-1 --noBeamCalReco --usingParticleGun\n")
    assert manifest.read_log(str(log)) == {
        "key4hep": "2026-04-08", "ildconfig": "6922e3bbd7e7d221557a2aa62c945a23cf318467",
        "patches": "marlintrkprocessors", "compact": "ILD/compact/ILD_l5_o1_v02/ILD_l5_o1_v02.xml",
        "reconstruction_args": "--noBeamCalReco"}
    assert manifest.read_log(str(tmp_path / "none.log")) == {}


def test_which_record_each_file_goes_into():
    assert manifest.classify("input_single_particle_gun/a.slcio") == (
        "essential", "required", "a.slcio", None)
    assert manifest.classify("observables/geant4.npz") == ("essential", "required",
                                                            "geant4.npz", None)
    assert manifest.classify("input_single_particle_gun/a.h5",
                             {"input_single_particle_gun/a.slcio": "s"}) == (
        "input_truth__s", "optional", "a.h5", None)
    assert manifest.classify("input_single_particle_gun/a.h5")[0] == "input_truth"
    assert manifest.classify("observables/geant4_n10.npz") == ("other", "archive",
                                                                "observables.tar",
                                                                "observables.tar")
    assert manifest.classify("sim_output/custom/geant4/x.log") == (
        "sim_output__custom__geant4", "archive", "x.log", None)
    assert manifest.classify("slurm/pipeline_1/logs/001-plot.out")[2] == "slurm.tar"
    # Full paths recorded before the records were relative, from any data folder
    assert manifest.recorded_path("/data/dust/x/data/rec_output/s/geant4/a_REC.edm4hep.root") \
        == "rec_output/s/geant4/a_REC.edm4hep.root"
    assert manifest.reconstruction_sources("rec_output/s/geant4/a_n5_simseed2_REC.edm4hep.root") \
        == ("input_single_particle_gun/a.slcio", "sim_output/s/geant4/a_n5_simseed2.log",
            "rec_output/s/geant4/a_n5_simseed2.log")
    # The inputs of each scenario in its own folder
    assert manifest.classify("input_physics_samples/taus.slcio") == (
        "essential", "required", "taus.slcio", None)
    assert manifest.classify("my_inputs/taus.slcio", None, ("my_inputs",))[0] == "essential"
    assert manifest.classify("my_inputs/taus.slcio")[0] == "other"
    assert manifest.classify("input_physics_samples/t.h5", {"input_physics_samples/t.slcio": "x"}
                             )[0] == "input_truth__x"
    assert manifest.reconstruction_sources("rec_output/taus/geant4/t_simseed42_REC.edm4hep.root",
                                           {"taus": "input_physics_samples"})[0] \
        == "input_physics_samples/t.slcio"


def test_the_records_of_a_data_folder_with_inputs_made_elsewhere(bench):
    """tau_pairs' inputs, in their own folder, are required inputs like the
    guns', said to be made elsewhere, and the measurements were derived from
    them; a configured input that isn't there is warned about."""
    bench.with_tau_pairs()
    bench.run("pipeline.sh")
    out = os.path.join(bench.tmp, "zenodo")
    status, output = bench.run("scripts/ddml-bench", "manifest", "--out", out, "--records",
                               "essential")
    assert "warning" not in output, output
    essential = load(out, "essential")
    by_name = dict((entry["name"], entry) for entry in essential["files"])
    for name in TAU_INPUTS:
        entry = by_name[name]
        assert entry["path"] == "input_physics_samples/" + name and entry["role"] == "required"
        assert entry["external"] is True and entry["scenarios"] == ["tau_pairs"]
        assert "options" not in entry and "generator" not in entry
        assert entry["sha256"] == digest(bench.path(entry["path"]))
    geant4 = by_name["geant4.npz"]
    assert geant4["up_to_date"] is True
    assert sorted(geant4["scenarios"]) == ["di_photon", "single_photon", "tau_pairs"]
    assert geant4["observables"]["pi0"]["scenarios"] == ["tau_pairs"]
    assert [(r["seed"], r["events"]) for r in geant4["reconstructions"]["tau_pairs"]] == [
        (42, 12), (1764, 12)]
    assert sorted(d["path"] for d in geant4["derived_from"] if "physics" in d["path"]) == [
        "input_physics_samples/" + name for name in TAU_INPUTS]
    # The reconstructions of the physics samples aren't told they are a
    # particle gun's, as the others are, which isn't the software's stack
    assert geant4["stack"]["reconstruction_args"] == "--noBeamCalReco"
    assert essential["stack"]["reconstruction_args"] == "--noBeamCalReco"

    os.remove(bench.path("input_physics_samples", TAU_INPUTS[1]))
    status, output = bench.run("scripts/ddml-bench", "manifest", "--out", out, "--records",
                               "essential")
    assert "the configuration's input input_physics_samples/taus_b.slcio isn't in the data " \
        "folder, so runs here leave out tau_pairs, and its measurements with it" in output, output
