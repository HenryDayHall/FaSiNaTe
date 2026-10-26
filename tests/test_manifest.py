"""The manifests of a data folder's records (ddml-bench manifest), from a run
of the pipeline on the fakes, and the reading of the logs' headers."""

import hashlib
import json
import os
import shutil
import tarfile

from ddml_bench import manifest

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
    assert "inputs: 6 files" in output and "measurements: 2 files" in output, output

    status, output = bench.run("scripts/ddml-bench", "manifest", "--out", out, "--stage")
    assert "warning" not in output, output

    # The inputs: the .slcio files required, the guns' truth optional, with
    # the scenarios and options that made them
    inputs = load(out, "inputs")
    assert inputs["format"] == "fasinate-manifest/1" and inputs["record"] == "inputs"
    by_name = dict((entry["name"], entry) for entry in inputs["files"])
    assert sorted(by_name) == sorted(name + ending for name in INPUTS + [DI_PHOTON_INPUT]
                                     for ending in (".h5", ".slcio"))
    for name, entry in by_name.items():
        source = bench.path(entry["path"])
        assert entry["path"] == "input_single_particle_gun/" + name
        assert entry["sha256"] == digest(source) and entry["size"] == os.path.getsize(source)
        assert entry["role"] == ("required" if name.endswith(".slcio") else "optional")
        # staged as links, under the same flat names
        assert os.path.realpath(os.path.join(out, "inputs", name)) == os.path.realpath(source)
    gun = by_name[INPUTS[0] + ".slcio"]
    assert gun["scenarios"] == ["single_photon"] and gun["generator"] == "ILD_Particle_Gun.py"
    assert "--save_dir" not in gun["options"] and "--energy" in gun["options"]
    assert by_name[DI_PHOTON_INPUT + ".h5"]["scenarios"] == ["di_photon"]
    assert inputs["configuration"]["scenarios"]["di_photon"]["simulation_seeds"] == [1, 2]
    assert inputs["configuration"]["pins"]["key4hep.release"] == "2026-04-08"

    # The measurements of the full samples, what they hold, and what made them
    measurements = load(out, "measurements")
    by_name = dict((entry["name"], entry) for entry in measurements["files"])
    assert sorted(by_name) == ["caloclouds3.npz", "geant4.npz"]
    geant4 = by_name["geant4.npz"]
    assert geant4["role"] == "required" and geant4["up_to_date"] is True
    assert geant4["simulator"] == "geant4" and geant4["scenarios"] == ["di_photon",
                                                                       "single_photon"]
    assert geant4["sha256"] == digest(bench.path("observables", "geant4.npz"))
    assert geant4["observables"]["radial_profile"]["profiles"] == ["r100", "r30"]
    assert geant4["observables"]["photon_separation"]["scenarios"] == ["di_photon"]
    assert geant4["observables"]["energy"]["settings"]["particle_collections"] == ["PandoraPFOs"]
    assert [(r["path"], r["seed"], r["events"]) for r in geant4["reconstructions"]["di_photon"]] \
        == [("rec_output/di_photon/geant4/%s_REC.edm4hep.root" % name, seed, 30)
            for seed, name in zip((1, 2), DI_PHOTON_SIMULATIONS)]
    hashes = dict((entry["path"], entry["sha256"]) for entry in inputs["files"])
    assert sorted(d["path"] for d in geant4["derived_from"]) == sorted(
        "input_single_particle_gun/%s.slcio" % name for name in INPUTS + [DI_PHOTON_INPUT])
    assert all(d["sha256"] == hashes[d["path"]] for d in geant4["derived_from"])
    assert geant4["stack"] == {
        "compact": "ILD/compact/ILD_l5_o1_v02/ILD_l5_o1_v02.xml",
        "ddml": "0123456789abcdef0123456789abcdef01234567",
        "ildconfig": geant4["stack"]["ildconfig"], "key4hep": "2026-04-08", "patches": "none",
        "reconstruction_args": geant4["stack"]["reconstruction_args"],
        "steering": "ddsim_steer_geant4.py"}
    assert len(geant4["stack"]["ildconfig"]) == 40
    assert geant4["stack"]["reconstruction_args"] == "--noBeamCalReco --usingParticleGun"
    assert measurements["stack"] == dict((k, v) for k, v in geant4["stack"].items()
                                         if k != "steering")
    assert by_name["caloclouds3.npz"]["stack"]["presets"] == "CC3_BARREL_PY_INTERFACE"

    # Everything else in archives of the folders, never fetched
    bulk = load(out, "bulk")
    by_name = dict((entry["name"], entry) for entry in bulk["files"])
    assert sorted(by_name) == sorted(
        ["%s__%s__%s.tar" % (top, scenario, simulator) for top in ("sim_output", "rec_output")
         for scenario in ("single_photon", "di_photon") for simulator in ("geant4", "caloclouds3")]
        + ["observables.tar"])
    assert all(entry["role"] == "archive" for entry in bulk["files"])
    archive = by_name["rec_output__di_photon__geant4.tar"]
    assert archive["sha256"] == digest(os.path.join(out, "bulk", archive["name"]))
    assert [m["path"] for m in archive["members"]] == sorted(
        "rec_output/di_photon/geant4/%s%s" % (name, ending) for name in DI_PHOTON_SIMULATIONS
        for ending in (".log", "_AIDA.root", "_PfoAnalysis.root", "_REC.edm4hep.root"))
    # The test sample's measurements and the plots go with the bulk
    names = [m["path"] for m in by_name["observables.tar"]["members"]]
    assert "observables/geant4_n2.npz" in names and "observables/plots/metrics.csv" in names

    # Unpacked in another data folder, the archives put their members in place
    elsewhere = os.path.join(bench.tmp, "elsewhere")
    for entry in bulk["files"]:
        with tarfile.open(os.path.join(out, "bulk", entry["name"])) as tar:
            if hasattr(tarfile, "data_filter"):
                tar.extractall(elsewhere, filter="data")
            else:
                tar.extractall(elsewhere)
        for member in entry["members"]:
            unpacked = os.path.join(elsewhere, member["path"])
            assert digest(unpacked) == member["sha256"] == digest(bench.path(member["path"]))
            # with their times, so that reconstructions stay older than measurements
            assert abs(os.path.getmtime(unpacked)
                       - os.path.getmtime(bench.path(member["path"]))) < 1e-3


def test_measurements_from_a_data_folder_without_logs_say_so(bench):
    bench.run("pipeline.sh")
    for folder in ("sim_output", "rec_output"):
        shutil.rmtree(bench.path(folder))
    out = os.path.join(bench.tmp, "zenodo")
    status, output = bench.run("scripts/ddml-bench", "manifest", "--out", out,
                               "--records", "measurements")
    assert "logs of its simulations and reconstructions aren't there" in output, output
    entry = load(out, "measurements")["files"][0]
    assert entry["stack"] == {} and entry["up_to_date"] is True
    assert sorted(os.listdir(out)) == ["measurements"]


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
        "reconstruction_args": "--noBeamCalReco --usingParticleGun"}
    assert manifest.read_log(str(tmp_path / "none.log")) == {}


def test_which_record_each_file_goes_into():
    assert manifest.classify("input_single_particle_gun/a.slcio") == ("inputs", "required",
                                                                       "a.slcio")
    assert manifest.classify("input_single_particle_gun/a.h5") == ("inputs", "optional", "a.h5")
    assert manifest.classify("observables/geant4.npz") == ("measurements", "required",
                                                            "geant4.npz")
    assert manifest.classify("observables/geant4_n10.npz") == ("bulk", "archive",
                                                                "observables.tar")
    assert manifest.classify("sim_output/custom/geant4/x.log") == (
        "bulk", "archive", "sim_output__custom__geant4.tar")
    assert manifest.classify("slurm/pipeline_1/logs/001-plot.out")[2] == "slurm.tar"
    # Full paths recorded before the records were relative, from any data folder
    assert manifest.recorded_path("/data/dust/x/data/rec_output/s/geant4/a_REC.edm4hep.root") \
        == "rec_output/s/geant4/a_REC.edm4hep.root"
    assert manifest.reconstruction_sources("rec_output/s/geant4/a_n5_simseed2_REC.edm4hep.root") \
        == ("input_single_particle_gun/a.slcio", "sim_output/s/geant4/a_n5_simseed2.log",
            "rec_output/s/geant4/a_n5_simseed2.log")
