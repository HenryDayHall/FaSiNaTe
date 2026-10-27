"""Fetching published data instead of making it (ddml_bench/fetch.py): a
record published from a run of the pipeline on the fakes, with its manifest
(ddml-bench manifest), served from files as Zenodo serves it, and fresh data
folders that fetch from it."""

import hashlib
import json
import os
import shutil

import pytest

from conftest import Bench
from test_pipeline_local import DI_PHOTON_INPUT, INPUTS, read_metrics

RECORD = "123"


def digest(path, kind="sha256"):
    return hashlib.new(kind, open(path, "rb").read()).hexdigest()


@pytest.fixture(scope="module")
def published(tmp_path_factory):
    """A record of the essential files of a run of the pipeline, as
    {"folder": where the record's files are, "data": the data folder it was
    published from, "md5": the md5 sum of its manifest}."""
    bench = Bench(str(tmp_path_factory.mktemp("publisher")))
    bench.run("pipeline.sh")
    build = os.path.join(bench.tmp, "build")
    bench.run("scripts/ddml-bench", "manifest", "--out", build, "--records", "essential",
              "--stage", "--copy")
    zenodo = os.path.join(bench.tmp, "zenodo")
    shutil.copytree(os.path.join(build, "essential"), os.path.join(zenodo, RECORD))
    return {"zenodo": zenodo, "folder": os.path.join(zenodo, RECORD), "data": bench.data,
            "md5": digest(os.path.join(zenodo, RECORD, "manifest.json"), "md5")}


def fetching(bench, published, md5=None, extra=""):
    """Configure the bench to fetch from the published record, into a data
    folder of its own."""
    override = os.path.join(bench.tmp, "published.yaml")
    with open(override, "w") as stream:
        stream.write("published:\n  enabled: true\n  records:\n    - zenodo: %s\n"
                     "      manifest_md5: \"%s\"\n  url: \"file://%s/{record}/{name}\"\n%s"
                     % (RECORD, md5 or published["md5"], published["zenodo"], extra))
    bench.env["FASINATE_CONFIG"] = bench.config + ":" + override
    return override


def test_a_fresh_data_folder_fetches_what_is_published(bench, published):
    fetching(bench, published)
    status, output = bench.run("pipeline.sh", "--dry-run")
    assert "would fetch 5 published file(s)" in output, output
    # The inputs and measurements are fetched, nothing is simulated or
    # reconstructed for them, and the plots are made
    assert "would run 6 task(s)" in output, output
    assert "  generate          0        3      0" in output, output
    assert "  observables       0        2      0" in output, output
    assert not os.path.exists(bench.path("observables"))

    status, output = bench.run("pipeline.sh")
    assert "fetching 5 published file(s)" in output and "made 1 of 1" in output, output
    for folder in ("input_single_particle_gun", "observables"):
        for name in bench.files(folder):
            if name.endswith((".slcio", ".npz")):
                assert digest(bench.path(folder, name)) == \
                    digest(os.path.join(published["data"], folder, name)), name
    assert bench.files("input_single_particle_gun") == sorted(
        name + ".slcio" for name in INPUTS + [DI_PHOTON_INPUT])
    assert sorted({row["simulator"] for row in read_metrics(
        bench.path("observables", "plots", "metrics.csv"))}) == ["caloclouds3"]
    status, output = bench.run("pipeline.sh")
    assert "everything selected is done" in output and "fetching" not in output, output

    # A model added runs only its own tasks, against the published measurements
    with open(os.path.join(bench.repo, "models", "caloclouds3.yaml")) as stream:
        model = stream.read().replace('linestyle: "-"', 'linestyle: "--"')
    with open(os.path.join(bench.repo, "models", "another.yaml"), "w") as stream:
        stream.write(model)
    status, output = bench.run("pipeline.sh")
    assert "made 10 of 10" in output and "fetching" not in output, output
    assert sorted({row["simulator"] for row in read_metrics(
        bench.path("observables", "plots", "metrics.csv"))}) == ["another", "caloclouds3"]


def test_slurm_fetches_on_the_login_node_and_submits_the_rest(bench, published):
    fetching(bench, published)
    status, output = bench.run("slurm_pipeline.sh", "--dry-run")
    assert "would fetch 5 published file(s)" in output, output
    assert "would submit 1 job(s)" in output and "plot" in output, output
    status, output = bench.run("slurm_pipeline.sh")
    assert "fetching 5 published file(s)" in output, output
    assert "submitting 1 job(s)" in output, output
    plot, = bench.slurm_jobs()
    assert plot["state"] == "COMPLETED" and plot["after"] == []
    assert "metrics.csv" in bench.files("observables", "plots")


def test_published_measurements_of_other_settings_arent_used(bench, published):
    fetching(bench, published, extra="observables:\n  energy:\n    pdg: [22]\n")
    status, output = bench.run("pipeline.sh", "--dry-run")
    assert "the published geant4.npz (Zenodo record 123) isn't used, as the settings of " \
        "energy differ from this configuration's" in output, output
    # The inputs are still fetched, and everything else made from them
    assert "would fetch 3 published file(s)" in output, output
    assert "would run 22 task(s)" in output, output


def test_published_inputs_made_otherwise_arent_used(bench, published):
    fetching(bench, published, extra="scenarios:\n  single_photon:\n    input:\n"
                                     "      n_events: 7\n")
    status, output = bench.run("pipeline.sh", "--dry-run")
    assert "the published %s.slcio (Zenodo record 123) isn't used, as it was made with other " \
        "options (--n_events: 7 here, 6 there)" % INPUTS[0] in output, output
    # nor the measurements made from them
    assert "the published geant4.npz (Zenodo record 123) isn't used, as its input %s.slcio " \
        "isn't here and won't be fetched" % INPUTS[0] in output, output
    assert "would fetch 1 published file(s)" in output, output  # di_photon's input


def test_measurements_of_other_inputs_than_yours_arent_used(bench, published):
    fetching(bench, published)
    os.makedirs(bench.path("input_single_particle_gun"))
    with open(bench.path("input_single_particle_gun", INPUTS[0] + ".slcio"), "w") as stream:
        stream.write("my own events\n")
    status, output = bench.run("pipeline.sh", "--dry-run")
    assert "your %s.slcio differs from the one it was measured from" % INPUTS[0] in output, output
    # Nothing that is there is replaced
    assert "would fetch 2 published file(s)" in output, output
    assert open(bench.path("input_single_particle_gun", INPUTS[0] + ".slcio")).read() == \
        "my own events\n"


def test_what_a_run_makes_itself_isnt_fetched(bench, published):
    fetching(bench, published)
    status, output = bench.run("pipeline.sh", "--dry-run", "--no-fetch")
    assert "would fetch" not in output and "would run 22 task(s)" in output, output
    status, output = bench.run("pipeline.sh", "--dry-run", "-f", "observables")
    assert "would fetch 3 published file(s)" in output, output
    # -n N: the test sample's measurements aren't published, its inputs are
    status, output = bench.run("pipeline.sh", "--dry-run", "-n", "2")
    assert "would fetch 3 published file(s)" in output, output


def test_a_published_file_that_isnt_what_its_manifest_says_stops_the_run(bench, published):
    tampered = os.path.join(bench.tmp, "tampered")
    shutil.copytree(published["zenodo"], tampered)
    with open(os.path.join(tampered, RECORD, "geant4.npz"), "ab") as stream:
        stream.write(b"more")
    fetching(bench, dict(published, zenodo=tampered))
    status, output = bench.run("pipeline.sh", check=False)
    assert status == 1 and "geant4.npz isn't the published file: its sha256 is" in output, output
    assert not os.path.exists(bench.path("observables", "geant4.npz"))
    assert not [n for n in os.listdir(bench.path("observables")) if n.startswith(".")]

    # A manifest that isn't the one configured isn't used at all
    fetching(bench, published, md5="0" * 32)
    status, output = bench.run("pipeline.sh", "--dry-run", check=False)
    assert status == 1 and "can't read the published data of Zenodo record 123" in output
    assert "its md5 is %s, not %s" % (published["md5"], "0" * 32) in output, output
    assert "--no-fetch" in output


def test_a_record_that_cant_be_reached_stops_the_run(bench, published):
    fetching(bench, dict(published, zenodo=os.path.join(bench.tmp, "nowhere")))
    status, output = bench.run("pipeline.sh", "--dry-run", check=False)
    assert status == 1 and "can't read the published data of Zenodo record 123" in output, output


def test_the_manifest_describes_the_models():
    """New manifests record each model's description, which fetching checks."""
    from ddml_bench import fetch
    task = {"simulator": "cc3", "params": {"observables": {}, "files": {},
                                           "data_dir": "/data"}}
    plan = {"models": [{"name": "cc3", "presets": ["P"], "compact_file": "D",
                        "weights_md5": "a" * 32}],
            "reference": {"name": "geant4", "compact_file": "D"}}
    published = {"simulator": "cc3", "observables": {}, "reconstructions": {},
                 "stack": {"presets": "P", "compact": "D"}}
    assert fetch.measurements_match(task, published, plan) == (True, "")
    assert fetch.measurements_match(task, dict(published, model={"weights_md5": "b" * 32}),
                                    plan)[1].startswith("the model's weights differ")
    assert fetch.measurements_match(task, dict(published, stack={"presets": "Q"}), plan) == (
        False, "the model ran with the presets Q, not P")
    assert fetch.measurements_match(task, dict(published, stack={"compact": "E"}), plan) == (
        False, "they were simulated in E, not D")
    assert fetch.gun_options(["--energy", "20", "--save_dir", "/x", "--pid", "22"]) == \
        fetch.gun_options(["--pid", "22", "--energy", "20"])


def test_a_manifest_sum_read_as_a_number_is_caught(bench, published):
    override = fetching(bench, published)
    with open(override) as stream:
        text = stream.read()
    with open(override, "w") as stream:
        stream.write(text.replace('"%s"' % published["md5"], "1" * 32))
    status, output = bench.run("pipeline.sh", "--dry-run", check=False)
    assert status == 1 and "should be 32 hex digits, in quotes" in output, output


def test_inputs_made_elsewhere_are_fetched_when_published(bench):
    """A record with tau_pairs' inputs, made elsewhere: a run fetches them (even
    forcing the generate stage, which can't make them), and the scenario runs;
    without them, it would be left out."""
    from conftest import TAU_INPUTS, write_tau_input
    folder = os.path.join(bench.tmp, "zenodo", RECORD)
    files = []
    for number, name in enumerate(TAU_INPUTS):
        path = os.path.join(folder, name)
        write_tau_input(path, 12, number)
        files.append({"name": name, "path": "input_physics_samples/" + name,
                      "sha256": digest(path), "size": os.path.getsize(path), "role": "required",
                      "kind": "input", "external": True, "scenarios": ["tau_pairs"]})
    with open(os.path.join(folder, "manifest.json"), "w") as stream:
        json.dump({"format": "fasinate-manifest/2", "record": "essential", "files": files},
                  stream)
    fetching(bench, {"zenodo": os.path.dirname(folder),
                     "md5": digest(os.path.join(folder, "manifest.json"), "md5")})
    bench.with_tau_pairs(write=False)

    status, output = bench.run("pipeline.sh", "--dry-run")
    assert "would fetch 2 published file(s), " in output, output
    assert "left out" not in output and "geant4 tau_pairs/taus_a_simseed42" in output, output
    status, output = bench.run("pipeline.sh", "--dry-run", "-f", "generate")
    assert "would fetch 2 published file(s), " in output, output

    status, output = bench.run("pipeline.sh", "--stages", "generate")
    assert "fetching 2 published file(s)" in output and "made 3 of 3" in output, output
    for name in TAU_INPUTS:
        assert digest(bench.path("input_physics_samples", name)) == digest(
            os.path.join(folder, name))
    status, output = bench.run("pipeline.sh", "--dry-run")
    assert "would fetch" not in output and "left out" not in output, output
