"""Fetching published data instead of making it (ddml_bench/fetch.py): a
record published from a run of the pipeline on the fakes, with its manifest
(ddml-bench manifest), served from files as Zenodo serves it, and fresh data
folders that fetch from it."""

import hashlib
import io
import json
import os
import shutil
import tarfile

import pytest

from conftest import Bench
from ddml_bench import ddml_configs, fetch
from test_manifest import write_timing
from test_pipeline_local import DI_PHOTON_INPUT, INPUTS, read_metrics

RECORD = "123"
# A new version of it, with the timing benchmark's results and table
TIMED_RECORD = "124"


def digest(path, kind="sha256"):
    return hashlib.new(kind, open(path, "rb").read()).hexdigest()


@pytest.fixture(scope="module")
def publisher(tmp_path_factory):
    """A run of the pipeline, to publish."""
    bench = Bench(str(tmp_path_factory.mktemp("publisher")))
    bench.run("pipeline.sh")
    return bench


def publish(bench, record):
    """The essential record of the bench's data folder, as Zenodo record
    record: {"zenodo": the folder of the records, "folder": where the
    record's files are, "data": the data folder it was published from, "md5":
    the md5 sum of its manifest, "record": its id}."""
    build = os.path.join(bench.tmp, "build" + record)
    bench.run("scripts/ddml-bench", "manifest", "--out", build, "--records", "essential",
              "--stage", "--copy")
    zenodo = os.path.join(bench.tmp, "zenodo")
    shutil.copytree(os.path.join(build, "essential"), os.path.join(zenodo, record))
    return {"zenodo": zenodo, "folder": os.path.join(zenodo, record), "data": bench.data,
            "md5": digest(os.path.join(zenodo, record, "manifest.json"), "md5"),
            "record": record}


@pytest.fixture(scope="module")
def published(publisher):
    """A record of the essential files of a run of the pipeline (publish())."""
    return publish(publisher, RECORD)


@pytest.fixture(scope="module")
def published_timing(publisher, published):
    """A new version of the record of published, with the timing benchmark's
    results and table (made up) of its data folder, timing.tar
    (publish(), with "timing": the paths of its members)."""
    members, _ = write_timing(publisher.data)
    return dict(publish(publisher, TIMED_RECORD), timing=members)


def fetching(bench, published, md5=None, extra=""):
    """Configure the bench to fetch from the published record, into a data
    folder of its own."""
    override = os.path.join(bench.tmp, "published.yaml")
    with open(override, "w") as stream:
        stream.write("published:\n  enabled: true\n  records:\n    - zenodo: %s\n"
                     "      manifest_md5: \"%s\"\n  url: \"file://%s/{record}/{name}\"\n%s"
                     % (published.get("record", RECORD), md5 or published["md5"],
                        published["zenodo"], extra))
    bench.env["FASINATE_CONFIG"] = bench.config + ":" + override
    return override


def test_a_fresh_data_folder_fetches_what_is_published(bench, published):
    fetching(bench, published)
    status, output = bench.run("pipeline.sh", "--dry-run")
    # (Geant4's second sample's measurements too)
    assert "would fetch 6 published file(s)" in output, output
    # The inputs and measurements are fetched, nothing is simulated or
    # reconstructed for them, and the plots are made
    assert "would run 7 task(s)" in output, output
    assert "  generate          0        3      0" in output, output
    assert "  observables       0        3      0" in output, output
    assert not os.path.exists(bench.path("observables"))

    status, output = bench.run("pipeline.sh")
    assert "fetching 6 published file(s)" in output and "made 1 of 1" in output, output
    for folder in ("input_single_particle_gun", "observables"):
        for name in bench.files(folder):
            if name.endswith((".slcio", ".npz")):
                assert digest(bench.path(folder, name)) == \
                    digest(os.path.join(published["data"], folder, name)), name
    assert bench.files("input_single_particle_gun") == sorted(
        name + ".slcio" for name in INPUTS + [DI_PHOTON_INPUT])
    assert sorted({row["simulator"] for row in read_metrics(
        bench.path("observables", "plots", "metrics.csv"))}) == ["caloclouds3_paper", "geant4_2"]
    status, output = bench.run("pipeline.sh")
    assert "everything selected is done" in output and "fetching" not in output, output

    # A model added runs only its own tasks (its preflight first), against the
    # published measurements
    with open(os.path.join(bench.repo, "models", "caloclouds3_paper.yaml")) as stream:
        model = stream.read().replace('linestyle: "-."', 'linestyle: "--"')
    with open(os.path.join(bench.repo, "models", "another.yaml"), "w") as stream:
        stream.write(model)
    status, output = bench.run("pipeline.sh")
    assert "made 11 of 11" in output and "fetching" not in output, output
    assert sorted({row["simulator"] for row in read_metrics(
        bench.path("observables", "plots", "metrics.csv"))}) == [
        "another", "caloclouds3_paper", "geant4_2"]


def test_slurm_fetches_on_the_login_node_and_submits_the_rest(bench, published):
    fetching(bench, published)
    status, output = bench.run("slurm_pipeline.sh", "--dry-run")
    assert "would fetch 6 published file(s)" in output, output
    assert "would submit 1 job(s)" in output and "plot" in output, output
    status, output = bench.run("slurm_pipeline.sh")
    assert "fetching 6 published file(s)" in output, output
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
    assert "would run 32 task(s)" in output, output


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
    assert "would fetch" not in output and "would run 32 task(s)" in output, output
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

    # A model with DDML model configurations written out: by those it ran with
    configs = [{"preset": "P", "plugin_properties": {"ModelPath": "../models/b.pt"}}]
    plan["models"][0].update(presets=[], model_configs=configs, config="models/cc3.yaml")
    ran = dict(published, stack={"model_configs": ddml_configs.canonical(configs),
                                 "compact": "D"})
    assert fetch.measurements_match(task, ran, plan) == (True, "")
    other = [dict(configs[0], plugin_properties={"ModelPath": "../models/c.pt"})]
    assert fetch.measurements_match(
        task, dict(ran, stack={"model_configs": ddml_configs.canonical(other)}), plan) == (
        False, "the model ran with other DDML model configurations than models/cc3.yaml's")
    assert fetch.measurements_match(
        task, dict(ran, model={"model_configs": other}), plan) == (
        False, "the model's DDML model configurations differ from models/cc3.yaml's")
    # and not by the presets of what it was before
    assert fetch.measurements_match(task, published, plan) == (
        False, "the model ran with the presets P, not the DDML model configurations of "
        "models/cc3.yaml")

    # What the model is, by the sha256 of its description and of its own Python
    # code: as the logs say it was simulated, and the manifest describes it
    own = {"own": {"path": "models/own.py", "sha256": "c" * 64}}
    plan["models"][0].update(description_sha256="d" * 64, python=own)
    ran = dict(ran, stack=dict(ran["stack"], description_sha256="d" * 64,
                               python="own:" + "c" * 64),
               model={"description_sha256": "d" * 64, "python": own})
    assert fetch.measurements_match(task, ran, plan) == (True, "")

    def changed(**stack):
        return fetch.measurements_match(task, dict(ran, stack=dict(ran["stack"], **stack)), plan)
    assert changed(description_sha256="e" * 64) == (
        False, "the model was simulated with another description than models/cc3.yaml (sha256 "
        "eeeeeeeeeeee there, dddddddddddd here)")
    assert changed(python="own:" + "f" * 64) == (
        False, "the model ran other Python code of its own than models/cc3.yaml's "
        "(own:ffffffffffff there, own:cccccccccccc here)")
    assert changed(python="none")[1].startswith("the model ran other Python code of its own")
    # its simulations made with several, as the manifest counts them by log
    assert changed(description_sha256={"d" * 64: 3, "e" * 64: 1}) == (
        False, "they were simulated with 2 versions of the model's description")
    assert changed(python={"own:" + "c" * 64: 3, "none": 1}) == (
        False, "they were simulated with 2 versions of the model's own Python code")
    # or the manifest describing it otherwise
    assert fetch.measurements_match(task, dict(ran, model={"description_sha256": "e" * 64}),
                                    plan) == (
        False, "the model's description differs from models/cc3.yaml")
    assert fetch.measurements_match(task, dict(ran, model={"python": {}}), plan) == (
        False, "the model's own Python code differs from models/cc3.yaml's (none there, "
        "own:cccccccccccc here)")
    # Manifests from before they were recorded don't say, so don't stop it
    old = dict(ran, model={}, stack=dict((key, value) for key, value in ran["stack"].items()
                                         if key not in ("description_sha256", "python")))
    assert fetch.measurements_match(task, old, plan) == (True, "")


def test_published_measurements_of_a_model_described_otherwise_arent_used(bench, published):
    fetching(bench, published)
    path = os.path.join(bench.repo, "models", "caloclouds3_paper.yaml")
    with open(path) as stream:
        text = stream.read()
    assert "\nname: CaloClouds3 (paper)\n" in text and 'color: "#1f6fb4"' in text
    # Its comments, label and plot style don't make it another model...
    with open(path, "w") as stream:
        stream.write("# Said otherwise.\n" + text.replace("\nname: CaloClouds3 (paper)\n",
                                                          "\nname: CC3\n")
                     .replace('color: "#1f6fb4"', 'color: "#000000"'))
    status, output = bench.run("pipeline.sh", "--dry-run")
    assert "would fetch 6 published file(s)" in output, output
    # ... but anything else does, e.g. a setting its own code would read
    with open(path, "w") as stream:
        stream.write(text + "energy_rescale: 1.01\n")
    status, output = bench.run("pipeline.sh", "--dry-run")
    assert "the published caloclouds3_paper.npz (Zenodo record 123) isn't used, as the model " \
        "was simulated with another description than models/caloclouds3_paper.yaml" in output, \
        output
    assert "would fetch 5 published file(s)" in output, output


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


def test_inputs_in_segments_dont_fetch_the_published_whole_ones(bench, published):
    """Once di_photon's inputs are in segments (input.segments), no task makes
    the published whole input, and the published measurements were measured
    on other reconstructions: only single_photon's inputs, which are the same,
    are fetched, and the segments are made here."""
    fetching(bench, published)
    bench.with_segments(3)
    status, output = bench.run("pipeline.sh", "--dry-run")
    assert "would fetch 2 published file(s)" in output, output
    for simulator in ("geant4", "caloclouds3_paper"):
        assert "the published %s.npz (Zenodo record 123) isn't used, as they were measured " \
            "on other reconstructions of di_photon" % simulator in output, output
    status, output = bench.run("pipeline.sh")
    assert "fetching 2 published file(s)" in output, output
    names = bench.files("input_single_particle_gun")
    assert DI_PHOTON_INPUT + ".slcio" not in names
    assert [name for name in names if name.startswith(DI_PHOTON_INPUT)] == sorted(
        "%s_seg%dof3%s" % (DI_PHOTON_INPUT, i, ending) for i in (1, 2, 3)
        for ending in (".h5", ".slcio"))
    for name in INPUTS:
        assert digest(bench.path("input_single_particle_gun", name + ".slcio")) == \
            digest(os.path.join(published["data"], "input_single_particle_gun", name + ".slcio"))


# ---------------------------------------------------------------------------
# The timing benchmark's results and table, which no task makes


def test_a_fresh_data_folder_fetches_the_timing_for_the_pareto_fronts(bench, published_timing):
    fetching(bench, published_timing)
    status, output = bench.run("pipeline.sh", "--dry-run")
    assert "would fetch 6 published file(s)" in output, output
    assert "would fetch the timing benchmark's published results and their table, for the " \
        "Pareto fronts: timing.tar, %d file(s)" % len(published_timing["timing"]) in output, output
    assert not os.path.exists(bench.path("timing"))
    status, output = bench.run("pipeline.sh", "--dry-run", "--no-fetch")
    assert "would fetch" not in output, output
    # Only a run that plots fetches it
    status, output = bench.run("pipeline.sh", "--dry-run", "--stages", "generate")
    assert "would fetch 6 published file(s)" in output, output
    assert "would fetch the timing" not in output, output

    status, output = bench.run("pipeline.sh")
    assert "fetching the timing benchmark's published results" in output, output
    for path in published_timing["timing"]:
        assert digest(bench.path(path)) == digest(os.path.join(published_timing["data"], path))
    hidden = [name for root, _, names in os.walk(bench.path("timing")) for name in names
              if name.startswith(".")]
    assert hidden == []
    # (the archive isn't kept)
    assert os.listdir(bench.path(".ddml-bench", "published", "zenodo-" + TIMED_RECORD)) == [
        "manifest.json"]
    plots = bench.files("observables", "plots")
    for scenario in ("single_photon", "di_photon"):
        assert "pareto_js_cpu__%s.png" % scenario in plots, plots
    assert {row["simulator"] for row in read_metrics(bench.path("observables", "plots",
                                                                "pareto.csv"))} == {
        "geant4", "caloclouds3_paper"}
    status, output = bench.run("pipeline.sh")
    assert "everything selected is done" in output and "fetching" not in output, output

    # A table made here again, from the results fetched and those of a model
    # timed here, is newer than the plots, which are drawn again
    write_timing(bench.data, times=(("another", 0.1),))
    status, output = bench.run("pipeline.sh", "--dry-run")
    assert "the timing table is newer than the plots" in output and "would fetch" not in output, \
        output


def test_plotting_fetches_the_timing_where_there_is_none(bench, published_timing):
    fetching(bench, published_timing)
    measured = os.path.join(published_timing["data"], "observables")
    shutil.copytree(measured, bench.path("observables"),
                    ignore=lambda folder, names: [n for n in names if not n.endswith(".npz")])
    status, output = bench.run("plot_observables.sh", "--format", "png", "--no-dashboard",
                               "--no-fetch")
    assert "there is no timing table" in output and not os.path.exists(bench.path("timing")), \
        output
    status, output = bench.run("plot_observables.sh", "--format", "png", "--no-dashboard")
    assert "fetching the timing benchmark's published results" in output, output
    assert "pareto_js_cpu__single_photon.png" in bench.files("observables", "plots"), output
    assert digest(bench.path("timing", "summary.csv")) == digest(
        os.path.join(published_timing["data"], "timing", "summary.csv"))
    status, output = bench.run("plot_observables.sh", "--format", "png", "--no-dashboard")
    assert "fetching" not in output, output

    # Measurements elsewhere (--dir): into their data folder, where the
    # Pareto fronts look for it
    elsewhere = os.path.join(bench.tmp, "elsewhere", "observables")
    shutil.copytree(bench.path("observables"), elsewhere,
                    ignore=lambda folder, names: [n for n in names if not n.endswith(".npz")])
    status, output = bench.run("plot_observables.sh", "--dir", elsewhere, "--format", "png",
                               "--no-dashboard")
    assert "fetching the timing benchmark's published results" in output, output
    assert os.path.isfile(os.path.join(bench.tmp, "elsewhere", "timing", "summary.csv"))
    assert os.path.isfile(os.path.join(elsewhere, "plots", "pareto.csv"))


def test_the_timing_isnt_fetched_into_a_folder_with_timing_results_of_its_own(
        bench, published_timing):
    fetching(bench, published_timing)
    # The published results, as an interrupted fetch would leave them, without
    # the table: the rest is fetched, as they are the published ones
    result = next(path for path in published_timing["timing"] if path.endswith("_t4.json"))
    os.makedirs(os.path.dirname(bench.path(result)))
    shutil.copy2(os.path.join(published_timing["data"], result), bench.path(result))
    status, output = bench.run("pipeline.sh", "--dry-run")
    assert "would fetch the timing benchmark's published results" in output, output
    # One of its own, or a published one changed: nothing is fetched, and
    # nothing there is replaced
    with open(bench.path(result), "a") as stream:
        stream.write("\n")
    status, output = bench.run("pipeline.sh", "--dry-run")
    assert "the published timing.tar (Zenodo record %s) isn't fetched, as %s has timing results " \
        "of its own (e.g. %s): ./timing_summary.sh makes the timing table from them" % (
            TIMED_RECORD, bench.path("timing"), result) in output, output
    assert "would fetch the timing" not in output, output
    status, output = bench.run("pipeline.sh")
    assert "isn't fetched, as" in output and not os.path.exists(bench.path("timing", "summary.csv"))
    assert open(bench.path(result)).read().endswith("}\n")


def test_a_published_timing_archive_that_isnt_what_its_manifest_says_isnt_used(
        bench, published_timing):
    tampered = os.path.join(bench.tmp, "tampered")
    shutil.copytree(published_timing["zenodo"], tampered)
    with open(os.path.join(tampered, TIMED_RECORD, "timing.tar"), "ab") as stream:
        stream.write(b"more")
    fetching(bench, dict(published_timing, zenodo=tampered))
    # The run goes on, without the Pareto fronts
    status, output = bench.run("pipeline.sh")
    assert "can't fetch the timing benchmark's published results (" in output, output
    assert "timing.tar isn't the published file: its sha256 is" in output, output
    assert "so the plots won't have their Pareto fronts; the next run tries again" in output
    assert "metrics.csv" in bench.files("observables", "plots")
    assert not [n for n in bench.files("observables", "plots") if n.startswith("pareto")]
    assert not os.path.exists(bench.path("timing"))
    assert os.listdir(bench.path(".ddml-bench", "published", "zenodo-" + TIMED_RECORD)) == [
        "manifest.json"]


def test_the_members_of_a_published_timing_archive_are_checked(tmp_path):
    def entry(*paths):
        return {"name": "timing.tar", "members": [{"path": p, "sha256": "0" * 64}
                                                  for p in paths]}
    for path in ("../x", "timing/../x", "/timing/x", "observables/geant4.npz", "timing",
                 "timing//x", "timing/./x", "timing\\x", None):
        with pytest.raises(fetch.FetchError, match="isn't a file of timing/"):
            fetch.timing_members(entry(path))
    with pytest.raises(fetch.FetchError, match="has no sha256 for timing/x"):
        fetch.timing_members({"name": "timing.tar", "members": [{"path": "timing/x"}]})

    # An archive with a link where a member is listed, and a member not
    # listed, which isn't unpacked
    contents = {"timing/summary.csv": b"table\n", "timing/cpu/s/g/a_t4.json": b"{}\n",
                "timing/../outside": b"not listed\n"}
    archive = str(tmp_path / "timing.tar")
    with tarfile.open(archive, "w") as tar:
        for name, data in contents.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        link = tarfile.TarInfo("timing/cpu/s/g/b_t4.json")
        link.type, link.linkname = tarfile.SYMTYPE, "/etc/passwd"
        tar.addfile(link)
    members = [{"path": path, "sha256": hashlib.sha256(contents[path]).hexdigest(),
                "size": len(contents[path])}
               for path in ("timing/summary.csv", "timing/cpu/s/g/a_t4.json")]
    data = tmp_path / "data"
    # The table last, as it says the rest is there
    assert fetch.unpack(archive, members, str(data)) == ["timing/cpu/s/g/a_t4.json",
                                                         "timing/summary.csv"]
    assert (data / "timing" / "summary.csv").read_bytes() == b"table\n"
    assert not (tmp_path / "outside").exists() and not (data / "outside").exists()
    # Nothing there is replaced
    (data / "timing" / "summary.csv").write_text("mine\n")
    os.remove(str(data / "timing" / "cpu" / "s" / "g" / "a_t4.json"))
    assert fetch.unpack(archive, members, str(data)) == ["timing/cpu/s/g/a_t4.json"]
    assert (data / "timing" / "summary.csv").read_text() == "mine\n"

    with pytest.raises(fetch.FetchError, match="timing/cpu/s/g/b_t4.json in timing.tar isn't a "
                                               "file"):
        fetch.unpack(archive, [dict(members[1], path="timing/cpu/s/g/b_t4.json")],
                     str(tmp_path / "linked"))
    # A member that isn't the file listed: neither it nor the table is put in place
    other = tmp_path / "other"
    with pytest.raises(fetch.FetchError, match="timing/cpu/s/g/a_t4.json in timing.tar isn't "
                                               "the published file: its sha256 is"):
        fetch.unpack(archive, [members[0], dict(members[1], sha256="0" * 64)], str(other))
    assert [name for _, _, names in os.walk(str(other)) for name in names] == []
    with pytest.raises(fetch.FetchError, match="isn't the published archive: it hasn't timing/x"):
        fetch.unpack(archive, members + [{"path": "timing/x", "sha256": "0" * 64}],
                     str(tmp_path / "missing"))
    (tmp_path / "not.tar").write_bytes(b"not an archive\n" * 100)
    with pytest.raises(fetch.FetchError, match="not.tar isn't the published archive"):
        fetch.unpack(str(tmp_path / "not.tar"), members, str(tmp_path / "none"))
