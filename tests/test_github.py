"""The low-cost mode on GitHub's runners (ddml_bench/github.py,
.github/workflows/low-cost.yml): how chains are packed into shards, the
settings, the workflow itself, and its jobs end to end on the fakes, against a
record published from a run of the pipeline (as tests/test_fetch.py publishes
one). Each job runs in a copy of the repository of its own, with a data
folder of its own, as on a runner of its own, and gets from the jobs before
it only what the workflow passes on as artifacts: the plan, the published
data, the measurements of the chains and the samples kept. What the report
merges is what a low-cost run measures in one go, array for array, and its
plots and metrics are the same."""

import json
import os
import shutil
import time

import numpy as np
import pytest
import yaml

from conftest import REPO, Bench
from ddml_bench import github
from ddml_bench.config import ConfigError
from test_fetch import fetching, publish
from test_low_cost import another_model, pinned, read_csv

SETTINGS = {"parallel": 2, "speed": 1.0, "shard_hours": 4, "max_shards": 10,
            "deadline_hours": 4.5, "stop_hours": 5.5, "setup_hours": 0.25, "budget_hours": 40,
            "min_free_gb": 0}


def chains(*hours):
    return [{"id": "%03d" % number, "hours": value} for number, value in enumerate(hours)]


def test_chains_are_packed_into_as_few_shards_as_keep_each_under_its_hours():
    # As a CaloClouds3-like model's 46 chains of 1000 events: 9 of tau pairs,
    # and 37 shorter ones
    found = github.pack(chains(*([2.07] * 9 + [0.2] * 37)), SETTINGS)
    # 26.0 h at 2 chains at once is 4 shards of 3.25 h, but 9 tau chains don't
    # go into 8 places for a chain without one of them taking 2 of them
    assert len(found) == 5
    assert all(shard["hours"] <= SETTINGS["shard_hours"] for shard in found)
    assert sorted(i for shard in found for i in shard["chains"]) == ["%03d" % i for i in range(46)]
    # each shard has 2 places for a chain at a time, so 10 for the 9 tau chains
    assert sorted(sum(1 for i in shard["chains"] if int(i) < 9) for shard in found) == \
        [1, 2, 2, 2, 2]
    assert github.pack([], SETTINGS) == []


def test_shards_are_as_long_as_the_longest_chain_if_that_is_longer():
    # No number of shards makes the run shorter than its longest chain, so the
    # rest go into as few shards as take no longer
    found = github.pack(chains(5.0, 1.0, 1.0, 1.0), dict(SETTINGS, parallel=1, shard_hours=2))
    assert found == [{"chains": ["000"], "hours": 5.0},
                     {"chains": ["001", "002", "003"], "hours": 3.0}]
    # and at most max_shards
    found = github.pack(chains(*[1.0] * 40), dict(SETTINGS, max_shards=3))
    assert len(found) == 3 and max(shard["hours"] for shard in found) == 7.0


def test_the_github_settings_are_checked():
    assert github.github_settings({"github": dict(SETTINGS)}) == SETTINGS
    for key, value in (("parallel", 1.5), ("parallel", 0), ("speed", "fast"),
                       ("budget_hours", -1), ("min_free_gb", True), ("max_shards", None)):
        with pytest.raises(ConfigError, match="github.%s" % key):
            github.github_settings({"github": dict(SETTINGS, **{key: value})})
    assert github.github_settings({"github": dict(SETTINGS, min_free_gb=0)})["min_free_gb"] == 0
    with pytest.raises(ConfigError, match="stop_hours"):
        github.github_settings({"github": dict(SETTINGS, stop_hours=4)})


def test_the_workflow_runs_the_commands_in_the_container_it_names():
    with open(os.path.join(REPO, ".github", "workflows", "low-cost.yml")) as stream:
        workflow = yaml.safe_load(stream)
    jobs = workflow["jobs"]
    image = workflow["env"]["FASINATE_GITHUB_IMAGE"]
    assert workflow["env"]["FASINATE_CONFIG"] == "configs/github.yaml"
    # The jobs with the software in DDML CI's container (its key has the image)
    for name in ("software", "shard"):
        assert jobs[name]["container"] == {"image": image, "options": "--privileged"}, name
        steps = jobs[name]["steps"]
        assert any(step.get("run") == "/mount.sh" for step in steps), name
        # The software's cache is the same everywhere
        caches = [step["with"] for step in steps if step.get("uses", "").startswith("actions/cache")]
        assert caches and all(cache["path"] == caches[0]["path"]
                              and cache["key"] == "${{ needs.plan.outputs.software-key }}"
                              for cache in caches), name
    commands = " ".join(step.get("run", "") for job in jobs.values() for step in job["steps"])
    for command in ("github keys", "github plan", "github check", "github shard", "github report"):
        assert "python3 scripts/ddml-bench " + command in commands, command
    assert "models/**" in workflow[True]["push"]["paths"]


# ---------------------------------------------------------------------------
# End to end, on the fakes


@pytest.fixture(scope="module")
def published(tmp_path_factory):
    """A record of the essential files of a run of the pipeline."""
    publisher = Bench(str(tmp_path_factory.mktemp("publisher")))
    publisher.run("pipeline.sh")
    return publish(publisher, "321")


# Each chain alone on its runner, one at a time, so that the 4 chains of the
# tests' sample are 4 shards
SHARDS = "github:\n  parallel: 1\n  shard_hours: 0.1\n  min_free_gb: 0\n"


class Runner(object):
    """A runner of the workflow's: a copy of the repository of its own, with
    its own data folder, configured to fetch from the published record, with
    the software it was made with pinned, the model 'another' (CaloClouds3
    (paper) by another name) and the settings of SHARDS and settings; and the
    folder of the run, its outputs and its summary, as GitHub has them."""

    def __init__(self, tmp, published, name, settings="", artifacts=None):
        self.bench = bench = Bench(os.path.join(str(tmp), name))
        fetching(bench, published)
        # The Key4hep release the same on every runner, as on GitHub's (the
        # tests' has a path of each bench's own), and in the build's stamp
        with open(os.path.join(bench.tmp, "key4hep.yaml"), "w") as stream:
            stream.write("key4hep:\n  setup_script: /cvmfs/sw.hsf.org/key4hep/setup.sh\n")
        bench.env["FASINATE_CONFIG"] += ":" + os.path.join(bench.tmp, "key4hep.yaml")
        pinned(bench, published)
        another_model(bench)
        override = os.path.join(bench.tmp, "github.yaml")
        with open(override, "w") as stream:
            stream.write(SHARDS + settings)
        bench.env["FASINATE_CONFIG"] += ":" + override
        self.work = os.path.join(bench.tmp, "low_cost_run")
        self.output = os.path.join(bench.tmp, "github_output")
        self.summary = os.path.join(bench.tmp, "github_summary")
        bench.env.update(GITHUB_OUTPUT=self.output, GITHUB_STEP_SUMMARY=self.summary,
                         GITHUB_REPOSITORY="someone/FaSiNaTe", GITHUB_RUN_ID="77",
                         FAKE_GH_DIR=artifacts or os.path.join(str(tmp), "artifacts"))

    def run(self, *args, **options):
        return self.bench.run("scripts/ddml-bench", "github", "--dir", self.work, *args,
                              **options)

    def outputs(self):
        found = {}
        if os.path.isfile(self.output):
            for line in open(self.output).read().splitlines():
                name, _, value = line.partition("=")
                found[name] = value
        return found

    def take(self, source, *folders):
        """What another job passed on, as artifacts: its folders (of its run's
        folder, or its data folder's published files)."""
        for folder in folders:
            if folder == "data":
                for name in os.listdir(source.bench.data):
                    if not name.startswith(".") and name != "timing":
                        shutil.copytree(os.path.join(source.bench.data, name),
                                        os.path.join(self.bench.data, name),
                                        dirs_exist_ok=True)
            elif os.path.isdir(os.path.join(source.work, folder)):
                shutil.copytree(os.path.join(source.work, folder),
                                os.path.join(self.work, folder), dirs_exist_ok=True)
            elif os.path.isfile(os.path.join(source.work, folder)):
                os.makedirs(self.work, exist_ok=True)
                shutil.copy2(os.path.join(source.work, folder), self.work)


def big_files(bench):
    """The simulations and reconstructions in the bench's data folder."""
    found = []
    for folder in ("sim_output", "rec_output"):
        for root, _, names in os.walk(bench.path(folder)):
            found += [name for name in names if name.endswith((".root", ".slcio"))]
    return found


def assert_same_measurements(path, other):
    with np.load(path) as one, np.load(other) as two:
        assert sorted(one.files) == sorted(two.files)
        for key in one.files:
            assert one[key].dtype == two[key].dtype, key
            np.testing.assert_array_equal(one[key], two[key], err_msg=key)


def run_shards(tmp, published, plan, shards, name="shards", settings=""):
    """The shard jobs, all on one runner (one after another, each from the
    plan alone); returns the runner and the output of each."""
    runner = Runner(tmp, published, name, settings)
    runner.take(plan, "plan.json", "data")
    outputs = []
    for index in shards:
        status, output = runner.run("shard", str(index), check=False)
        outputs.append((status, output))
    return runner, outputs


def test_a_model_is_tested_on_runners_of_their_own_as_a_low_cost_run_tests_it(
        tmp_path, published):
    plan = Runner(tmp_path, published, "plan")
    status, output = plan.run("plan", "--events", "3")
    outputs = plan.outputs()
    # The model without published measurements; 2 single-photon inputs, and a
    # di-photon input simulated with 2 seeds: 4 chains, one per shard
    assert outputs["models"] == "another" and outputs["report"] == "true", output
    assert outputs["chains"] == "4" and json.loads(outputs["shards"]) == [0, 1, 2, 3], output
    record = json.load(open(os.path.join(plan.work, "plan.json")))
    assert record["why"] == {"another": "no another.npz yet"}
    assert record["suffix"] == "_n3" and record["reused"] == {}
    assert record["compared"] == ["geant4", "geant4_2", "caloclouds3_paper"]
    assert [chain["scenario"] for chain in record["chains"]] == \
        ["single_photon"] * 2 + ["di_photon"] * 2
    assert "no sample kept by an earlier run" not in output
    summary = open(plan.summary).read()
    assert "## Low-cost plan" in summary and "`another`" in summary and "4 shard" in summary
    # (nothing kept yet: gh was asked, and found nothing)
    calls = open(os.path.join(str(tmp_path), "artifacts", "calls.log")).read()
    assert "name=" + record["artifacts"]["another"] in calls

    # The software job: everything the chains need is there
    software = Runner(tmp_path, published, "software")
    software.take(plan, "plan.json", "data")
    status, output = software.run("check")
    assert "this environment can run the 4 chain(s) of the plan" in output, output

    # The shards, on runners of their own; only their measurements and logs
    # are passed on, as the simulations and reconstructions are deleted
    first, done = run_shards(tmp_path, published, plan, [0, 1], "shards-a")
    second, more = run_shards(tmp_path, published, plan, [2, 3], "shards-b")
    for status, output in done + more:
        assert status == 0 and "1 of 1 chain(s) done" in output, output
    parts = {}
    for runner in (first, second):
        assert big_files(runner.bench) == []
        logs = os.listdir(os.path.join(runner.work, "logs"))
        for name in os.listdir(os.path.join(runner.work, "parts")):
            parts[name] = os.path.join(runner.work, "parts", name)
            chain = name.split(".")[0]
            assert {chain + ".log", chain + ".simulation.log",
                    chain + ".reconstruction.log"} <= set(logs), logs
    assert sorted(parts) == ["000.json", "000.npz", "001.json", "001.npz", "002.json",
                             "002.npz", "003.json", "003.npz"]
    kept = json.load(open(parts["003.json"]))
    assert kept["status"] == "done" and kept["events"] == 3 and kept["scenario"] == "di_photon"
    # A shard run again (a job re-run) finds its chains done
    status, output = first.run("shard", "0")
    assert "1 of the shard's 1 chain(s) are done already" in output, output
    assert big_files(first.bench) == []

    # The report, on a runner of its own
    report = Runner(tmp_path, published, "report")
    report.take(plan, "plan.json", "data")
    for runner in (first, second):
        report.take(runner, "parts", "logs")
    status, output = report.run("report")
    merged = report.bench.path("observables", "another_n3.npz")
    results = os.path.join(report.work, "results")
    assert {"metrics.csv", "fits.csv", "dashboard.html", "summary.md", "run.json",
            "radial_profile_r100__single_photon.png"} <= set(os.listdir(results))
    assert os.listdir(os.path.join(results, "samples")) == ["another_n3.npz"]
    # No Pareto fronts
    assert not any(name.startswith("pareto") for name in os.listdir(results))
    keep = json.loads(report.outputs()["keep"])
    assert keep == [record["artifacts"]["another"]]
    assert sorted(os.listdir(os.path.join(report.work, "keep", keep[0]))) == [
        "another_n3.npz", "sample.json"]
    summary = open(report.summary).read()
    assert "## Low-cost test of Another (`another`)" in summary, summary
    assert "| radial_profile r100, single_photon | JS |" in summary, summary
    assert "caloclouds3_paper: " in summary and "the runners ran at" in summary, summary

    # The same as a low-cost run measures and draws, in one go, on the fakes
    alone = Runner(tmp_path, published, "alone")
    status, output = alone.bench.run("low_cost.sh", "another", "-n", "3")
    assert "made 10 of 10" in output, output
    assert_same_measurements(merged, alone.bench.path("observables", "another_n3.npz"))
    plots = alone.bench.path("observables", "plots", "low_cost_n3")
    for name in ("metrics.csv", "fits.csv"):
        assert read_csv(os.path.join(results, name)) == read_csv(os.path.join(plots, name)), name


def test_a_sample_kept_by_an_earlier_run_is_taken_rather_than_made_again(tmp_path, published):
    plan = Runner(tmp_path, published, "plan")
    plan.run("plan", "--events", "3")
    runner, outputs = run_shards(tmp_path, published, plan, range(4))
    assert all(status == 0 for status, _ in outputs), outputs
    report = Runner(tmp_path, published, "report")
    report.take(plan, "plan.json", "data")
    report.take(runner, "parts")
    report.run("report")
    # The keep job keeps it as an artifact of the run
    artifact = json.loads(report.outputs()["keep"])[0]
    stored = os.path.join(str(tmp_path), "artifacts", artifact, "77")
    shutil.copytree(os.path.join(report.work, "keep", artifact), stored)

    # A later run of the same model, software and settings takes it
    later = Runner(tmp_path, published, "later")
    status, output = later.run("plan", "--events", "3")
    assert "another: its sample was made by run 77" in output, output
    assert later.outputs()["chains"] == "0" and later.outputs()["report"] == "true"
    again = Runner(tmp_path, published, "report-again")
    again.take(later, "plan.json", "samples", "data")
    status, output = again.run("report")
    assert_same_measurements(again.bench.path("observables", "another_n3.npz"),
                             report.bench.path("observables", "another_n3.npz"))
    assert json.loads(again.outputs()["keep"]) == []
    assert "made by an earlier run, https://github.com/someone/FaSiNaTe/actions/runs/77" in \
        open(again.summary).read()
    # A new label is only drawn otherwise, but a change to what the model is
    # makes it again, as does --fresh
    status, output = later.run("plan", "--events", "3", "--fresh")
    assert later.outputs()["chains"] == "4", output
    path = os.path.join(later.bench.repo, "models", "another.yaml")
    text = open(path).read()
    with open(path, "w") as stream:
        stream.write(text.replace("name: Another", "name: Yet another"))
    later.run("plan", "--events", "3")
    assert later.outputs()["chains"] == "0"
    with open(path, "w") as stream:
        stream.write(text.replace("training_range: [1, 127]", "training_range: [1, 126]"))
    assert "training_range: [1, 126]" in open(path).read()
    status, output = later.run("plan", "--events", "3")
    assert later.outputs()["chains"] == "4" and "its sample was made by" not in output, output


def test_a_failed_chain_is_made_by_the_job_run_again(tmp_path, published):
    plan = Runner(tmp_path, published, "plan", "  parallel: 2\n  shard_hours: 4\n")
    plan.run("plan", "--events", "3")
    assert json.loads(plan.outputs()["shards"]) == [0]
    runner = Runner(tmp_path, published, "shard", "  parallel: 2\n")
    runner.take(plan, "plan.json", "data")
    runner.bench.fail_ddsim("* *en80*")
    status, output = runner.run("shard", "0", check=False)
    assert status == 1 and "3 of 4 chain(s) done" in output, output
    assert "FAILED (exit status 1)" in output and "the end of chain 001's log" in output, output
    report = Runner(tmp_path, published, "report")
    report.take(plan, "plan.json", "data")
    report.take(runner, "parts")
    status, output = report.run("report", check=False)
    assert status == 1 and "another isn't drawn: 1 of its 4 chain(s) have no measurement" \
        in output, output
    assert "**another isn't drawn:** 1 of its chains" in open(report.summary).read()
    assert not os.path.exists(report.bench.path("observables", "another_n3.npz"))

    # Run again, it makes only the chain that failed
    runner.bench.fail_ddsim(None)
    status, output = runner.run("shard", "0")
    assert "3 of the shard's 4 chain(s) are done already" in output, output
    assert "1 of 1 chain(s) done" in output, output
    report.take(runner, "parts")
    status, output = report.run("report")
    assert os.path.isfile(report.bench.path("observables", "another_n3.npz"))


def test_a_shard_starts_no_chain_after_its_deadline_and_stops_them_at_its_end(
        tmp_path, published):
    plan = Runner(tmp_path, published, "plan")
    plan.run("plan", "--events", "3")
    # Past its deadline from the start of its job, nothing starts
    runner = Runner(tmp_path, published, "late", "  deadline_hours: 1\n  stop_hours: 3\n")
    runner.take(plan, "plan.json", "data")
    runner.bench.env[github.JOB_START] = str(time.time() - 7200)
    status, output = runner.run("shard", "0", check=False)
    assert status == 1 and "NOT STARTED, past github.deadline_hours" in output, output
    # A chain still running at its end is stopped, and leaves nothing
    runner.bench.env.pop(github.JOB_START)
    control = os.path.join(runner.bench.tmp, "fake-control")
    os.makedirs(control, exist_ok=True)
    with open(os.path.join(control, "ddsim_event_seconds"), "w") as stream:
        stream.write("60\n")
    override = os.path.join(runner.bench.tmp, "github.yaml")
    with open(override, "w") as stream:
        stream.write(SHARDS + "  deadline_hours: 0.005\n  stop_hours: 0.01\n")
    status, output = runner.run("shard", "0", check=False)
    assert status == 1 and "STOPPED at github.stop_hours" in output, output
    assert not os.path.exists(os.path.join(runner.work, "parts", "000.npz"))
    assert big_files(runner.bench) == []


def test_a_run_over_budget_is_refused(tmp_path, published):
    plan = Runner(tmp_path, published, "plan", "  budget_hours: 0.5\n")
    status, output = plan.run("plan", "--events", "3", check=False)
    assert status == 1 and "more than github.budget_hours (0.5)" in output, output
    assert "**Stopped:**" in open(plan.summary).read()
    assert "shards" not in plan.outputs()


def test_nothing_is_tested_where_every_model_is_published(tmp_path, published):
    plan = Runner(tmp_path, published, "plan")
    os.remove(os.path.join(plan.bench.repo, "models", "another.yaml"))
    status, output = plan.run("plan")
    assert "nothing to test" in output, output
    assert plan.outputs()["report"] == "false" and plan.outputs()["chains"] == "0"
    # unless named
    status, output = plan.run("plan", "--models", "caloclouds3_paper", "--events", "2")
    assert plan.outputs()["models"] == "caloclouds3_paper" and plan.outputs()["chains"] == "4"


def test_the_caches_keys_follow_what_is_in_them(tmp_path, published):
    runner = Runner(tmp_path, published, "keys")
    runner.run("keys")
    first = runner.outputs()
    override = os.path.join(runner.bench.tmp, "ddml.yaml")
    with open(override, "w") as stream:
        stream.write('ddml:\n  commit: "%s"\n' % ("f" * 40))
    runner.bench.env["FASINATE_CONFIG"] += ":" + override
    os.remove(runner.output)
    runner.run("keys")
    second = runner.outputs()
    assert second["published-key"] == first["published-key"]
    assert second["software-key"] != first["software-key"]
