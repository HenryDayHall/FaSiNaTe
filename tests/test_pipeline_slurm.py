"""slurm_pipeline.sh end to end, with the fake sbatch and squeue of tests/fakes:
the jobs, their dependencies, resubmission, waiting for the queued jobs of
earlier submissions, and cancelling what needs a job that failed."""

import glob
import os

from test_pipeline_local import INPUTS, SIMULATORS


def submissions(bench):
    return sorted(glob.glob(bench.path("slurm", "pipeline_*")))


def records(folder):
    with open(os.path.join(folder, "jobs.tsv")) as stream:
        return [line.split("\t") for line in stream.read().splitlines()[1:]]


def by_stage(jobs):
    found = {}
    for job in jobs:
        stage = job["args"][0].split(":", 1)[0]
        found.setdefault(stage, []).append(job)
    return found


def test_slurm_pipeline_runs_every_stage_as_jobs(bench):
    status, output = bench.run("slurm_pipeline.sh", "--dry-run")
    assert "would submit 13 job(s)" in output, output
    assert submissions(bench) == [] and bench.slurm_jobs() == []

    status, output = bench.run("slurm_pipeline.sh", "--", "--account=test")
    assert "submitting 13 job(s)" in output, output
    jobs = bench.slurm_jobs()
    assert [job["state"] for job in jobs] == ["COMPLETED"] * 13, jobs
    folder, = submissions(bench)
    assert len(records(folder)) == 13
    for name in ("plan.json", "tasks.json", "job.sh", "report.txt"):
        assert os.path.isfile(os.path.join(folder, name))
    for job in jobs:
        assert "with status 0" in open(job["output"]).read()

    # Each job waits for the jobs making what it reads
    ids = dict((job["args"][0], job["id"]) for job in jobs)
    stages = by_stage(jobs)
    for job in stages["reconstruct"]:
        assert len(job["after"]) == 1 and job["after"][0] in [
            j["id"] for j in stages["geant4"] + stages["models"]]
    for job in stages["observables"]:
        assert len(job["after"]) == 2
    plot, = stages["plot"]
    assert sorted(plot["after"]) == sorted(j["id"] for j in stages["observables"])
    geant4 = stages["geant4"][0]
    for option in ("--partition=maxcpu", "--constraint=EPYC&7402", "--time=2-00:00:00",
                   "--nodes=1", "--account=test", "--job-name=fasinate-geant4"):
        assert option in geant4["options"], geant4["options"]
    assert ids  # every job ran a task of tasks.json

    for simulator in SIMULATORS:
        assert len(bench.files("rec_output", "single_photon", simulator)) == 8
    assert "metrics.csv" in bench.files("observables", "plots")

    # Run again: nothing to submit, and no folder for that
    status, output = bench.run("slurm_pipeline.sh")
    assert "everything selected is done, so there is nothing to submit" in output, output
    assert len(submissions(bench)) == 1


def test_slurm_pipeline_waits_for_queued_jobs(bench):
    queue = {"FAKE_SLURM_MODE": "queue"}
    status, output = bench.run("slurm_pipeline.sh", env=queue)
    assert [job["state"] for job in bench.slurm_jobs()] == ["PENDING"] * 13

    # The jobs are still queued, so they are waited for, not submitted again
    status, output = bench.run("slurm_pipeline.sh", env=queue)
    assert "nothing new to submit; 13 job(s) of earlier submissions are still queued" in output, \
        output
    assert len(bench.slurm_jobs()) == 13

    bench.run_fake("sbatch", "--drain")
    assert [job["state"] for job in bench.slurm_jobs()] == ["COMPLETED"] * 13
    status, output = bench.run("slurm_pipeline.sh")
    assert "everything selected is done" in output, output


def test_slurm_pipeline_cancels_what_needs_a_failed_job(bench):
    bench.fail_ddsim("geant4 *en80*")
    status, output = bench.run("slurm_pipeline.sh")
    states = dict((job["args"][0], job["state"]) for job in bench.slurm_jobs())
    failed = [key for key, state in states.items() if state == "FAILED"]
    cancelled = sorted(key.split(":", 1)[0] for key, state in states.items()
                       if state == "CANCELLED")
    assert len(failed) == 1 and failed[0].startswith("geant4:") and "en80" in failed[0]
    assert cancelled == ["observables", "plot", "reconstruct"], states
    assert list(states.values()).count("COMPLETED") == 9

    # Run again: only what failed, or was cancelled, is submitted
    bench.fail_ddsim(None)
    status, output = bench.run("slurm_pipeline.sh")
    assert "submitting 4 job(s)" in output, output
    assert [job["state"] for job in bench.slurm_jobs()[13:]] == ["COMPLETED"] * 4
    assert "metrics.csv" in bench.files("observables", "plots")


def test_slurm_pipeline_test_sample(bench):
    status, output = bench.run("slurm_pipeline.sh", "-n", "2", "geant4")
    assert "submitting 8 job(s)" in output, output
    assert bench.files("rec_output", "single_photon", "geant4") == sorted(
        name + "_n2" + ending for name in INPUTS
        for ending in ("_AIDA.root", "_PfoAnalysis.root", "_REC.edm4hep.root", ".log"))
    assert bench.files("observables") == ["geant4_n2.npz", "plots"]
    assert "metrics_n2.csv" in bench.files("observables", "plots")


def test_slurm_simulate_geant4_submits_the_geant4_stage(bench):
    bench.run("generate_input.sh")
    status, output = bench.run("slurm_simulate_geant4.sh", "-s", "single_photon")
    assert "submitting 2 job(s)" in output, output
    assert [job["options"][0] for job in bench.slurm_jobs()] == ["--job-name=fasinate-geant4"] * 2
    status, output = bench.run("slurm_simulate_geant4.sh", "-m", "4", check=False)
    assert status == 1 and "is gone" in output
