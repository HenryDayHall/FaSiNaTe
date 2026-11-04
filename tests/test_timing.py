"""The timing benchmark (ddml_timing/, slurm_timing.sh, timing.sh and
timing_summary.sh): reading ddsim's output event by event and the statistics
of the times; the models' embedded-Python inference timed, and moved to the GPU;
and the workflow end to end, here and as slurm jobs on CPU and GPU nodes, on
the fakes of tests/fakes (ddsim printing its lines event by event, sbatch, and
for the GPU a stand-in for PyTorch, tests/fakes/timing)."""

import csv
import importlib.util
import json
import os
import shutil
import sys
import types

import pytest

from conftest import REPO, _write

# What the copy of the repository needs besides conftest.COPIED
TIMING = ["ddml_timing", "slurm_timing.sh", "timing.sh", "timing_summary.sh"]
FAKE_TIMING = os.path.join(REPO, "tests", "fakes", "timing")
FAKE_TORCH = os.path.join(FAKE_TIMING, "python")
# The inputs of the bench (tests/conftest.py): two single-photon energies of 6
# events, and one di-photon energy of 30, timed with its first seed
SINGLE_PHOTON = ["p22_th43-137_ph79-109_en20_seed0_ip_surface",
                 "p22_th43-137_ph79-109_en80_seed1_ip_surface"]
DI_PHOTON = "diphoton_en20_sep90_x0_z70_r10_seed0"


@pytest.fixture
def timing(bench):
    for name in TIMING:
        source, target = os.path.join(REPO, name), os.path.join(bench.repo, name)
        if os.path.isdir(source):
            shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__"))
        else:
            shutil.copy2(source, target)
    return bench


def control(bench, name, text=""):
    _write(os.path.join(bench.tmp, "fake-control", name), text)


def with_torch(bench, gpu=True, cuda_built=True):
    """The stand-in for PyTorch on the environment's Python path (through the
    fake DDML build's env file), DDML's CaloClouds3 module as it is (using
    it), a GPU if gpu, and the fake ddsim calling the models' inference."""
    env_file = os.path.join(bench.software, "env", "ddml.sh")
    with open(env_file) as stream:
        text = stream.read()
    line = "export PYTHONPATH=%s:$PYTHONPATH\n" % FAKE_TORCH
    if line not in text:
        _write(env_file, text.replace("return 0\n", line + "return 0\n"))
    shutil.copy2(os.path.join(FAKE_TIMING, "cc3_sf_2a_wrapper.py"),
                 os.path.join(bench.ddml, "python", "examples", "cc3_sf_2a_wrapper.py"))
    if gpu:
        control(bench, "cuda_available")
    if not cuda_built:
        control(bench, "cuda_not_built")
    control(bench, "ddsim_run_inference")


def result(bench, device, scenario, simulator, name):
    with open(bench.path("timing", device, scenario, simulator, name + ".json")) as stream:
        return json.load(stream)


def events_csv(bench, device, scenario, simulator, name):
    with open(bench.path("timing", device, scenario, simulator, name + ".events.csv")) as stream:
        return list(csv.DictReader(stream))


# ---------------------------------------------------------------------------
# Reading ddsim's output, and the statistics


def start(number):
    return "GenerationInit   INFO  +++ Initializing event %d. Within run:0 event %d." % (number,
                                                                                         number)


def test_each_event_is_timed_from_its_start_to_the_next():
    from ddml_timing.events import Events
    events = Events()
    events.line("ddml-timing: model cc3_sf_2a_wrapper._model on cuda:0 (Some GPU, PyTorch 2)", 0.1)
    events.line("something before the first event", 0.5)
    lines = []
    for number, (wall, cpu) in enumerate([(1.0, 0.5), (3.0, 2.4), (4.0, 3.4), (6.5, 5.9)], 1):
        lines.append((start(number), wall, cpu))
        lines.append(("ParticleHandler  INFO  +++ Event %d Begin event action." % (number - 1),
                      wall + 0.01, None))
        for _ in range(number):
            lines.append(("ddml-timing: inference cc3_sf_2a_wrapper._model 0.125", wall + 0.1,
                          None))
            lines.append(("PolyhedraBarrelGeometry::localToGlobal INFO  nLayer = 30", wall + 0.1,
                          None))
    for text, wall, cpu in lines:
        events.line(text, wall, cpu)
    # The last event begun has nothing after it to end it
    records = events.records(warmup=1)
    assert [r["event"] for r in records] == [1, 2, 3]
    assert [r["warmup"] for r in records] == [1, 0, 0]
    assert [r["wall_s"] for r in records] == pytest.approx([2.0, 1.0, 2.5])
    assert [r["cpu_s"] for r in records] == pytest.approx([1.9, 1.0, 2.5])
    assert [r["start_s"] for r in records] == pytest.approx([0.0, 2.0, 3.0])
    assert [r["model_calls"] for r in records] == [1, 2, 3]
    assert [r["barrel_showers"] for r in records] == [1, 2, 3]
    assert [r["inference_s"] for r in records] == pytest.approx([0.125, 0.25, 0.375])
    assert events.begun == 4 and not events.input_ended
    assert events.models == [{"model": "cc3_sf_2a_wrapper._model", "device": "cuda:0",
                              "detail": "Some GPU, PyTorch 2"}]

    # An input that runs out: ddsim begins one more event, which ends the last
    events.line(start(5), 8.0, 7.4)
    events.line("UI               INFO  ++ End of file reached, ending run...", 8.01)
    events.line("DDSim            INFO Total Time:   9.81 s (User), 0.09 s (System)", 8.5)
    events.line("DDSim            INFO StartUp Time: 1.55 s, Processing and Init: 7.26 s "
                "(~1.81 s/Event) ", 8.5)
    records = events.records(warmup=1)
    assert [r["wall_s"] for r in records] == pytest.approx([2.0, 1.0, 2.5, 1.5])
    assert events.input_ended and events.begun == 4
    assert events.ddsim == {"user_s": 9.81, "system_s": 0.09, "startup_s": 1.55,
                            "processing_and_init_s": 7.26}


def test_statistics_of_the_times():
    from ddml_timing.events import events_needed, statistics
    found = statistics([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert found["n"] == 6 and found["mean"] == pytest.approx(3.5)
    assert found["sd"] == pytest.approx(1.8708287)
    assert found["sem"] == pytest.approx(1.8708287 / 6 ** 0.5)
    assert found["rel_sem"] == pytest.approx(found["sem"] / 3.5)
    assert found["median"] == pytest.approx(3.5) and found["total"] == pytest.approx(21.0)
    assert (found["first_half_mean"], found["second_half_mean"]) == (2.0, 5.0)
    assert found["drift_z"] == pytest.approx(-3.0 / (2 * (1.0 / 3 ** 0.5) ** 2) ** 0.5)
    assert statistics([2.0])["sd"] is None and statistics([])["mean"] is None
    assert statistics([None, 1.0, 3.0])["n"] == 2
    # A relative uncertainty of 20% with 25 events: 100 events for 10%
    assert events_needed(0.2, 25) == 100 and events_needed(None, 25) is None


# ---------------------------------------------------------------------------
# The models' inference, timed and moved to the GPU


def fake_torch(monkeypatch, tmp_path, gpu=True):
    """The tests' stand-in for PyTorch, as torch, with a GPU if gpu."""
    monkeypatch.setenv("FAKE_CONTROL", str(tmp_path))
    if gpu:
        _write(str(tmp_path / "cuda_available"))
    spec = importlib.util.spec_from_file_location(
        "torch", os.path.join(FAKE_TORCH, "torch", "__init__.py"))
    torch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(torch)
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


def model_module(torch, name="fake_cc3"):
    """A module as DDML's CaloClouds3 module: its network at the top level,
    loaded on the CPU, called with tensors made on the CPU."""
    import numpy as np
    module = types.ModuleType(name)
    module._model = torch.jit.load(__file__)
    module.not_a_model = 3

    def run_inference(inputs):
        out = module._model(*[torch.from_numpy(a) for a in inputs])
        return out.detach().cpu().numpy().astype(np.float32, copy=False).ravel()

    module.run_inference = run_inference
    return module


def test_a_models_inference_runs_on_the_gpu_and_is_timed(monkeypatch, tmp_path, capsys):
    import numpy as np
    from ddml_timing import inference
    torch = fake_torch(monkeypatch, tmp_path)
    module = model_module(torch)
    monkeypatch.setitem(sys.modules, "fake_cc3", module)
    inputs = [np.array([[50.0]], dtype=np.float32), np.array([[10.0]], dtype=np.float32),
              np.array([[90.0]], dtype=np.float32)]
    # As it is, on the CPU
    assert module.run_inference(inputs)[-1] == 50.0

    done = inference.prepare({"FASINATE_TIMED_INFERENCE": "cuda",
                              "FASINATE_TIMED_PYTHON_MODULES": "fake_cc3"})
    assert done == ["fake_cc3._model"]
    assert isinstance(module._model, inference.Timed) and module._model.device == "cuda:0"
    assert module.not_a_model == 3
    # run_inference, unchanged, now runs the network on the GPU and gets the
    # result back on the CPU
    out = module.run_inference(inputs)
    assert out.dtype == np.float32 and out[-1] == 50.0 and out[0] == 1.0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == ("ddml-timing: model fake_cc3._model on cuda:0 (Fake A100-PCIE-80GB, "
                        "PyTorch 2.99.0+fake, CUDA 12.9)")
    assert lines[1].startswith("ddml-timing: inference fake_cc3._model ")
    assert float(lines[1].split()[-1]) >= 0
    # Once only
    assert inference.wrap(module, torch, "cuda:0") == []


def test_a_models_inference_on_the_cpu_is_timed_where_it_is(monkeypatch, tmp_path, capsys):
    import numpy as np
    from ddml_timing import inference
    torch = fake_torch(monkeypatch, tmp_path, gpu=False)
    module = model_module(torch, "fake_cc3_cpu")
    monkeypatch.setitem(sys.modules, "fake_cc3_cpu", module)
    assert inference.prepare({"FASINATE_TIMED_PYTHON_MODULES": "fake_cc3_cpu"}) == \
        ["fake_cc3_cpu._model"]
    assert module._model.device == "cpu"
    module.run_inference([np.array([[5.0]], dtype=np.float32)] * 3)
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "ddml-timing: model fake_cc3_cpu._model on cpu (PyTorch 2.99.0+fake)"
    assert lines[1].startswith("ddml-timing: inference fake_cc3_cpu._model ")


def test_no_gpu_or_no_network_stops_a_gpu_run(monkeypatch, tmp_path):
    from ddml_timing import inference
    torch = fake_torch(monkeypatch, tmp_path, gpu=False)
    monkeypatch.setitem(sys.modules, "fake_cc3_nogpu", model_module(torch, "fake_cc3_nogpu"))
    with pytest.raises(RuntimeError, match="finds no GPU it can use"):
        inference.prepare({"FASINATE_TIMED_INFERENCE": "cuda",
                           "FASINATE_TIMED_PYTHON_MODULES": "fake_cc3_nogpu"})
    _write(str(tmp_path / "cuda_available"))
    empty = types.ModuleType("fake_empty")
    empty.run_inference = lambda inputs: None
    monkeypatch.setitem(sys.modules, "fake_empty", empty)
    with pytest.raises(RuntimeError, match="no torch module at its top level"):
        inference.prepare({"FASINATE_TIMED_INFERENCE": "cuda",
                           "FASINATE_TIMED_PYTHON_MODULES": "fake_empty"})
    assert inference.prepare({"FASINATE_TIMED_INFERENCE": "cpu"}) == []


# ---------------------------------------------------------------------------
# The summary


def write_result(folder, device, scenario, simulator, name, times, inference_on="cpu"):
    from ddml_timing.events import statistics
    from ddml_timing.plan import FORMAT
    path = os.path.join(folder, device, scenario, simulator, name + ".json")
    _write(path, json.dumps({
        "format": FORMAT, "key": "/".join((device, scenario, simulator, name)),
        "device": device, "scenario": scenario, "simulator": simulator,
        "node": {"host": "node1", "cpu_model": "AMD EPYC 7402 24-Core Processor", "gpus": []},
        "inference": {"on": inference_on}, "events": {"requested": len(times), "warmup": 1},
        "wall": statistics(times), "cpu": statistics(times), "model_calls": statistics([1.0]),
        "inference_time": None}))


def test_the_summary_of_the_results(tmp_path):
    from ddml_timing import summary
    folder = str(tmp_path / "timing")
    names = ["p22_en10_seed0_ip_surface_t4", "p22_en100_seed9_ip_surface_t4"]
    write_result(folder, "cpu", "single_photon", "geant4", names[0], [1.0, 1.2, 0.8, 1.0])
    write_result(folder, "cpu", "single_photon", "geant4", names[1], [10.0, 12.0, 8.0, 10.0])
    write_result(folder, "cpu", "single_photon", "caloclouds3", names[0], [0.01] * 4)
    write_result(folder, "cpu", "single_photon", "caloclouds3", names[1], [0.1, 0.12, 0.08, 0.1])
    write_result(folder, "gpu", "single_photon", "caloclouds3", names[1], [0.05] * 4, "cuda")
    _write(os.path.join(folder, "gpu", "single_photon", "caloclouds3", names[0] + ".failed.json"),
           json.dumps({"format": 1, "key": "gpu/single_photon/caloclouds3/" + names[0],
                       "reason": "ddsim exited with status 1\nmore"}))
    printed = []
    assert summary.main(folder, out=printed.append) == 0
    text = "\n".join(printed)
    assert "1 task(s) failed the last time they ran" in text and "status 1" in text
    assert "drifts" not in text
    with open(os.path.join(folder, "summary.csv")) as stream:
        rows = list(csv.DictReader(stream))
    by = dict(((r["device"], r["simulator"], r["input"]), r) for r in rows)
    one = by[("cpu", "caloclouds3", names[1])]
    assert float(one["mean_s"]) == pytest.approx(0.1)
    assert float(one["speedup"]) == pytest.approx(100.0)
    assert one["speedup_reference"] == "geant4 on cpu"
    # The GPU's speed-up is on the CPU nodes' Geant4, as Geant4 isn't timed on the GPU
    assert float(by[("gpu", "caloclouds3", names[1])]["speedup"]) == pytest.approx(200.0)
    # Over all the inputs, each weighing the same
    everything = by[("cpu", "caloclouds3", "all")]
    assert float(everything["mean_s"]) == pytest.approx(0.055)
    assert float(everything["speedup"]) == pytest.approx(5.5 / 0.055)
    assert by[("cpu", "geant4", "all")]["speedup"] == ""
    # Results of another number of events are a sample of their own
    write_result(folder, "cpu", "single_photon", "caloclouds3", "p22_en10_seed0_ip_surface_t2",
                 [0.02, 0.02])
    assert summary.main(folder, out=printed.append) == 0
    with open(os.path.join(folder, "summary.csv")) as stream:
        rows = list(csv.DictReader(stream))
    totals = sorted((r["input"], r["mean_s"]) for r in rows
                    if r["simulator"] == "caloclouds3" and r["device"] == "cpu"
                    and r["input"].startswith("all"))
    assert totals == [("all_t2", "0.02"), ("all_t4", "0.055")]
    # A standard error of 0 needs no more events
    assert by[("cpu", "caloclouds3", names[0])]["events_for_10pct"] == "0"
    assert summary._short_names(names) == {names[0]: "en10_seed0", names[1]: "en100_seed9"}


# ---------------------------------------------------------------------------
# End to end


def test_timing_here_on_the_cpu(timing):
    bench = timing
    bench.run("generate_input.sh")
    status, output = bench.run("timing.sh", "--dry-run", "-n", "3", "--warmup", "1")
    assert "would time 6 task(s)" in output, output
    assert not os.path.exists(bench.path("timing", "cpu"))

    # Geant4 and CaloClouds3 on the 3 inputs
    status, output = bench.run("timing.sh", "-n", "3", "--warmup", "1")
    assert "timed 6 of 6" in output, output
    for simulator in ("geant4", "caloclouds3"):
        assert bench.files("timing", "cpu", "single_photon", simulator) == sorted(
            "%s_t3%s" % (name, ending) for name in SINGLE_PHOTON
            for ending in (".events.csv", ".json", ".log"))
        # di_photon's input with the first of its seeds, 1
        assert bench.files("timing", "cpu", "di_photon", simulator) == sorted(
            DI_PHOTON + "_simseed1_t3" + ending for ending in (".events.csv", ".json", ".log"))
    name = SINGLE_PHOTON[0] + "_t3"
    found = result(bench, "cpu", "single_photon", "caloclouds3", name)
    assert found["wall"]["n"] == 3 and found["wall"]["mean"] > 0
    assert found["cpu"]["n"] == 3
    # 1 of warm-up, 3 timed, and the one more simulated whose start ended them
    assert found["events"] == {"requested": 3, "warmup": 1, "ended": 4, "begun": 5, "timed": 3,
                               "input_ended": False, "timed_fewer": False}
    assert found["model_calls"]["mean"] == 0  # the fake ddsim calls no inference here
    assert found["barrel_showers"]["mean"] == 1
    assert found["inference"]["on"] == "cpu"
    assert found["inference"]["backends"] == [{
        "plugin": "CaloCloudsTwoAngleModelPolyhedraBarrelPyEmbeddedModel/BarrelModelPython",
        "backend": "embedded Python", "python_module": "cc3_sf_2a_wrapper"}]
    assert found["run"]["pinned_cpu"] is not None and found["run"]["line_buffered"]
    assert found["run"]["steering"].endswith(os.path.join("DDML", "scripts", "ddsim_steer.py"))
    assert "--ml-model" in found["run"]["command"]
    assert found["run"]["command"][found["run"]["command"].index("--steeringFile") + 1] \
        .endswith(os.path.join("ddml_timing", "ddsim_steer_timing.py"))
    assert "--numberOfEvents=5" in found["run"]["command"]
    assert found["identity"]["simulator"]["presets"] == ["CC3_BARREL_PY_INTERFACE"]
    assert found["node"]["cpu_model"] and found["software"]["key4hep"] == "2026-04-08"
    rows = events_csv(bench, "cpu", "single_photon", "caloclouds3", name)
    assert [row["warmup"] for row in rows] == ["1", "0", "0", "0"]
    assert [row["event"] for row in rows] == ["1", "2", "3", "4"]
    log = open(bench.path("timing", "cpu", "single_photon", "caloclouds3", name + ".log")).read()
    assert log.startswith("# ddml-timing: caloclouds3 on single_photon, on a cpu node")
    assert "# model     " in log and "+++ Initializing event 4." in log
    assert "# finished" in log
    geant4 = result(bench, "cpu", "single_photon", "geant4", name)
    assert geant4["run"]["steering"].endswith(os.path.join("scripts", "ddsim_steer_geant4.py"))
    assert geant4["inference"]["on"] == "none" and geant4["barrel_showers"]["mean"] == 0

    # Again: done, unless forced, or timed with other settings
    status, output = bench.run("timing.sh", "-n", "3", "--warmup", "1")
    assert "everything selected is done" in output, output
    status, output = bench.run("timing.sh", "-n", "3", "--warmup", "2", "--dry-run", "geant4")
    assert "would time 3 task(s)" in output and "timed with other settings (warmup_events)" \
        in output, output
    status, output = bench.run("timing.sh", "-n", "3", "--warmup", "1", "-f", "-s",
                               "di_photon", "geant4")
    assert "timed 1 of 1" in output, output

    status, output = bench.run("timing_summary.sh")
    assert "cpu nodes, single_photon" in output and "cpu nodes, di_photon" in output, output
    with open(bench.path("timing", "summary.csv")) as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 6 + 4  # each result, and each simulator over each scenario's inputs
    assert all(row["speedup"] for row in rows if row["simulator"] == "caloclouds3")


def test_an_input_that_runs_out_is_timed_to_its_end(timing):
    bench = timing
    bench.run("generate_input.sh", "single_photon")
    status, output = bench.run("timing.sh", "-n", "10", "--warmup", "2", "-s", "single_photon",
                               "geant4", "--file",
                               bench.path("input_single_particle_gun", SINGLE_PHOTON[1] + ".slcio"))
    assert "timed 1 of 1" in output, output
    found = result(bench, "cpu", "single_photon", "geant4", SINGLE_PHOTON[1] + "_t10")
    # All 6 events are ended (the last by the start of one that finds no event)
    assert found["events"] == {"requested": 10, "warmup": 2, "ended": 6, "begun": 6, "timed": 4,
                               "input_ended": True, "timed_fewer": True}


def test_a_scenario_without_its_inputs_is_blocked(timing):
    bench = timing
    status, output = bench.run("timing.sh", "-n", "3", check=False)
    assert status == 1 and "its input isn't there" in output, output
    assert "nothing to time, and some tasks are blocked" in output, output


def test_slurm_timing_on_cpu_and_gpu_nodes(timing):
    bench = timing
    # Besides caloclouds3, a model DDML runs with LibTorch, and one that runs
    # its own inference code (models/caloclouds3_rescaled.py)
    bench.add_model("caloclouds3_paper")
    bench.add_model("caloclouds3_rescaled")
    with_torch(bench)
    bench.run("generate_input.sh")
    status, output = bench.run("slurm_timing.sh", "--dry-run", "-n", "2", "--warmup", "1")
    # cpu: geant4 and the 3 models on 3 inputs; gpu: caloclouds3 and
    # caloclouds3_rescaled (caloclouds3_paper runs with LibTorch, so is skipped)
    assert "would submit 18 job(s)" in output, output
    assert "gpu: caloclouds3_paper (3 tasks, di_photon, single_photon): DDML runs its inference " \
        "with LibTorch, on the CPU only" in output, output
    assert bench.slurm_jobs() == []

    status, output = bench.run("slurm_timing.sh", "-n", "2", "--warmup", "1", "--",
                               "--account=test")
    assert "submitting 18 job(s)" in output, output
    jobs = bench.slurm_jobs()
    assert [job["state"] for job in jobs] == ["COMPLETED"] * 18, [
        (job["args"], open(job["output"]).read()[-2000:]) for job in jobs
        if job["state"] != "COMPLETED"]
    devices = dict((job["args"][0], job) for job in jobs)
    cpu = [job for key, job in devices.items() if key.startswith("cpu/")]
    gpu = [job for key, job in devices.items() if key.startswith("gpu/")]
    assert len(cpu) == 12 and len(gpu) == 6
    for job in cpu:
        for option in ("--partition=maxcpu", "--constraint=EPYC&7402&512G", "--time=04:00:00",
                       "--exclusive", "--nodes=1", "--account=test"):
            assert option in job["options"], job["options"]
    for job in gpu:
        for option in ("--partition=maxgpu", "--constraint=A100-PCIE-80GB&GPUx1", "--exclusive",
                       "--job-name=fasinate-timing-gpu-" + job["args"][0].split("/")[2]):
            assert option in job["options"], job["options"]
        assert "with status 0" in open(job["output"]).read()

    name = SINGLE_PHOTON[1] + "_t2"
    on_gpu = result(bench, "gpu", "single_photon", "caloclouds3", name)
    assert on_gpu["inference"]["on"] == "cuda"
    assert on_gpu["inference"]["models"] == [{
        "model": "cc3_sf_2a_wrapper._model", "device": "cuda:0",
        "detail": "Fake A100-PCIE-80GB, PyTorch 2.99.0+fake, CUDA 12.9"}]
    # The fake ddsim calls the model once per event, as DDML does per shower
    assert on_gpu["model_calls"]["mean"] == 1 and on_gpu["inference_time"]["n"] == 2
    assert on_gpu["node"]["job"]["slurm_job_id"]
    # Its own inference code, imported from models/ as scripts/ddsim_steer_models.py has it
    rescaled = result(bench, "gpu", "single_photon", "caloclouds3_rescaled", name)
    assert [m["model"] for m in rescaled["inference"]["models"]] == \
        ["caloclouds3_rescaled._model"] and rescaled["inference"]["models"][0]["device"] == "cuda:0"
    assert rescaled["run"]["steering"].endswith(os.path.join("scripts", "ddsim_steer_models.py"))
    assert rescaled["model_calls"]["mean"] == 1
    assert rescaled["identity"]["simulator"]["python"]["caloclouds3_rescaled"]
    on_cpu = result(bench, "cpu", "single_photon", "caloclouds3", name)
    assert on_cpu["inference"]["models"][0]["device"] == "cpu"
    assert on_cpu["model_calls"]["mean"] == 1
    libtorch = result(bench, "cpu", "single_photon", "caloclouds3_paper", name)
    assert libtorch["inference"]["models"] == [] and libtorch["model_calls"]["mean"] == 0
    assert not os.path.exists(bench.path("timing", "gpu", "single_photon", "caloclouds3_paper"))
    submission, = [d for d in os.listdir(bench.path("timing", "slurm"))
                   if d.startswith("timing_")]
    script = open(bench.path("timing", "slurm", submission, "job.sh")).read()
    assert "    gpu/*)\n        if command -v apptainer >/dev/null; then export APPTAINER_NV=1" \
        in script, script

    # Again: nothing to submit; the summary has every result
    status, output = bench.run("slurm_timing.sh", "-n", "2", "--warmup", "1")
    assert "everything selected is timed, so there is nothing to submit" in output, output
    status, output = bench.run("timing_summary.sh")
    assert "gpu nodes, single_photon" in output, output


def test_gpu_timing_needs_pytorch_with_cuda(timing):
    bench = timing
    with_torch(bench, cuda_built=False)
    bench.run("generate_input.sh")
    status, output = bench.run("slurm_timing.sh", "-n", "2", "--warmup", "1")
    # The CPU nodes' jobs go; the GPU nodes' are blocked, saying why
    assert "submitting 6 job(s)" in output, output
    assert "built without CUDA" in output, output
    assert all(job["args"][0].startswith("cpu/") for job in bench.slurm_jobs())


def test_a_gpu_job_without_a_gpu_fails_saying_why(timing):
    bench = timing
    with_torch(bench, gpu=False)
    bench.run("generate_input.sh", "single_photon")
    status, output = bench.run("timing.sh", "-d", "gpu", "-n", "2", "--warmup", "1", check=False)
    assert status == 1 and "FAILED" in output and "finds no GPU it can use" in output, output
    failed = bench.path("timing", "gpu", "single_photon", "caloclouds3",
                        SINGLE_PHOTON[0] + "_t2.failed.json")
    assert os.path.isfile(failed)
    status, output = bench.run("timing_summary.sh", check=False)
    assert "failed the last time they ran" in output, output
    status, output = bench.run("timing.sh", "-d", "gpu", "-n", "2", "--warmup", "1", "--dry-run",
                               check=False)
    assert "failed last time: ddsim exited with status" in output, output
