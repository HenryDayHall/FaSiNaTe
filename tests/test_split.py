"""Splitting what was made from whole inputs into the segments of the inputs
(input.segments; scripts/split_data.py, ddml_bench/split.py), end to end on
the fakes: the pipeline run with whole inputs, then their scenarios put in
segments, and the data split rather than made again."""

import csv
import json
import os
import subprocess
import sys

import awkward as ak
import numpy as np
import pytest
import uproot

from conftest import REPO
from ddml_bench import manifest

DI_PHOTON = "diphoton_en20_sep90_x0_z70_r10_seed0"
SEGMENTS = [(0, 10), (10, 20), (20, 30)]  # of its 30 pairs
TAU_SEGMENTS = [(0, 4), (4, 8), (8, 12)]  # of each tau input's 12 events


def json_lines(path):
    with open(path) as stream:
        return [json.loads(line) for line in stream]


def events_tree(path):
    with uproot.open(path) as stream:
        return stream["events"].arrays(library="ak")


def every_file(folder):
    found = set()
    for root, folders, names in os.walk(folder):
        folders[:] = [f for f in folders if not f.startswith(".")]
        found.update(os.path.relpath(os.path.join(root, name), folder) for name in names)
    return found


def read_csv(path):
    with open(path) as stream:
        return list(csv.DictReader(stream))


@pytest.fixture
def whole(bench):
    """The pipeline run with di_photon's and tau_pairs' inputs whole, and then
    both put in three segments, as the default configuration has them."""
    bench.with_tau_pairs()
    bench.run("pipeline.sh")
    bench.metrics = open(bench.path("observables", "plots", "metrics.csv")).read()
    bench.with_segments(3)
    return bench


def test_whole_data_is_split_into_segments_that_measure_the_same(whole):
    bench = whole
    # The pipeline won't make the segments anew while the whole data is there
    status, output = bench.run("pipeline.sh", "--dry-run", check=False)
    assert status != 0 and "the inputs of di_photon and tau_pairs are in segments now " \
        "(input.segments), but 19 file(s) made from whole inputs before aren't split into " \
        "them yet" in output and "split them with scripts/split_data.py first" in output, output

    # Without --apply, nothing changes
    before = every_file(bench.data)
    status, output = bench.run("scripts/split_data.py")
    assert "di_photon: %s\n  30 events, counted from its input, in 3 segments of 10, 10, 10" \
        % DI_PHOTON in output, output
    assert "  sim_output/di_photon/geant4/%s_simseed1.edm4hep.root, 16.6 kB: to split\n" \
        "    and its .log (a log for each segment): then to .unsplit" % DI_PHOTON in output, output
    assert "    and its .log (a log for each segment), _AIDA.root and _PfoAnalysis.root (kept " \
        "whole): then to .unsplit" in output, output
    assert "tau_pairs: taus_b\n  12 events, counted from its input, in 3 segments of 4, 4, 4" \
        in output, output
    assert "split_data: run again with --apply to split" in output, output
    assert every_file(bench.data) == before

    # Split
    status, output = bench.run("scripts/split_data.py", "--apply")
    assert "split_data: split 20 file(s)" in output and "moved 52 file(s)" in output, output
    assert "copied 4 measurement and metrics file(s) to .unsplit/observables" in output, output
    unsplit = bench.path(".unsplit")
    gun = "input_single_particle_gun"
    assert bench.files(gun) == sorted(
        [name + ending for name in ("p22_th43-137_ph79-109_en20_seed0_ip_surface",
                                    "p22_th43-137_ph79-109_en80_seed1_ip_surface")
         for ending in (".h5", ".slcio")]
        + ["%s_seg%dof3%s" % (DI_PHOTON, i, ending) for i in (1, 2, 3)
           for ending in (".h5", ".slcio")])
    assert bench.files("input_physics_samples") == [
        "taus_%s_seg%dof3.slcio" % (name, i) for name in "ab" for i in (1, 2, 3)]
    # Each segment is the whole's events at its place
    whole_input = json_lines(os.path.join(unsplit, gun, DI_PHOTON + ".slcio"))
    for index, (start, stop) in enumerate(SEGMENTS, 1):
        records = json_lines(bench.path(gun, "%s_seg%dof3.slcio" % (DI_PHOTON, index)))
        assert records == whole_input[:1] + whole_input[1 + start:1 + stop]
    with open(os.path.join(unsplit, "sim_output", "di_photon", "geant4",
                           DI_PHOTON + "_simseed2.edm4hep.root")) as stream:
        simulation = json.load(stream)
    rec = events_tree(os.path.join(unsplit, "rec_output", "tau_pairs", "geant4",
                                   "taus_b_simseed1764_REC.edm4hep.root"))
    for index, (start, stop) in enumerate(SEGMENTS, 1):
        with open(bench.path("sim_output", "di_photon", "geant4", "%s_seg%dof3_simseed2"
                             ".edm4hep.root" % (DI_PHOTON, index))) as stream:
            part = json.load(stream)
        assert part["events"] == simulation["events"][start:stop]
        assert part["seed"] == 2 and part["simulator"] == "geant4"
    for index, (start, stop) in enumerate(TAU_SEGMENTS, 1):
        part = events_tree(bench.path("rec_output", "tau_pairs", "geant4",
                                      "taus_b_seg%dof3_simseed1764_REC.edm4hep.root" % index))
        assert part.fields == rec.fields
        assert ak.to_list(part) == ak.to_list(rec[start:stop])
    # Each segment's log has its whole job's header, without its time
    folder = bench.path("sim_output", "di_photon", "caloclouds3_paper")
    assert bench.files("sim_output", "di_photon", "caloclouds3_paper") == sorted(
        "%s_seg%dof3_simseed%d%s" % (DI_PHOTON, i, seed, ending) for i in (1, 2, 3)
        for seed in (1, 2) for ending in (".log", ".root"))
    log = os.path.join(folder, DI_PHOTON + "_seg3of3_simseed1.log")
    whole_log = os.path.join(unsplit, "sim_output", "di_photon", "caloclouds3_paper",
                             DI_PHOTON + "_simseed1.log")
    text = open(log).read()
    assert "# segment   3 of 3: events 20 to 29, 10 of the 30 of the job above, whose output " \
        "was sim_output/di_photon/caloclouds3_paper/%s_simseed1.root" % DI_PHOTON in text, text
    assert "# whole log .unsplit/sim_output/di_photon/caloclouds3_paper/%s_simseed1.log" \
        % DI_PHOTON in text and "# finished" not in text, text
    # which the manifests read as the whole job's
    found = manifest.read_log(log)
    assert found == manifest.read_log(whole_log) and found["seed"] == 1
    assert found["key4hep"] == "2026-04-08" and found["presets"] == "CC3_BARREL", found
    # The reconstructions' histograms are kept whole, beside the whole files
    assert bench.files("rec_output", "tau_pairs", "geant4") == sorted(
        "taus_%s_seg%dof3_simseed%d%s" % (name, i, seed, ending)
        for name, seed in (("a", 42), ("b", 1764)) for i in (1, 2, 3)
        for ending in (".log", "_REC.edm4hep.root"))
    assert sorted(os.listdir(os.path.join(unsplit, "rec_output", "tau_pairs", "geant4"))) == \
        sorted("taus_%s_simseed%d%s" % (name, seed, ending)
               for name, seed in (("a", 42), ("b", 1764))
               for ending in (".log", "_AIDA.root", "_PfoAnalysis.root", "_REC.edm4hep.root"))

    # The measurements are made again from the segments, and nothing else
    status, output = bench.run("pipeline.sh", "--dry-run")
    assert "would run 3 task(s)" in output and "the reconstructed files of di_photon changed" \
        in output, output
    status, output = bench.run("pipeline.sh")
    assert "made 3 of 3" in output, output
    # and they measure the same: the metrics are those of the whole data
    assert open(bench.path("observables", "plots", "metrics.csv")).read() == bench.metrics
    assert open(bench.path(".unsplit", "observables", "plots", "metrics.csv")).read() == \
        bench.metrics
    with np.load(bench.path("observables", "geant4.npz")) as data:
        assert list(data["di_photon__files"]) == [
            "rec_output/di_photon/geant4/%s_seg%dof3_simseed%d_REC.edm4hep.root"
            % (DI_PHOTON, i, seed) for i in (1, 2, 3) for seed in (1, 2)]
        assert list(data["di_photon__seeds"]) == [1, 2] * 3
        assert list(data["tau_pairs__seeds"]) == [42] * 3 + [1764] * 3
        assert list(data["tau_pairs__n_events"]) == [4] * 6

    # Run again, it has nothing left to do
    status, output = bench.run("scripts/split_data.py", "--apply")
    assert output.count("nothing of it to split: split already") == 3, output
    assert "split_data: nothing left to split" in output, output

    # The step times leave out the segments' logs, and time the whole jobs from
    # theirs, in .unsplit
    def step_times(data_dir):
        out = os.path.join(bench.tmp, "steps.csv")
        result = subprocess.run(
            [sys.executable, os.path.join(REPO, "scripts", "plot_step_times.py"),
             os.path.join(bench.tmp, "no-slurm"), "--data-dir", data_dir, "--csv", out,
             "-o", os.path.join(bench.tmp, "steps.png")],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        assert result.returncode == 0, result.stdout
        return result.stdout, read_csv(out)

    output, jobs = step_times(bench.data)
    assert "skipped    48: split from a whole job's output by split_data.py" in output, output
    assert {(job["step"], job["scenario"]) for job in jobs} == {("2", "single_photon"),
                                                                ("3", "single_photon")}
    output, jobs = step_times(unsplit)
    assert len([job for job in jobs if job["scenario"] == "di_photon"]) == 8, output

    # What would be published: the segments' inputs, and nothing of .unsplit
    out = os.path.join(bench.tmp, "records")
    bench.run("scripts/ddml-bench", "manifest", "--out", out, "--records", "essential")
    with open(os.path.join(out, "essential", "manifest.json")) as stream:
        published = json.load(stream)
    paths = sorted(entry["path"] for entry in published["files"] if "path" in entry)
    assert [path for path in paths if "diphoton" in path or "taus" in path] == sorted(
        ["%s/%s_seg%dof3.slcio" % (gun, DI_PHOTON, i) for i in (1, 2, 3)]
        + ["input_physics_samples/taus_%s_seg%dof3.slcio" % (name, i)
           for name in "ab" for i in (1, 2, 3)])
    assert not [path for path in paths if path.startswith(".")]


def test_what_doesnt_split_cleanly_is_left_as_it_is(bench):
    """A file without as many events as its input, or with a segment there
    already that isn't its events, is left whole with the rest of its
    simulation or reconstruction, and said so; the rest is split, and a run
    after the problem is fixed splits what is left. --only splits some."""
    bench.run("pipeline.sh", "--stages", "generate,geant4", "-s", "di_photon")
    bench.with_segments(3)
    folder = bench.path("sim_output", "di_photon", "geant4")
    first = os.path.join(folder, DI_PHOTON + "_simseed1.edm4hep.root")
    second = os.path.join(folder, DI_PHOTON + "_simseed2.edm4hep.root")
    with open(first) as stream:
        simulation = json.load(stream)
    shorter = dict(simulation, events=simulation["events"][:-1])
    with open(first, "w") as stream:
        json.dump(shorter, stream)
    # A segment there already, but not of these events
    with open(os.path.join(folder, DI_PHOTON + "_seg2of3_simseed2.edm4hep.root"), "w") as stream:
        json.dump(dict(simulation, events=simulation["events"][:10]), stream)
    leftover = os.path.join(folder, ".split.elsewhere.1.%s_seg1of3_simseed1.edm4hep.root"
                            % DI_PHOTON)
    open(leftover, "w").close()

    status, output = bench.run("scripts/split_data.py", "--apply", "--only", "inputs",
                               check=False)
    assert status == 0 and "split 2 file(s)" in output, output
    assert bench.files("input_single_particle_gun").count(DI_PHOTON + "_seg1of3.slcio") == 1
    assert os.path.exists(first) and os.path.exists(second)

    status, output = bench.run("scripts/split_data.py", "--apply", "--only", "geant4",
                               check=False)
    assert status == 1, output
    assert "30 events, counted from its input's segments, in 3 segments of 10, 10, 10" \
        in output, output
    assert "PROBLEM: sim_output/di_photon/geant4/%s_simseed1.edm4hep.root: it has 29 events, " \
        "not 30 as its input" % DI_PHOTON in output, output
    assert "PROBLEM: sim_output/di_photon/geant4/%s_simseed2.edm4hep.root: its segment " \
        "%s_seg2of3_simseed2.edm4hep.root is there already, but isn't its events 10 to 19 " \
        "(its event 0 isn't event 10 of the whole (its collections, their sizes or its " \
        "parameters differ)); move that away" % (DI_PHOTON, DI_PHOTON) \
        in output, output
    assert "warning: 1 hidden file(s) of a split that stopped (or that is running now)" in \
        output, output
    assert "split_data: 2 problem(s), whose files are left as they were:" in output, output
    # Both left whole, with their logs
    for path in (first, second):
        assert os.path.exists(path) and os.path.exists(path[:-len(".edm4hep.root")] + ".log")
    assert not os.path.exists(os.path.join(folder, DI_PHOTON + "_seg1of3_simseed1.edm4hep.root"))

    # Fixed, the rest is split, checking the segment there already
    with open(first, "w") as stream:
        json.dump(simulation, stream)
    with open(os.path.join(folder, DI_PHOTON + "_seg2of3_simseed2.edm4hep.root"), "w") as stream:
        json.dump(dict(simulation, events=simulation["events"][10:20]), stream)
    os.remove(leftover)
    status, output = bench.run("scripts/split_data.py", "--apply")
    assert "split 2 file(s)" in output and "with 1 of their segments there already, checked" \
        in output, output
    assert bench.files("sim_output", "di_photon", "geant4") == sorted(
        "%s_seg%dof3_simseed%d%s" % (DI_PHOTON, i, seed, ending) for i in (1, 2, 3)
        for seed in (1, 2) for ending in (".edm4hep.root", ".log"))
    # and the pipeline goes on from there: di_photon's segments made by Geant4
    # are done, and the rest is to make
    status, output = bench.run("pipeline.sh", "--dry-run")
    assert "  generate          3      2       0          0            0\n" \
        "  geant4            6      2       0          0            0" in output, output


def test_a_fresh_data_folder_takes_the_inputs_in_segments(bench):
    """With the inputs in segments from the start, the gun makes di_photon's,
    and tau_pairs' whole files are split by split_data.py first."""
    bench.with_tau_pairs()
    bench.with_segments(3)
    status, output = bench.run("pipeline.sh", "--dry-run", check=False)
    assert status != 0 and "e.g. input_physics_samples/taus_a.slcio: split them with " \
        "scripts/split_data.py first" in output, output
    status, output = bench.run("scripts/split_data.py", "--apply")
    assert "split 2 file(s)" in output and "moved 2 file(s)" in output, output
    status, output = bench.run("pipeline.sh")
    # 5 inputs made, 14 simulations by each simulator, each reconstructed, 2
    # measurements and the plots
    assert "made 64 of 64" in output, output
    assert bench.files("input_single_particle_gun").count(DI_PHOTON + "_seg3of3.h5") == 1
    rows = read_csv(bench.path("observables", "plots", "metrics.csv"))
    assert {row["observable"] for row in rows} >= {"photon_separation", "pi0"}
    # The JS of the separation between seeds, each of its three segments
    js, = [row for row in rows if row["observable"] == "photon_separation"]
    assert float(js["uncertainty"]) > 0


@pytest.fixture
def formats(monkeypatch):
    """ddml_bench.split's formats, in this process, on the fakes' pyLCIO and
    podio (tests/fakes/python)."""
    from conftest import FAKES
    from ddml_bench import split
    monkeypatch.syspath_prepend(os.path.join(FAKES, "python"))
    for module in ("pyLCIO", "podio", "podio.reading", "podio.root_io"):
        monkeypatch.delitem(sys.modules, module, raising=False)
    return split


def test_the_checks_find_segments_that_arent_the_wholes_events(formats, tmp_path):
    from conftest import write_tau_input
    split = formats
    # LCIO: the events of a segment, after the run header, numbered as in the whole
    whole = str(tmp_path / "taus.slcio")
    write_tau_input(whole, 9, 0)
    lcio = split.Lcio()
    assert lcio.count(whole) == 9
    part = str(tmp_path / "taus_seg2of3.slcio")
    lcio.write(whole, 3, 6, part)
    records = json_lines(whole)
    assert json_lines(part) == records[:1] + records[4:7]
    assert lcio.check(whole, 3, 6, part) is None
    assert lcio.check(whole, 0, 3, part) == "its event 0 isn't event 0 of the whole"
    assert lcio.check(whole, 3, 7, part) == "it has 3 events, not 4"

    # podio (the fake k4run's REC files): the events' frames, value by value
    rec = str(tmp_path / "taus_REC.edm4hep.root")
    columns = {"PandoraPFOs.energy": ak.values_astype(ak.Array(
        [[float(i), 2.0 * i] for i in range(9)]), np.float32),
        "PandoraPFOs.PDG": ak.values_astype(ak.Array([[22, 22]] * 9), np.int32)}
    with uproot.recreate(rec) as stream:
        stream.mktree("events", dict((name, array.type) for name, array in columns.items()))
        stream["events"].extend(columns)
    podio = split.Podio()
    assert podio.count(rec) == 9
    part = str(tmp_path / "taus_seg2of3_REC.edm4hep.root")
    podio.write(rec, 3, 6, part)
    assert podio.check(rec, 3, 6, part) is None
    assert ak.to_list(events_tree(part)["PandoraPFOs.energy"]) == [[3.0, 6.0], [4.0, 8.0],
                                                                   [5.0, 10.0]]
    # The sizes are the same in every event, so only the values tell
    assert podio.check(rec, 0, 3, part) == \
        "PandoraPFOs.energy of its event 1 isn't that of event 1 of the whole"
    assert podio.check(rec, 3, 7, part) == "it has 3 frames of events, not 4"

    # The guns' truth
    import h5py
    truth = str(tmp_path / "truth.h5")
    with h5py.File(truth, "w") as stream:
        stream.create_dataset("energy", data=np.arange(9.0))
        stream.create_dataset("input_p_global", data=np.arange(54.0).reshape(9, 2, 3))
    h5 = split.Truth()
    assert h5.count(truth) == 9
    part = str(tmp_path / "truth_seg3of3.h5")
    h5.write(truth, 6, 9, part)
    assert h5.check(truth, 6, 9, part) is None
    assert h5.check(truth, 3, 6, part) == "its energy isn't that of events 3 to 5 of the whole"
