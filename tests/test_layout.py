"""The layout: the one place that names files."""

import os
import re
import subprocess
import sys

import pytest

from conftest import REPO
from ddml_bench.layout import (Layout, LayoutError, is_sample, is_sample_suffix,
                               measurement_files, sample_suffix, seed_suffix, simulation_stem)

INPUT = "/data/input_single_particle_gun/p22_en20.slcio"


def test_a_sample_is_named_alike_at_every_stage():
    layout = Layout("/data", events=10)
    assert layout.simulation("sp", "geant4", INPUT) == \
        "/data/sim_output/sp/geant4/p22_en20_n10.edm4hep.root"
    assert layout.simulation("sp", "cc3", INPUT) == "/data/sim_output/sp/cc3/p22_en20_n10.root"
    assert layout.simulation_log("sp", "cc3", INPUT) == "/data/sim_output/sp/cc3/p22_en20_n10.log"
    base = layout.reconstruction_base("sp", "cc3", INPUT)
    assert base == "/data/rec_output/sp/cc3/p22_en20_n10"
    # The reconstruction of a simulation is named after the simulation
    assert os.path.basename(base) == simulation_stem(
        os.path.basename(layout.simulation("sp", "cc3", INPUT)))
    assert layout.observables("cc3") == "/data/observables/cc3_n10.npz"
    assert layout.metrics == "/data/observables/plots/metrics_n10.csv"

    full = Layout("/data")
    assert full.simulation("sp", "geant4", INPUT) == "/data/sim_output/sp/geant4/p22_en20.edm4hep.root"
    assert full.reconstruction_base("sp", "geant4", INPUT) == "/data/rec_output/sp/geant4/p22_en20"
    assert full.observables("geant4") == "/data/observables/geant4.npz"
    assert full.metrics == "/data/observables/plots/metrics.csv"


@pytest.mark.parametrize("events", [0, -3, True, "5"])
def test_a_sample_has_at_least_one_event(events):
    with pytest.raises(LayoutError):
        sample_suffix(events)


def test_a_sample_can_have_a_number_of_events_for_each_scenario():
    events = {"sp": 1000, "dp": 1000, "taus": 500}
    layout = Layout("/data", events=events)
    # Each input's simulation, and what is made from it, by its scenario's N
    assert layout.simulation("sp", "cc3", INPUT) == "/data/sim_output/sp/cc3/p22_en20_n1000.root"
    assert layout.simulation("taus", "cc3", INPUT, 42) == \
        "/data/sim_output/taus/cc3/p22_en20_n500_simseed42.root"
    assert layout.reconstruction_base("taus", "cc3", INPUT, 42) == \
        "/data/rec_output/taus/cc3/p22_en20_n500_simseed42"
    assert layout.events_of("taus") == 500 and layout.events_of("sp") == 1000
    # The measurements and plots by all of them: the most common N, then the
    # scenarios of others, in order
    assert layout.suffix == "_n1000_taus500"
    assert layout.observables("cc3") == "/data/observables/cc3_n1000_taus500.npz"
    assert layout.low_cost_plots_dir == "/data/observables/plots/low_cost_n1000_taus500"
    # The same N everywhere is the sample of -n N
    assert Layout("/data", {"sp": 7, "dp": 7}).suffix == "_n7" == sample_suffix(7)
    assert sample_suffix({"a": 5, "b": 6, "c": 6}) == "_n6_a5"
    # (ties go to the first scenario's)
    assert sample_suffix({"a": 5, "b": 6}) == "_n5_b6"
    with pytest.raises(LayoutError):
        layout.sample_name(INPUT)  # which scenario's?
    with pytest.raises(LayoutError):
        layout.events_of("elsewhere")
    for wrong in ({}, {"sp": 0}, {"sp": "5"}, {"no/good": 5}):
        with pytest.raises(LayoutError):
            sample_suffix(wrong)


def test_the_measurements_of_any_sample_are_told_from_the_full_ones():
    for name in ("geant4_n5", "cc3_n1000_tau_pairs500", "cc3_n1000_di_photon2_tau_pairs500"):
        assert is_sample(name), name
    for name in ("geant4", "geant4_2", "caloclouds3_recompiled_5GeV", "l2lflows", "cc_n"):
        assert not is_sample(name), name
    assert is_sample_suffix("_n1000") and is_sample_suffix("_n1000_tau_pairs500")
    for text in ("n1000", "_n", "_n1000_", "_n1000_tau_pairs", "_n10/0"):
        assert not is_sample_suffix(text), text


def test_a_simulation_with_a_seed_of_its_own_is_named_after_it():
    layout = Layout("/data", events=10)
    assert layout.sample_name(INPUT, 2) == "p22_en20_n10_simseed2"
    assert layout.simulation("dp", "geant4", INPUT, 2) == \
        "/data/sim_output/dp/geant4/p22_en20_n10_simseed2.edm4hep.root"
    assert layout.simulation_log("dp", "cc3", INPUT, 2) == \
        "/data/sim_output/dp/cc3/p22_en20_n10_simseed2.log"
    base = layout.reconstruction_base("dp", "cc3", INPUT, 2)
    assert base == "/data/rec_output/dp/cc3/p22_en20_n10_simseed2"
    assert os.path.basename(base) == simulation_stem(
        os.path.basename(layout.simulation("dp", "cc3", INPUT, 2)))
    # Without a seed of its own, as before
    assert Layout("/data").simulation("dp", "cc3", INPUT, None) == \
        "/data/sim_output/dp/cc3/p22_en20.root"
    assert seed_suffix(None) == "" and seed_suffix(0) == "_simseed0"
    for seed in (-1, True, "2", 1.5):
        with pytest.raises(LayoutError):
            seed_suffix(seed)


def test_simulations_given_directly(tmp_path):
    data = str(tmp_path / "data")
    layout = Layout(data)
    inside = os.path.join(data, "sim_output", "sp", "geant4", "x.edm4hep.root")
    outside = str(tmp_path / "elsewhere" / "y.root")
    for path in (inside, outside):
        os.makedirs(os.path.dirname(path))
        open(path, "w").close()
    assert layout.given_simulation(inside) == (
        "sp", "geant4", os.path.join(os.path.realpath(data), "rec_output", "sp", "geant4", "x")
        .replace(os.path.realpath(data), data))
    assert layout.given_simulation(outside) == (
        "custom", "files", os.path.join(data, "rec_output", "custom", "files", "y"))
    with pytest.raises(LayoutError):
        layout.given_simulation(os.path.join(os.path.dirname(outside), ".hidden.root"))


def test_measurement_files(tmp_path):
    for name in ("geant4.npz", "geant4_n5.npz", ".geant4.npz.123.npz", "cc3.npz", "notes.txt",
                 "cc3_n5_tau_pairs2.npz"):
        open(str(tmp_path / name), "w").close()
    folder = str(tmp_path)
    assert measurement_files(folder) == [("cc3", os.path.join(folder, "cc3.npz")),
                                         ("geant4", os.path.join(folder, "geant4.npz"))]
    assert measurement_files(folder, "_n5") == [("geant4", os.path.join(folder, "geant4_n5.npz"))]
    assert measurement_files(folder, "_n5_tau_pairs2") == [
        ("cc3", os.path.join(folder, "cc3_n5_tau_pairs2.npz"))]


def test_records_are_relative_to_the_data():
    layout = Layout("/data")
    assert layout.relative("/data/rec_output/x_REC.edm4hep.root") == "rec_output/x_REC.edm4hep.root"
    assert layout.relative("/elsewhere/x.root") == "/elsewhere/x.root"
    assert layout.absolute("rec_output/x") == "/data/rec_output/x"
    assert layout.absolute("/elsewhere/x") == "/elsewhere/x"


def test_planning_needs_only_the_standard_library():
    """The layout, the command line and naming the gun's inputs work with any
    Python: without PyYAML, numpy or pyLCIO (python -S has no site-packages)."""
    code = """
import sys
sys.path.insert(0, %r)
import ddml_bench.cli, ddml_bench.plot
from ddml_bench.scenarios import di_photon_gun_files, particle_gun_files
files = particle_gun_files([["python3", "ILD_Particle_Gun.py", "--pid", "22", "--energy", "20",
                             "--save_dir", "/data/in"]])
assert files == ["/data/in/p22_th90_ph90_en20_seed0_ip.slcio"], files
files = di_photon_gun_files([["python3", "ILD_Di_Photon_Gun.py", "--energy", "100", "--seed", "1",
                              "--save_dir", "/data/in"]])
assert files == ["/data/in/diphoton_en100_sep90_x0_z70_r10_seed1.slcio"], files
files = di_photon_gun_files([["python3", "ILD_Di_Photon_Gun.py", "--seed", "1", "--segment", "2",
                              "--segments", "3", "--save_dir", "/data/in"]])
assert files == ["/data/in/diphoton_en20_sep90_x0_z70_r10_seed1_seg2of3.slcio"], files
heavy = [m for m in ("yaml", "numpy", "pyLCIO", "h5py", "particle") if m in sys.modules]
assert not heavy, heavy
""" % REPO
    subprocess.check_call([sys.executable, "-S", "-c", code])


def test_each_scenarios_inputs_go_in_a_folder_of_the_data_folder():
    from ddml_bench.layout import check_input_folder
    layout = Layout("/data")
    assert layout.input_dir == "/data/input_single_particle_gun"
    assert layout.inputs("input_physics_samples") == "/data/input_physics_samples"
    assert check_input_folder("input_physics_samples", "x") == "input_physics_samples"
    for folder, message in (("rec_output", "can't be rec_output"),
                            ("../elsewhere", "not '../elsewhere'"),
                            (".hidden", "not '.hidden'")):
        with pytest.raises(LayoutError, match=re.escape(message)):
            check_input_folder(folder, "scenarios.s.input.folder")


def test_an_inputs_segments_are_its_events_in_order():
    """input.segments: the events of an input in N files, as evenly as they
    go, the first ones one longer, each named after the whole input."""
    from ddml_bench.layout import segment_bounds, segment_suffix
    assert segment_bounds(15000, 3) == [(0, 5000), (5000, 10000), (10000, 15000)]
    assert segment_bounds(6791, 3) == [(0, 2264), (2264, 4528), (4528, 6791)]
    assert segment_bounds(10, 3) == [(0, 4), (4, 7), (7, 10)]
    assert segment_bounds(7, 1) == [(0, 7)]
    assert segment_bounds(3, 3) == [(0, 1), (1, 2), (2, 3)]
    assert segment_suffix(2, 3) == "_seg2of3" and segment_suffix(1, 1) == ""
    for args, message in (((2, 3), "can't be split into 3 segments"),
                          ((10, 0), "at least 1"), ((10, True), "at least 1")):
        with pytest.raises(LayoutError, match=message):
            segment_bounds(*args)
    with pytest.raises(LayoutError, match="there's no segment 4 of 3"):
        segment_suffix(4, 3)
    # Everything made from a segment is named after it, as after any input
    layout = Layout("/data")
    segment = "/data/input_single_particle_gun/diphoton_en20_seed0" + segment_suffix(2, 3) \
        + ".slcio"
    assert layout.simulation("di_photon", "geant4", segment, 1) == \
        "/data/sim_output/di_photon/geant4/diphoton_en20_seed0_seg2of3_simseed1.edm4hep.root"
    assert layout.reconstruction_base("di_photon", "geant4", segment, 1) == \
        "/data/rec_output/di_photon/geant4/diphoton_en20_seed0_seg2of3_simseed1"
