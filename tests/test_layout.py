"""The layout: the one place that names files."""

import os
import subprocess
import sys

import pytest

from conftest import REPO
from ddml_bench.layout import (Layout, LayoutError, measurement_files, sample_suffix,
                               seed_suffix, simulation_stem)

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
    for name in ("geant4.npz", "geant4_n5.npz", ".geant4.npz.123.npz", "cc3.npz", "notes.txt"):
        open(str(tmp_path / name), "w").close()
    folder = str(tmp_path)
    assert measurement_files(folder) == [("cc3", os.path.join(folder, "cc3.npz")),
                                         ("geant4", os.path.join(folder, "geant4.npz"))]
    assert measurement_files(folder, "_n5") == [("geant4", os.path.join(folder, "geant4_n5.npz"))]


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
heavy = [m for m in ("yaml", "numpy", "pyLCIO", "h5py", "particle") if m in sys.modules]
assert not heavy, heavy
""" % REPO
    subprocess.check_call([sys.executable, "-S", "-c", code])
