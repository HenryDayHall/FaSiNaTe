"""ILD_Di_Photon_Gun.py: pairs of photons from the IP into the ECAL barrel's
top stave, placed as the earlier di-photon generator placed them."""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

from conftest import FAKES, REPO

import ILD_Di_Photon_Gun as gun

N = 20000


def pairs(seed=0, **options):
    settings = dict(n_events=N, energy=20.0, max_separation=90.0, centre_radius=10.0,
                    centre_x=0.0, centre_z=70.0, surface_y=gun.SURFACE_Y)
    settings.update(options)
    rng = np.random.default_rng(seed)
    return gun.pick_pairs(settings.pop("n_events"), settings.pop("energy"),
                          settings.pop("max_separation"), settings.pop("centre_radius"),
                          settings.pop("centre_x"), settings.pop("centre_z"),
                          settings.pop("surface_y"), rng)


def test_both_photons_start_on_the_face_and_fly_from_the_ip():
    vertex, momentum, separation = pairs()
    assert vertex.shape == momentum.shape == (N, 2, 3) and separation.shape == (N,)
    np.testing.assert_array_equal(vertex[:, :, 1], gun.SURFACE_Y)
    # along the line from the IP through where each starts, with its energy
    np.testing.assert_allclose(np.linalg.norm(momentum, axis=2), 20.0)
    np.testing.assert_allclose(momentum / 20.0,
                               vertex / np.linalg.norm(vertex, axis=2, keepdims=True), atol=1e-15)
    np.testing.assert_allclose(np.linalg.norm(vertex[:, 0] - vertex[:, 1], axis=1), separation)
    # so near theta = 87.8 and phi = 90 degrees, well inside the top stave
    theta = np.degrees(np.arctan2(np.hypot(vertex[..., 0], vertex[..., 1]), vertex[..., 2]))
    phi = np.degrees(np.arctan2(vertex[..., 1], vertex[..., 0]))
    assert 86 < theta.min() and theta.max() < 89.6
    assert 88.2 < phi.min() and phi.max() < 91.8


def test_the_pairs_are_placed_as_the_earlier_generator_placed_them():
    """The midpoint uniform over a disc of 10 mm about (0, 70), the
    separation uniform from 0 to 90 mm, and the line between the two photons
    at a uniform angle in the plane of the face."""
    vertex, _, separation = pairs()
    assert separation.min() >= 0 and separation.max() <= 90
    # uniform: a quarter in each quarter of the range, within the statistics
    counts = np.histogram(separation, bins=4, range=(0, 90))[0]
    assert np.all(np.abs(counts / N - 0.25) < 4 * np.sqrt(0.25 * 0.75 / N)), counts
    middle = vertex.mean(axis=1)
    radius = np.hypot(middle[:, 0], middle[:, 2] - 70.0)
    assert radius.max() <= 10.0 + 1e-9
    # uniform over the disc's area: half of the midpoints within 10/sqrt(2) mm
    assert abs(np.mean(radius < 10 / np.sqrt(2)) - 0.5) < 4 * np.sqrt(0.25 / N)
    # the line between the two at any angle
    line = vertex[:, 0] - vertex[:, 1]
    angle = np.arctan2(line[:, 2], line[:, 0])
    counts = np.histogram(angle, bins=4, range=(-np.pi, np.pi))[0]
    assert np.all(np.abs(counts / N - 0.25) < 4 * np.sqrt(0.25 * 0.75 / N)), counts


def test_the_options_move_and_size_the_pairs():
    vertex, momentum, separation = pairs(n_events=500, energy=100.0, max_separation=30.0,
                                         centre_radius=0.0, centre_x=-50.0, centre_z=200.0,
                                         surface_y=1500.0)
    assert separation.max() <= 30
    np.testing.assert_allclose(vertex.mean(axis=1), [[-50.0, 1500.0, 200.0]] * 500, atol=1e-9)
    np.testing.assert_allclose(np.linalg.norm(momentum, axis=2), 100.0)


def test_a_seed_gives_the_same_pairs_every_time():
    first, again, other = pairs(seed=4)[0], pairs(seed=4)[0], pairs(seed=5)[0]
    np.testing.assert_array_equal(first, again)
    assert not np.allclose(first, other)


def test_the_file_names_say_how_the_pairs_were_made():
    parser = gun.get_parser()
    args = parser.parse_args(["--energy", "100", "--seed", "3", "--save_dir", "/data/in"])
    assert gun.get_file_name(args) == "/data/in/diphoton_en100_sep90_x0_z70_r10_seed3.slcio"
    args = parser.parse_args(["--centre_z", "-12.5", "--max_separation", "30", "--surface_y",
                              "1500", "--save_dir", "/data/in"])
    assert gun.get_file_name(args) == "/data/in/diphoton_en20_sep30_x0_z-12.5_r10_y1500_seed0.slcio"


@pytest.mark.parametrize("option, message", [
    (["--n_events", "0"], "n_events should be at least 1"),
    (["--energy", "-5"], "energy should be above 0 GeV"),
    (["--max_separation", "0"], "max_separation should be above 0 mm"),
    (["--centre_radius", "-1"], "centre_radius should be at least 0 mm"),
    (["--surface_y", "-1804.7"], "surface_y should be above 0 mm"),
])
def test_options_that_cant_make_pairs_are_refused(option, message):
    with pytest.raises(ValueError, match=message):
        gun.check_args(gun.get_parser().parse_args(option))


def test_it_writes_two_photons_per_event_and_the_truth(tmp_path):
    """As the generate stage runs it, with the tests' pyLCIO, which writes
    JSON."""
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([os.path.join(FAKES, "python"), REPO]))
    subprocess.check_call([sys.executable, os.path.join(REPO, "ILD_Di_Photon_Gun.py"),
                           "--n_events", "50", "--energy", "20", "--seed", "7",
                           "--save_dir", str(tmp_path)], env=env, stdout=subprocess.DEVNULL)
    name = "diphoton_en20_sep90_x0_z70_r10_seed7"
    assert sorted(p.name for p in tmp_path.iterdir()) == [name + ".h5", name + ".slcio"]
    with open(str(tmp_path / (name + ".slcio"))) as stream:
        records = [json.loads(line) for line in stream]
    assert records[0]["run"]["Generator"] == "ILD_Di_Photon_Gun.py"
    events = records[1:]
    assert [event["event"] for event in events] == list(range(50))
    import h5py
    with h5py.File(str(tmp_path / (name + ".h5")), "r") as h5:
        vertex, momentum = h5["input_gun_position"][:], h5["input_p_global"][:]
        separation, energy = h5["separation"][:], h5["energy"][:]
    np.testing.assert_array_equal(energy, 20.0)
    expected = pairs(seed=7, n_events=50)
    np.testing.assert_array_equal(vertex, expected[0])
    np.testing.assert_array_equal(separation, expected[2])
    for event, event_vertex, event_momentum in zip(events, vertex, momentum):
        assert len(event["particles"]) == 2
        for particle, v, p in zip(event["particles"], event_vertex, event_momentum):
            assert (particle["generatorStatus"], particle["PDG"], particle["mass"],
                    particle["charge"]) == (1, 22, 0.0, 0.0)
            assert particle["vertex"] == particle["endpoint"] == list(v)
            # (LCIO keeps momenta as floats)
            np.testing.assert_allclose(particle["momentum"], p, rtol=1e-6)
