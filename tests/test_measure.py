"""Measuring the reconstructed energy of each event, on a small EDM4hep-style
file written here (a podio "events" tree, as tests/fakes/bin/k4run writes)."""

import json
import os

import numpy as np
import pytest

from ddml_bench import measure
from ddml_bench.layout import Layout

SIMULATOR = {"name": "geant4", "label": "Geant4", "color": "#9a9a9a", "linestyle": "-",
             "compact_file": None}


def write_events(path, events):
    """A reconstructed file of events, each (generator status of its MC
    particle, its momentum along y, [(energy, PDG code) of each reconstructed
    particle]), with a hit each in the ECAL barrel."""
    import awkward as ak
    import uproot
    columns = {"MCParticles.generatorStatus": [[status] for status, _, _ in events],
               "MCParticles.mass": [[0.0] for _ in events],
               "PandoraPFOs.energy": [[e for e, _ in pfos] for _, _, pfos in events],
               "PandoraPFOs.PDG": [[pdg for _, pdg in pfos] for _, _, pfos in events],
               "EcalBarrelCollectionRec.energy": [[0.1] for _ in events],
               "EcalBarrelCollectionRec.cellID": [[1 << 16] for _ in events]}
    for c in "xyz":
        columns["MCParticles.vertex." + c] = [[0.0] for _ in events]
        columns["MCParticles.momentum." + c] = [[p if c == "y" else 0.0] for _, p, _ in events]
        columns["EcalBarrelCollectionRec.position." + c] = [
            [1800.0 if c == "y" else 1.0] for _ in events]
    types = {"MCParticles.generatorStatus": np.int32, "PandoraPFOs.PDG": np.int32,
             "PandoraPFOs.energy": np.float32, "EcalBarrelCollectionRec.cellID": np.uint64}
    arrays = dict((name, ak.values_astype(ak.Array(values), types.get(name, np.float64)))
                  for name, values in columns.items())
    with uproot.recreate(path) as output:
        output.mktree("events", dict((name, array.type) for name, array in arrays.items()))
        output["events"].extend(arrays)
    return path


@pytest.fixture
def reconstructed(tmp_path):
    # Two reconstructed particles, none, and an event without the incident
    # photon (generator status 1)
    return write_events(str(tmp_path / "photons_REC.edm4hep.root"), [
        (1, 20.0, [(19.0, 22), (0.5, -11)]),
        (1, 20.0, []),
        (0, 20.0, [(3.0, 22)]),
    ])


def configured(**energy):
    settings = dict({"scenarios": ["s"], "particle_collections": ["PandoraPFOs"]}, **energy)
    return measure.check_observables({"observables": {
        "longitudinal_profile": {"scenarios": ["s"], "hit_collections": ["EcalBarrelCollectionRec"],
                                 "max_layer": 30, "cell_id_encoding": "system:5,module:3,"
                                 "stave:4,tower:4,layer:6,wafer:6,slice:4,cellX:32:-16,cellY:-16"},
        "energy": settings}})[0]


def measured(observables, reconstructed, out):
    measure.compute_simulator(observables, SIMULATOR, {"s": [reconstructed]},
                              measure.wanted(observables), out)
    with np.load(out) as data:
        return dict((key, data[key]) for key in data.files)


def test_the_reconstructed_energy_is_the_sum_of_the_reconstructed_particles(reconstructed,
                                                                             tmp_path):
    data = measured(configured(), reconstructed, str(tmp_path / "geant4.npz"))
    np.testing.assert_allclose(data["energy__s__reconstructed"], [19.5, 0.0])
    np.testing.assert_allclose(data["energy__s__true"], [20.0, 20.0])
    assert list(data["s__n_events"]) == [2] and list(data["s__n_no_primary"]) == [1]
    # measured along with the profiles
    assert data["longitudinal_profile__s__layers__sum"][0][1] == pytest.approx(0.2)


def test_only_the_pdg_codes_asked_for_count_with_either_sign(reconstructed, tmp_path):
    out = str(tmp_path / "geant4.npz")
    assert list(measured(configured(pdg=[22]), reconstructed, out)
                ["energy__s__reconstructed"]) == [19.0, 0.0]
    assert list(measured(configured(pdg=[11]), reconstructed, out)
                ["energy__s__reconstructed"]) == [0.5, 0.0]


def test_measurements_are_made_again_when_the_energy_settings_change(reconstructed, tmp_path):
    out = str(tmp_path / "geant4.npz")
    observables = configured()
    measured(observables, reconstructed, out)
    files = {"s": [reconstructed]}
    assert measure.is_up_to_date(out, measure.wanted(observables), files) == (True, "")
    assert measure.is_up_to_date(out, measure.wanted(configured(pdg=[22])), files) == (
        False, "the settings of energy changed")
    # Measurements made before the energy was an observable lack it
    with np.load(out) as data:
        arrays = dict((key, data[key]) for key in data.files if not key.startswith("energy__"))
    np.savez_compressed(out, **arrays)
    assert measure.is_up_to_date(out, measure.wanted(observables), files) == (
        False, "geant4.npz has no energy__s__reconstructed")


def test_what_a_measurement_records_says_whether_it_is_up_to_date(reconstructed, tmp_path):
    """Whether or not the reconstructed files are still there (or ever were, if
    the .npz file was fetched), but not once one that is there is newer."""
    observables = configured()
    wanted = measure.wanted(observables)
    relative = Layout(str(tmp_path)).relative
    out = str(tmp_path / "observables" / "geant4.npz")
    measure.compute_simulator(observables, SIMULATOR, {"s": [reconstructed]}, wanted, out,
                              relative)
    files = {"s": [reconstructed]}
    assert measure.is_up_to_date(out, wanted, files, relative) == (True, "")
    later = os.path.getmtime(out) + 10
    os.utime(reconstructed, (later, later))
    assert measure.is_up_to_date(out, wanted, files, relative) == (
        False, "files of s were reconstructed again since")
    os.remove(reconstructed)
    assert measure.is_up_to_date(out, wanted, files, relative) == (True, "")
    assert measure.is_up_to_date(out, wanted, {"s": [str(tmp_path / "other_REC.edm4hep.root")]},
                                 relative) == (False, "the reconstructed files of s changed")


def test_full_paths_recorded_in_another_data_folder_are_the_same_files():
    relative = Layout("/here/data").relative
    paths = ["/here/data/rec_output/s/geant4/%s_REC.edm4hep.root" % name for name in "ab"]
    assert measure.same_files(["rec_output/s/geant4/%s_REC.edm4hep.root" % name
                               for name in "ab"], paths, relative)
    assert measure.same_files(paths, paths, relative)
    elsewhere = ["/data/dust/user/someone/data/rec_output/s/geant4/%s_REC.edm4hep.root" % name
                 for name in "ab"]
    assert measure.same_files(elsewhere, paths, relative)
    assert not measure.same_files(elsewhere[::-1], paths, relative)
    assert not measure.same_files(elsewhere[:1], paths, relative)
    assert not measure.same_files(["/data/old_rec_output/s/geant4/a_REC.edm4hep.root"],
                                  paths[:1], relative)


def test_a_missing_particle_collection_says_what_to_set(reconstructed, tmp_path):
    with pytest.raises(measure.ObservableError) as error:
        measured(configured(particle_collections=["MyPFOs"]), reconstructed,
                 str(tmp_path / "geant4.npz"))
    assert "has no MyPFOs.energy" in str(error.value)
    assert "observables.energy.particle_collections" in str(error.value)
    assert not os.path.exists(str(tmp_path / "geant4.npz"))


def test_the_energy_settings_are_checked():
    for settings, message in (
            ({"scenarios": ["s"]}, "observables.energy needs particle_collections"),
            ({"scenarios": ["s"], "particle_collections": "PandoraPFOs"},
             "observables.energy needs particle_collections"),
            ({"scenarios": ["s"], "particle_collections": ["PandoraPFOs"], "pdg": 22},
             "observables.energy.pdg should be a list of PDG codes"),
            ({"particle_collections": ["PandoraPFOs"]},
             "observables.energy needs a list of scenarios")):
        with pytest.raises(measure.ObservableError) as error:
            measure.check_observables({"observables": {"energy": settings}})
        assert message in str(error.value)
    with pytest.raises(measure.ObservableError) as error:
        measure.check_observables({"observables": {"energy_resolution": {"scenarios": ["s"]}}})
    assert str(error.value).endswith(
        "radial_profile, longitudinal_profile, energy, intrinsic_angle, photon_separation, pi0")


# ---------------------------------------------------------------------------
# The intrinsic angles: the axes of the showers, from energy-weighted PCAs of
# their hits, against the incident particles' directions


def direction(theta, phi):
    """The unit vector of polar angle theta and azimuth phi, in degrees."""
    theta, phi = np.radians(theta), np.radians(phi)
    return np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])


def towards_theta(theta, phi):
    """The unit vector in which theta grows, at theta and phi."""
    theta, phi = np.radians(theta), np.radians(phi)
    return np.array([np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), -np.sin(theta)])


def towards_phi(theta, phi):
    """The unit vector in which phi grows, at phi."""
    phi = np.radians(phi)
    return np.array([-np.sin(phi), np.cos(phi), 0.0])


def shower(theta, phi, depths=100):
    """[(energy, position)] of a made-up shower of a photon from the IP along
    theta and phi, from the ECAL's front face (y = 1804.7 mm): a hit on its
    line of flight every 2 mm in depth, its energy rising and falling with the
    depth, and a pair of hits either side of it in each of the two directions
    across it, each with a tenth of that energy. It is symmetric about the
    line, so the axis of its hits, or of any most energetic of them, is the
    line."""
    line = direction(theta, phi)
    start = line * 1804.7 / line[1]
    across = (towards_theta(theta, phi), towards_phi(theta, phi))
    hits = []
    for depth in 2.0 * np.arange(depths):
        energy = 0.01 * (depth / 30) ** 2 * np.exp(-depth / 30) + 1e-5
        centre = start + depth * line
        hits.append((energy, centre))
        for side in across:
            hits += [(0.1 * energy, centre + 3.0 * side), (0.1 * energy, centre - 3.0 * side)]
    return hits


def flattened(showers):
    """event, energy and position of the hits of the showers ([[(energy, position)]])."""
    event = np.repeat(np.arange(len(showers)), [len(hits) for hits in showers])
    energy = np.array([e for hits in showers for e, _ in hits], dtype=float)
    position = np.array([p for hits in showers for _, p in hits], dtype=float).reshape(-1, 3)
    return event, energy, position


def residuals(showers, directions, fraction):
    """dtheta and dphi of the axes of the showers' most energetic hits."""
    event, energy, position = flattened(showers)
    chosen = measure.most_energetic(event, energy, len(showers), fraction)
    axes = measure.principal_axes(event[chosen], energy[chosen], position[chosen], len(showers))
    return measure.angle_residuals(axes, np.array(directions))


ANGLES = [(43.5, 79.5), (70.0, 95.0), (90.0, 90.0), (112.0, 84.0), (136.5, 108.5)]


def test_the_axis_of_a_shower_is_its_line_of_flight():
    showers = [shower(theta, phi) for theta, phi in ANGLES]
    lines = [direction(theta, phi) for theta, phi in ANGLES]
    for fraction in (1.0, 0.04):
        dtheta, dphi = residuals(showers, lines, fraction)
        np.testing.assert_allclose(dtheta, 0, atol=1e-9)
        np.testing.assert_allclose(dphi, 0, atol=1e-9)
    # A principal axis has no sign of its own: the residuals are those of the
    # axis taken along the photon, either way round
    for sign in (1, -1):
        dtheta, dphi = measure.angle_residuals(sign * np.array(lines), np.array(lines))
        np.testing.assert_allclose(np.concatenate([dtheta, dphi]), 0, atol=1e-9)


def test_the_residuals_are_reconstructed_less_true_with_phi_from_minus_to_plus_180():
    axes = np.array([direction(71.0, 90.0), direction(90.0, -179.5), direction(90.0, 179.5)])
    dtheta, dphi = measure.angle_residuals(
        axes, np.array([direction(70.0, 90.0), direction(90.0, 179.5), direction(90.0, -179.5)]))
    np.testing.assert_allclose(dtheta, [1.0, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(dphi, [0.0, 1.0, -1.0], atol=1e-9)


def test_the_pca_weighs_the_hits_by_their_energies():
    """Five energetic hits along x, and two soft ones further out along y: the
    hits spread most along y, but their energy along x."""
    position = np.array([[x, 0.0, 0.0] for x in (-2, -1, 0, 1, 2)] + [[0, -3, 0], [0, 3, 0]])
    event = np.zeros(7, dtype=np.int64)
    (axis,) = measure.principal_axes(event, np.array([1.0] * 5 + [0.01] * 2), position, 1)
    np.testing.assert_allclose(np.abs(axis), [1, 0, 0], atol=1e-12)
    (axis,) = measure.principal_axes(event, np.ones(7), position, 1)
    np.testing.assert_allclose(np.abs(axis), [0, 1, 0], atol=1e-12)


def test_a_soft_hit_far_out_tilts_the_axis_of_all_the_hits_but_not_of_the_top_4_percent():
    """As in the paper: hits far from the core (e.g. from backscatter) tilt the
    axis of all the hits, and the most energetic 4% leave them out."""
    theta, phi = 70.0, 95.0
    line = direction(theta, phi)
    hits = shower(theta, phi)
    start = line * 1804.7 / line[1]
    # Deep in the shower and 1 m off it in theta, with 0.1 MeV, far less than
    # the hits of the core
    hits.append((1e-4, start + 150 * line + 1000 * towards_theta(theta, phi)))
    dtheta, dphi = residuals([hits], [line], 1.0)
    assert dtheta[0] > 0.05 and abs(dphi[0]) < 1e-9
    dtheta, dphi = residuals([hits], [line], 0.04)
    assert abs(dtheta[0]) < 1e-9 and abs(dphi[0]) < 1e-9


def test_the_most_energetic_fraction_of_each_events_hits():
    energies = [np.arange(1.0, 101), np.arange(1.0, 31), np.arange(1.0, 11), [5.0],
                np.arange(1.0, 101), [2.0, 7.0, 7.0, 7.0, 1.0] * 20]
    event = np.repeat(np.arange(len(energies)), [len(e) for e in energies])
    energy = np.concatenate(energies)

    def kept(fraction):
        chosen = measure.most_energetic(event, energy, len(energies), fraction)
        return [sorted(energy[chosen & (event == i)]) for i in range(len(energies))]

    top4 = kept(0.04)
    # 4% of 100 hits, of 30 rounded up, and at least two
    assert top4[0] == [97, 98, 99, 100] and top4[1] == [29, 30] and top4[2] == [9, 10]
    assert top4[3] == [5.0]
    # equal energies in the order of the hits
    chosen = measure.most_energetic(event, energy, len(energies), 0.04)
    assert list(np.flatnonzero(chosen & (event == 5)) - np.flatnonzero(event == 5)[0]) \
        == [1, 2, 3, 6]
    # 7% of 100 is 7 (0.07 * 100 is a little more than 7)
    assert kept(0.07)[4] == list(range(94, 101))
    assert [len(e) for e in kept(1.0)] == [len(e) for e in energies]


def test_showers_without_hits_at_two_points_have_no_axis():
    position = np.array([[0.0, 1800, 0], [0, 1800, 0], [0, 1800, 0], [0, 1810, 0], [0, 1820, 0],
                         [1, 1800, 0], [1, 1810, 0]])
    event = np.array([1, 2, 2, 3, 3, 4, 4])
    energy = np.array([1.0, 1.0, 2.0, 0.0, 0.0, 1.0, 3.0])
    axes = measure.principal_axes(event, energy, position, 5)
    # no hits, one, two at the same point, two without energy
    assert np.isnan(axes[:4]).all()
    np.testing.assert_allclose(np.abs(axes[4]), [0, 1, 0], atol=1e-12)


def write_showers(path, events):
    """A reconstructed file of events, each (generator status of its MC
    particle, its momentum, [(energy, position)] of its ECAL barrel hits) and
    its vertex, if not where the particle gun puts it: on the ECAL's front face
    (y = 1804.7 mm), on its line from the IP."""
    import awkward as ak
    import uproot
    events = [event if len(event) == 4 else
              tuple(event) + (np.asarray(event[1], dtype=float) * 1804.7 / event[1][1],)
              for event in events]
    columns = {"MCParticles.generatorStatus": [[status] for status, _, _, _ in events],
               "MCParticles.mass": [[0.0] for _ in events],
               "EcalBarrelCollectionRec.energy": [[e for e, _ in hits]
                                                  for _, _, hits, _ in events],
               "EcalBarrelCollectionRec.cellID": [[0] * len(hits) for _, _, hits, _ in events]}
    for i, c in enumerate("xyz"):
        columns["MCParticles.vertex." + c] = [[float(v[i])] for _, _, _, v in events]
        columns["MCParticles.momentum." + c] = [[float(p[i])] for _, p, _, _ in events]
        columns["EcalBarrelCollectionRec.position." + c] = [
            [float(position[i]) for _, position in hits] for _, _, hits, _ in events]
    types = {"MCParticles.generatorStatus": np.int32, "EcalBarrelCollectionRec.cellID": np.uint64}
    arrays = dict((name, ak.values_astype(ak.Array(values), types.get(name, np.float64)))
                  for name, values in columns.items())
    with uproot.recreate(path) as output:
        output.mktree("events", dict((name, array.type) for name, array in arrays.items()))
        output["events"].extend(arrays)
    return path


def configured_angles(**angles):
    settings = dict({"scenarios": ["s"], "hit_collections": ["EcalBarrelCollectionRec"],
                     "hit_fractions": [1.0, 0.04], "max_residual": 5, "n_bins": 100}, **angles)
    return measure.check_observables({"observables": {"intrinsic_angle": settings}})[0]


@pytest.fixture
def showers(tmp_path):
    """Two showers along their photons' lines of flight, a third tilted by a
    soft hit far out, one with a single hit, and one without the incident
    photon (generator status 1)."""
    (theta, phi), tilted = ANGLES[1], shower(*ANGLES[1])
    line = direction(theta, phi)
    tilted.append((1e-4, line * 1804.7 / line[1] + 150 * line + 1000 * towards_theta(theta, phi)))
    return write_showers(str(tmp_path / "photons_REC.edm4hep.root"), [
        (1, 20 * direction(*ANGLES[0]), shower(*ANGLES[0])),
        (1, 50 * direction(*ANGLES[4]), shower(*ANGLES[4])),
        (1, 80 * line, tilted),
        (1, 20 * direction(90, 90), [(1.0, (0.0, 1810.0, 0.0))]),
        (0, 20 * direction(90, 90), shower(90, 90)),
    ])


def test_the_intrinsic_angles_are_stored_per_event(showers, tmp_path):
    data = measured(configured_angles(), showers, str(tmp_path / "geant4.npz"))
    key = "intrinsic_angle__s__"
    assert sorted(k[len(key):] for k in data if k.startswith(key)) == [
        "dphi_all", "dphi_top4", "dtheta_all", "dtheta_top4", "phi_true", "settings",
        "theta_true"]
    # The event without the incident photon is left out
    assert list(data["s__n_events"]) == [4] and list(data["s__n_no_primary"]) == [1]
    np.testing.assert_allclose(data[key + "theta_true"], [43.5, 136.5, 70, 90], atol=1e-9)
    np.testing.assert_allclose(data[key + "phi_true"], [79.5, 108.5, 95, 90], atol=1e-9)
    # The showers along their lines of flight, and the tilted one, whose top 4% isn't
    for hits in ("all", "top4"):
        np.testing.assert_allclose(data[key + "dphi_" + hits][:3], 0, atol=1e-9)
        np.testing.assert_allclose(data[key + "dtheta_" + hits][:2], 0, atol=1e-9)
    assert data[key + "dtheta_all"][2] > 0.05 and abs(data[key + "dtheta_top4"][2]) < 1e-9
    # and a shower of one hit has no axis
    for what in ("dtheta_all", "dphi_all", "dtheta_top4", "dphi_top4"):
        assert np.isnan(data[key + what][3])


def test_the_bounding_box_is_centred_on_the_line_of_flight_in_every_layer():
    """In each layer, a plane of constant y, the box is a square centred on
    where the photon's line of flight crosses it; in depth it reaches half its
    side either side of the vertex."""
    vertex = np.array([[0.0, 1800.0, -50.0], [100.0, 1800.0, 0.0]])
    # The first photon heads straight across the layers, the second slanted,
    # going 0.5 mm in x and -1.5 mm in z for each mm in y
    direction = np.array([[0.0, 1.0, 0.0], np.array([0.5, 1.0, -1.5]) / np.sqrt(3.5)])
    hits = [(0, (400.0, 2200.0, -450.0)),  # on a corner of the first event's box: in it
            (0, (400.5, 1800.0, -50.0)),   # beyond its edge in x,
            (0, (0.0, 1399.5, -50.0)),     # too far in front of the vertex,
            (0, (0.0, 1800.0, 350.5)),     # beyond its edge in z
            (0, (450.0, 1800.0, -50.0)),   # in the second event's box, not its own
            (1, (450.0, 1800.0, 0.0)),     # in its own
            # 300 mm deep: on the line of flight, 450 mm from the vertex in z,
            (1, (250.0, 2100.0, -450.0)),
            (1, (250.0, 2100.0, -60.0)),   # 390 mm from the line in z: in,
            (1, (100.0, 2100.0, 0.0))]     # 450 mm from it, though at the vertex's z
    event = np.array([e for e, _ in hits])
    position = np.array([p for _, p in hits])
    np.testing.assert_array_equal(
        measure.in_bounding_box(event, position, vertex, direction, 800.0),
        [True, False, False, False, False, True, True, True, False])
    # A particle not heading across the layers has no hit in its box
    assert not measure.in_bounding_box(event[:1], position[:1], vertex,
                                       np.array([[1.0, 0.0, 0.0]] * 2), 800.0).any()


def test_the_bounding_box_keeps_the_hits_either_side_of_the_axis_alike(tmp_path):
    """Each layer of the box is shifted along the shower's axis: two soft hits
    in the last layer, either side of the axis and 390 mm from it, are both in
    the box, though one is further than half its side from the vertex in x
    (which a box about the vertex would leave out, tilting the axis)."""
    theta, phi = ANGLES[0]  # the most slanted
    line = direction(theta, phi)
    start = line * 1804.7 / line[1]
    # Across the line of flight, in a layer
    across = np.array([line[2], 0.0, -line[0]]) / np.hypot(line[2], line[0])
    centre = start + 190 / line[1] * line  # where the line crosses the layer 190 mm deep
    pair = [centre + 390 * across, centre - 390 * across]
    assert abs(pair[0] - start)[0] > 400 and (abs(pair[1] - start) < 400).all()
    path = write_showers(str(tmp_path / "photons_REC.edm4hep.root"),
                         [(1, 50 * line, shower(theta, phi) + [(1e-4, hit) for hit in pair])])
    data = measured(configured_angles(bounding_box=800), path, str(tmp_path / "geant4.npz"))
    for what in ("dtheta_all", "dphi_all", "dtheta_top4", "dphi_top4"):
        np.testing.assert_allclose(data["intrinsic_angle__s__" + what], 0, atol=1e-9)


def test_the_bounding_box_leaves_out_the_hits_far_from_the_shower(tmp_path):
    """As the 800 mm box of the paper's optimal shower generators does: only
    the hits in a box about the photon's line of flight go into the PCAs, so a
    soft hit on the other side of the detector (e.g. from backscatter) doesn't
    tilt the axis of all the hits, and an energetic one off the shower isn't
    among the top 4% of those left."""
    theta, phi = ANGLES[1]
    line = direction(theta, phi)
    start = line * 1804.7 / line[1]
    soft, hard = shower(theta, phi), shower(theta, phi)
    # 0.1 MeV in the bottom stave, below where the photon enters the top one
    soft.append((1e-4, (start[0], -1850.0, start[2])))
    # 1 GeV 1 m off the shower in theta, more than all the rest of it
    hard.append((1.0, start + 150 * line + 1000 * towards_theta(theta, phi)))
    path = write_showers(str(tmp_path / "photons_REC.edm4hep.root"),
                         [(1, 50 * line, soft), (1, 50 * line, hard)])
    key = "intrinsic_angle__s__"
    data = measured(configured_angles(), path, str(tmp_path / "every_hit.npz"))
    assert abs(data[key + "dtheta_all"][0]) > 1 and abs(data[key + "dtheta_top4"][0]) < 1e-9
    assert abs(data[key + "dtheta_all"][1]) > 1 and abs(data[key + "dtheta_top4"][1]) > 1
    data = measured(configured_angles(bounding_box=800), path, str(tmp_path / "in_box.npz"))
    for what in ("dtheta_all", "dphi_all", "dtheta_top4", "dphi_top4"):
        np.testing.assert_allclose(data[key + what], 0, atol=1e-9)
    assert json.loads(str(data[key + "settings"]))["bounding_box"] == 800


def test_a_shower_with_none_of_its_hits_in_its_box_has_no_axis(tmp_path, capsys):
    """The box reaches half its side either side of the vertex, so a photon
    started at the IP rather than on the ECAL's face has none of its hits in
    it: it has no axis, and that's said."""
    line = direction(*ANGLES[2])
    path = write_showers(str(tmp_path / "photons_REC.edm4hep.root"), [
        (1, 20 * line, shower(*ANGLES[2])),
        (1, 20 * line, shower(*ANGLES[2]), (0.0, 0.0, 0.0)),
        (1, 20 * line, [])])
    data = measured(configured_angles(bounding_box=800), path, str(tmp_path / "geant4.npz"))
    for what in ("dtheta_all", "dphi_all", "dtheta_top4", "dphi_top4"):
        values = data["intrinsic_angle__s__" + what]
        assert abs(values[0]) < 1e-9 and np.isnan(values[1:]).all(), (what, values)
    # (the shower without hits has no axis with or without a box)
    assert "warning: 1 of the 2 showers with hits have none in the bounding box (800 mm about " \
        "the incident particle's line of flight), so they have no axis" in capsys.readouterr().err


def test_changing_how_the_residuals_are_drawn_needs_no_measuring_again(showers, tmp_path):
    out = str(tmp_path / "geant4.npz")
    measured(configured_angles(), showers, out)
    files = {"s": [showers]}
    # (nor does a bounding box of null, every hit, as before there was one)
    for drawn in ({"max_residual": [10, 2]}, {"n_bins": 40}, {"metric": "l1"},
                  {"bounding_box": None}):
        assert measure.is_up_to_date(out, measure.wanted(configured_angles(**drawn)), files) \
            == (True, ""), drawn
    for changed, why in (({"hit_fractions": [1.0, 0.1]},
                          "geant4.npz has no intrinsic_angle__s__dtheta_top10"),
                         ({"hit_collections": ["EcalBarrelCollectionRec",
                                               "EcalEndcapsCollectionRec"]},
                          "the settings of intrinsic_angle changed"),
                         ({"bounding_box": 800}, "the settings of intrinsic_angle changed")):
        assert measure.is_up_to_date(out, measure.wanted(configured_angles(**changed)), files) \
            == (False, why)


def test_the_intrinsic_angle_settings_are_checked():
    assert measure.residual_binning("intrinsic_angle", {"hit_fractions": [1.0, 0.04, 0.025],
                                                        "max_residual": [10, 5, 5]}) == {
        "all": (-10, 10, 100), "top4": (-5, 5, 100), "top2.5": (-5, 5, 100)}
    assert [measure.bounding_box("intrinsic_angle", settings) for settings in (
        {"bounding_box": 800}, {"bounding_box": 512.5}, {"bounding_box": None}, {})] == [
        800.0, 512.5, None, None]
    for side in (0, -800, True, "800", [800], float("nan")):
        with pytest.raises(measure.ObservableError) as error:
            configured_angles(bounding_box=side)
        assert "observables.intrinsic_angle.bounding_box should be the side, in mm, of the box " \
            "about each shower that its hits must be in (e.g. 800), or null" in str(error.value)
    for settings, message in (
            ({"hit_collections": None}, "observables.intrinsic_angle needs hit_collections"),
            ({"hit_fractions": None}, "observables.intrinsic_angle.hit_fractions should be a "
                                      "number or a list of numbers"),
            ({"hit_fractions": [1.0, 0]}, "hit_fractions should be fractions of the hits"),
            ({"hit_fractions": [1.5]}, "hit_fractions should be fractions of the hits"),
            ({"hit_fractions": [0.04, 0.04]}, "hit_fractions has a fraction twice: 0.04, 0.04"),
            ({"max_residual": -5}, "max_residual and n_bins need a positive number each"),
            ({"n_bins": [100, 50, 20]}, "or one for each of hit_fractions")):
        with pytest.raises(measure.ObservableError) as error:
            configured_angles(**settings)
        assert message in str(error.value), (settings, str(error.value))


# ---------------------------------------------------------------------------
# The di-photon separation: of the two incident photons, on the front face of
# the ECAL barrel, against the number of reconstructed photons

FACE = 1804.7


def on_face(x, z, stave_phi=90.0):
    """A point on the front face of the stave centred at stave_phi (degrees):
    x along the face, across the beam, and z along it."""
    phi = np.radians(stave_phi)
    normal = np.array([np.cos(phi), np.sin(phi), 0.0])
    along = np.array([np.sin(phi), -np.cos(phi), 0.0])
    return FACE * normal + x * along + np.array([0.0, 0.0, z])


def from_ip(point, energy=20.0, start=None):
    """An incident photon flying from the IP through the point, starting there
    (or at start): (generator status, vertex, momentum)."""
    point = np.asarray(point, dtype=float)
    return (1, point if start is None else np.asarray(start, dtype=float),
            energy * point / np.linalg.norm(point))


def write_pairs(path, events):
    """A reconstructed file of events, each ([(generator status, vertex,
    momentum)] of its MC particles, [PDG code] of its reconstructed particles)."""
    import awkward as ak
    import uproot
    columns = {"MCParticles.generatorStatus": [[s for s, _, _ in mc] for mc, _ in events],
               "MCParticles.mass": [[0.0] * len(mc) for mc, _ in events],
               "PandoraPFOs.energy": [[10.0] * len(pfos) for _, pfos in events],
               "PandoraPFOs.PDG": [list(pfos) for _, pfos in events]}
    for i, c in enumerate("xyz"):
        columns["MCParticles.vertex." + c] = [[float(v[i]) for _, v, _ in mc] for mc, _ in events]
        columns["MCParticles.momentum." + c] = [[float(p[i]) for _, _, p in mc]
                                                for mc, _ in events]
    types = {"MCParticles.generatorStatus": np.int32, "PandoraPFOs.PDG": np.int32,
             "PandoraPFOs.energy": np.float32}
    arrays = dict((name, ak.values_astype(ak.Array(values), types.get(name, np.float64)))
                  for name, values in columns.items())
    with uproot.recreate(path) as output:
        output.mktree("events", dict((name, array.type) for name, array in arrays.items()))
        output["events"].extend(arrays)
    return path


def configured_separation(**separation):
    settings = dict({"scenarios": ["s"], "particle_collections": ["PandoraPFOs"], "pdg": [22]},
                    **separation)
    return measure.check_observables({"observables": {"photon_separation": settings}})[0]


@pytest.fixture
def pairs_file(tmp_path):
    secondary = (0, on_face(5.0, 75.0), np.array([0.0, 0.1, 0.0]))
    return write_pairs(str(tmp_path / "pairs_REC.edm4hep.root"), [
        # Two photons starting on the face, 30 mm apart, reconstructed as two
        # photons and a neutron, with a secondary of the simulation
        ([from_ip(on_face(-15.0, 70.0)), from_ip(on_face(15.0, 70.0)), secondary],
         [22, 22, 2112]),
        # Two starting at the IP, which reach the face 40 mm apart, as one
        ([from_ip(on_face(0.0, 50.0), start=(0, 0, 0)),
          from_ip(on_face(0.0, 90.0), start=(0, 0, 0))], [22]),
        # On the stave at phi = 0 (x = 1804.7), 12 mm apart, of 100 GeV
        ([from_ip(on_face(-6.0, 0.0, 0.0), 100.0), from_ip(on_face(6.0, 0.0, 0.0), 100.0)],
         [22, 22, 22]),
        # One heading away from the face; then only one incident photon; then three
        ([from_ip(on_face(0.0, 70.0)), (1, on_face(10.0, 70.0), np.array([0.0, -20.0, 0.0]))],
         [22]),
        ([from_ip(on_face(0.0, 70.0))], [22, 22]),
        ([from_ip(on_face(0.0, 70.0)), from_ip(on_face(9.0, 70.0)), from_ip(on_face(18, 70.0))],
         []),
    ])


def test_the_separation_is_of_the_two_photons_on_the_ecal_face(pairs_file, tmp_path):
    data = measured(configured_separation(), pairs_file, str(tmp_path / "geant4.npz"))
    key = "photon_separation__s__"
    assert sorted(k[len(key):] for k in data if k.startswith(key)) == [
        "energy", "n_photons", "separation", "settings"]
    separation = data[key + "separation"]
    np.testing.assert_allclose(separation[:3], [30.0, 40.0, 12.0], rtol=1e-9)
    # Without two photons heading into the face, there is none
    assert np.isnan(separation[3:]).all()
    np.testing.assert_allclose(data[key + "energy"][:3], [20.0, 20.0, 100.0])
    # the photons (PDG 22) among the reconstructed particles
    assert list(data[key + "n_photons"]) == [2, 1, 3, 1, 2, 0]
    assert list(data["s__n_events"]) == [6] and list(data["s__seeds"]) == [-1]


def test_the_separation_crosses_the_face_of_the_stave_the_pair_heads_into():
    """Photons from the IP crossing the face at known points, on each of the
    eight staves, whether they start there or at the IP."""
    for stave_phi in np.arange(0, 360, 45.0):
        points = np.array([[on_face(-20.0, 30.0, stave_phi), on_face(10.0, 70.0, stave_phi)]])
        expected = np.linalg.norm(points[0, 0] - points[0, 1])
        momentum = 50 * points / np.linalg.norm(points, axis=2, keepdims=True)
        for vertex in (points, np.zeros_like(points), 0.5 * points):
            separation = measure.separations_on_surface(vertex, momentum, FACE, 8)
            np.testing.assert_allclose(separation, [expected], rtol=1e-12)
    # A pair that isn't a pair (NaN) has none
    assert np.isnan(measure.separations_on_surface(np.full((1, 2, 3), np.nan),
                                                   np.full((1, 2, 3), np.nan), FACE, 8)).all()


def test_drawing_the_separation_differently_needs_no_measuring_again(pairs_file, tmp_path):
    out = str(tmp_path / "geant4.npz")
    measured(configured_separation(), pairs_file, out)
    files = {"s": [pairs_file]}
    for drawn in ({"max_separation": 60}, {"n_bins": 30}, {"metric": "l1"}):
        assert measure.is_up_to_date(out, measure.wanted(configured_separation(**drawn)),
                                     files) == (True, ""), drawn
    assert measure.is_up_to_date(out, measure.wanted(configured_separation(surface_distance=1460)),
                                 files) == (False, "the settings of photon_separation changed")
    # Measurements made before the seeds were recorded lack them
    with np.load(out) as data:
        arrays = dict((key, data[key]) for key in data.files if key != "s__seeds")
    np.savez_compressed(out, **arrays)
    assert measure.is_up_to_date(out, measure.wanted(configured_separation()), files) == (
        False, "geant4.npz has no s__seeds")


def test_each_files_seed_is_recorded(pairs_file, tmp_path):
    observables = configured_separation()
    out = str(tmp_path / "geant4.npz")
    measure.compute_simulator(observables, SIMULATOR, {"s": [pairs_file, pairs_file]},
                              measure.wanted(observables), out, seeds={"s": [3, None]})
    with np.load(out) as data:
        assert list(data["s__seeds"]) == [3, -1] and list(data["s__n_events"]) == [6, 6]
    with pytest.raises(measure.ObservableError, match="1 seeds for the 2 reconstructed files"):
        measure.compute_simulator(observables, SIMULATOR, {"s": [pairs_file, pairs_file]},
                                  measure.wanted(observables), out, seeds={"s": [3]})


def test_the_separation_settings_are_checked():
    assert measure.separation_binning("photon_separation", {}) == (0.0, 90.0, 18)
    assert measure.separation_binning("photon_separation",
                                      {"max_separation": 60, "n_bins": 12}) == (0.0, 60.0, 12)
    assert measure.surface("photon_separation", {}) == (1804.7, 8)
    for settings, message in (
            ({"particle_collections": None}, "observables.photon_separation needs "
                                             "particle_collections"),
            ({"surface_distance": -1804.7}, "surface_distance should be the distance"),
            ({"surface_distance": "far"}, "surface_distance should be the distance"),
            ({"barrel_symmetry": 2}, "barrel_symmetry should be the number of staves"),
            ({"barrel_symmetry": 8.5}, "barrel_symmetry should be the number of staves"),
            ({"max_separation": 0}, "max_separation and n_bins should be a positive number"),
            ({"n_bins": [10, 20]}, "max_separation and n_bins should be a positive number")):
        with pytest.raises(measure.ObservableError) as error:
            configured_separation(**settings)
        assert message in str(error.value), (settings, str(error.value))


# ---------------------------------------------------------------------------
# The pi0s of tau decays: the pi0 candidates of the reconstruction, matched to
# the MC pi0s through their photons' links


PI0_MASS = 0.1349768


def photon_pair(energy, mass=PI0_MASS, share=0.6, theta=80.0, phi=95.0):
    """The momenta of two photons sharing the energy, with the invariant mass,
    about the direction of theta and phi."""
    e1, e2 = share * energy, (1 - share) * energy
    opening = np.arccos(1 - mass ** 2 / (2 * e1 * e2))
    axis, across = direction(theta, phi), towards_theta(theta, phi)
    # each turned from the axis so that the pair's momentum is along it
    a1 = np.arctan2(e2 * np.sin(opening), e1 + e2 * np.cos(opening))
    d1 = np.cos(a1) * axis + np.sin(a1) * across
    d2 = np.cos(opening - a1) * axis - np.sin(opening - a1) * across
    return e1 * d1, e2 * d2


def mc(pdg, status, momentum, mass=0.0, parents=(), daughters=()):
    return {"PDG": pdg, "status": status, "momentum": np.asarray(momentum, dtype=float),
            "mass": mass, "parents": list(parents), "daughters": list(daughters)}


def tau_decay(pi0_momentum, photons, status=2):
    """The MC particles of a tau- decaying to a rho- that decays into a pi- and
    a pi0, decaying into the two photons: [tau, nu, rho, pi-, pi0, photon,
    photon] (indices 0 to 6)."""
    return [mc(15, status, [0, 0, 120.0], 1.777, daughters=[1, 2]),
            mc(16, 1, [0, 0, 20.0], parents=[0]),
            mc(-213, 2, [0, 0, 100.0], 0.775, parents=[0], daughters=[3, 4]),
            mc(-211, 1, [0, 0, 40.0], 0.1396, parents=[2]),
            mc(111, 2, pi0_momentum, PI0_MASS, parents=[2], daughters=[5, 6]),
            mc(22, 1, photons[0], parents=[4]), mc(22, 1, photons[1], parents=[4])]


def write_pi0_events(path, events):
    """A reconstructed file of events, each {"mc": [mc()], "pfos": [(energy,
    momentum, PDG)], "candidates": [(energy, momentum, [pfo, pfo])], "links":
    [(weight, pfo, mc)], "reverse": [(weight, pfo, mc)]}, as podio writes them
    (the relations by begin and end in each event's list of indices)."""
    import awkward as ak
    import uproot
    columns = {}

    def add(name, values):
        columns.setdefault(name, []).append(values)

    for event in events:
        particles = event["mc"]
        add("MCParticles.PDG", [p["PDG"] for p in particles])
        add("MCParticles.generatorStatus", [p["status"] for p in particles])
        add("MCParticles.mass", [p["mass"] for p in particles])
        for i, c in enumerate("xyz"):
            add("MCParticles.vertex." + c, [0.0] * len(particles))
            add("MCParticles.momentum." + c, [float(p["momentum"][i]) for p in particles])
        for relation in ("parents", "daughters"):
            ends = np.cumsum([0] + [len(p[relation]) for p in particles])
            add("MCParticles.%s_begin" % relation, ends[:-1].tolist())
            add("MCParticles.%s_end" % relation, ends[1:].tolist())
            add("_MCParticles_%s.index" % relation, [i for p in particles for i in p[relation]])
        add("PandoraPFOs.energy", [e for e, _, _ in event["pfos"]])
        add("PandoraPFOs.PDG", [pdg for _, _, pdg in event["pfos"]])
        candidates = event["candidates"]
        add("GammaGammaCandidatePi0s.energy", [e for e, _, _ in candidates])
        for i, c in enumerate("xyz"):
            add("PandoraPFOs.momentum." + c, [float(p[i]) for _, p, _ in event["pfos"]])
            add("GammaGammaCandidatePi0s.momentum." + c, [float(p[i]) for _, p, _ in candidates])
        ends = np.cumsum([0] + [len(made_of) for _, _, made_of in candidates])
        add("GammaGammaCandidatePi0s.particles_begin", ends[:-1].tolist())
        add("GammaGammaCandidatePi0s.particles_end", ends[1:].tolist())
        add("_GammaGammaCandidatePi0s_particles.index",
            [i for _, _, made_of in candidates for i in made_of])
        for name, key in (("RecoMCTruthLink", "links"), ("MCTruthRecoLink", "reverse")):
            add(name + ".weight", [float(w) for w, _, _ in event[key]])
            add("_%s_from.index" % name, [pfo for _, pfo, _ in event[key]])
            add("_%s_to.index" % name, [particle for _, _, particle in event[key]])
    types = {}
    for name in columns:
        if name.endswith(("_begin", "_end")):
            types[name] = np.uint32
        elif name.endswith((".index", ".PDG", "generatorStatus")):
            types[name] = np.int32
        elif name.startswith(("PandoraPFOs", "GammaGamma")) or name.endswith(".weight"):
            types[name] = np.float32
    arrays = dict((name, ak.values_astype(ak.Array(values), types.get(name, np.float64)))
                  for name, values in columns.items())
    with uproot.recreate(path) as output:
        output.mktree("events", dict((name, array.type) for name, array in arrays.items()))
        output["events"].extend(arrays)
    return path


def link(cluster, track=0.0):
    """A link's weight, as ILD's RecoMCTruthLinker encodes it."""
    return 10000 * int(round(1000 * cluster)) + int(round(1000 * track))


@pytest.fixture
def pi0_file(tmp_path):
    """Four events: (a) a pi0 of a tau decay reconstructed as two photons and
    found (good); one of them paired with a photon split off the pi-'s shower
    (confused); and a pair of the photons of a pi0 made by the simulation and of
    one with no tau ancestor (fake); (b) a pi0 whose first photon's MC photon
    is best matched, the other way, by a fragment (confused, but good without
    the reverse links); (c) a pi0 not found; (d) an event without an incident
    particle (generator status 1), which is left out with its candidate."""
    g1, g2 = photon_pair(30.0)
    pi0 = g1 + g2
    a = tau_decay(pi0, (g1, g2))
    # a pi0 made by the simulation (generator status 0), and one with no tau
    a += [mc(111, 0, [5.0, 0, 0], PI0_MASS, daughters=[8, 9]),
          mc(22, 0, [3.0, 0, 0], parents=[7]), mc(22, 0, [2.0, 0, 0], parents=[7]),
          mc(111, 2, [0, 5.0, 0], PI0_MASS, daughters=[11, 12]),
          mc(22, 1, [0, 3.0, 0], parents=[10]), mc(22, 1, [0, 2.0, 0], parents=[10])]
    r1, r2 = 0.97 * g1, 1.01 * g2  # reconstructed a little off
    split = 0.05 * a[3]["momentum"]
    event_a = {"mc": a,
               "pfos": [(40.0, a[3]["momentum"], -211), (np.linalg.norm(r1), r1, 22),
                        (np.linalg.norm(r2), r2, 22), (3.0, np.array([3.0, 0, 0]), 22),
                        (3.0, np.array([0, 3.0, 0]), 22), (2.0, split, 22)],
               "candidates": [(29.5, 29.5 * pi0 / np.linalg.norm(pi0), [1, 2]),
                              (20.0, r1 + split, [1, 5]),
                              (6.0, np.array([3.0, 3.0, 0]), [3, 4])],
               "links": [(link(1.0, 1.0), 0, 3), (link(0.96), 1, 5), (link(0.03), 1, 6),
                         (link(0.95), 2, 6), (link(0.04), 2, 5), (link(0.99), 3, 8),
                         (link(0.99), 4, 11), (link(0.9), 5, 3)],
               "reverse": [(link(0.97, 1.0), 0, 3), (link(0.02), 5, 3), (link(0.96), 1, 5),
                           (link(0.95), 2, 6), (link(0.99), 3, 8), (link(0.99), 4, 11)]}
    h1, h2 = photon_pair(20.0, share=0.5, theta=100.0, phi=80.0)
    b = tau_decay(h1 + h2, (h1, h2))
    event_b = {"mc": b,
               "pfos": [(0.4 * 20, 0.4 * h1 / 0.5, 22), (np.linalg.norm(h2), h2, 22),
                        (6.0, 6.0 * h1 / np.linalg.norm(h1), 22)],
               "candidates": [(19.0, h1 + h2, [0, 1])],
               "links": [(link(0.9), 0, 5), (link(0.95), 1, 6), (link(0.97), 2, 5)],
               "reverse": [(link(0.3), 0, 5), (link(0.6), 2, 5), (link(0.95), 1, 6)]}
    c1, c2 = photon_pair(10.0)
    event_c = {"mc": tau_decay(c1 + c2, (c1, c2)), "pfos": [], "candidates": [], "links": [],
               "reverse": []}
    no_primary = tau_decay(c1 + c2, (c1, c2))
    for particle in no_primary:
        particle["status"] = 2
    event_d = {"mc": no_primary, "pfos": [(10.0, c1 + c2, 22), (1.0, c1, 22)],
               "candidates": [(10.0, c1 + c2, [0, 1])], "links": [], "reverse": []}
    return write_pi0_events(str(tmp_path / "taus_REC.edm4hep.root"),
                            [event_a, event_b, event_c, event_d])


def configured_pi0(**pi0):
    settings = dict({"scenarios": ["s"], "candidate_collection": "GammaGammaCandidatePi0s",
                     "particle_collection": "PandoraPFOs", "link_collection": "RecoMCTruthLink",
                     "reverse_link_collection": "MCTruthRecoLink", "ancestor_pdg": [15]}, **pi0)
    return measure.check_observables({"observables": {"pi0": settings}})[0]


def test_the_pi0_candidates_are_matched_to_the_mc_pi0s_through_their_photons(pi0_file,
                                                                             tmp_path):
    data = measured(configured_pi0(), pi0_file, str(tmp_path / "geant4.npz"))
    key = "pi0__s__"
    assert sorted(k[len(key):] for k in data if k.startswith(key)) == sorted(
        measure.PI0_EVENT_ITEMS + measure.PI0_CANDIDATE_ITEMS + ("settings",))
    # The event without an incident particle is left out, with its candidate
    assert list(data["s__n_events"]) == [3] and list(data["s__n_no_primary"]) == [1]
    assert list(data[key + "n_candidates"]) == [3, 1, 0]
    # Only the pi0s of the generator, of tau decays, into two photons
    assert list(data[key + "n_true"]) == [1, 1, 1]
    good, confused, fake = measure.GOOD, measure.CONFUSED, measure.FAKE
    assert list(data[key + "match"]) == [good, confused, fake, confused]
    assert list(data[key + "true_index"]) == [0, 0, -1, 0]
    # The invariant mass of the two photons, as reconstructed
    e1, e2 = 0.97 * 18.0, 1.01 * 12.0
    expected = PI0_MASS * np.sqrt(0.97 * 1.01)  # each photon's energy scaled
    np.testing.assert_allclose(data[key + "mass"][0], expected, rtol=1e-5)
    np.testing.assert_allclose(data[key + "photon_energy"][0], e1 + e2, rtol=1e-5)
    np.testing.assert_allclose(data[key + "energy"], [29.5, 20.0, 6.0, 19.0], rtol=1e-6)
    true_energy = np.sqrt(np.sum(np.square(sum(photon_pair(30.0)))) + PI0_MASS ** 2)
    np.testing.assert_allclose(data[key + "true_energy"][:2], true_energy, rtol=1e-6)
    assert np.isnan(data[key + "true_energy"][2]) and np.isnan(data[key + "true_theta"][2])
    np.testing.assert_allclose(data[key + "theta"][0], data[key + "true_theta"][0], atol=1e-4)
    np.testing.assert_allclose(data[key + "true_theta"][0], 80.0, atol=1e-6)
    np.testing.assert_allclose(data[key + "true_phi"][0], 95.0, atol=1e-6)


def test_without_the_reverse_links_a_photon_needs_only_its_best_mc_particle(pi0_file, tmp_path):
    data = measured(configured_pi0(reverse_link_collection=None), pi0_file,
                    str(tmp_path / "geant4.npz"))
    assert list(data["pi0__s__match"]) == [measure.GOOD, measure.CONFUSED, measure.FAKE,
                                           measure.GOOD]


def test_without_ancestor_pdg_every_mc_pi0_of_the_generator_counts(pi0_file, tmp_path):
    data = measured(configured_pi0(ancestor_pdg=None), pi0_file, str(tmp_path / "geant4.npz"))
    # The pi0 without a tau too, but not the simulation's
    assert list(data["pi0__s__n_true"]) == [2, 1, 1]
    assert list(data["pi0__s__match"]) == [measure.GOOD, measure.CONFUSED, measure.CONFUSED,
                                           measure.CONFUSED]
    assert list(data["pi0__s__true_index"]) == [0, 0, 1, 0]


def test_drawing_the_pi0s_differently_needs_no_measuring_again(pi0_file, tmp_path):
    out = str(tmp_path / "geant4.npz")
    measured(configured_pi0(), pi0_file, out)
    files = {"s": [pi0_file]}
    for drawn in ({"reconstructed_energy": "photons"}, {"mass_range": [0.1, 0.17]},
                  {"mass_bins": 30}, {"energy_range": [-3, 3], "energy_bins": 12},
                  {"theta_range": [-1, 1], "theta_bins": 20},
                  {"phi_range": [-0.5, 0.5], "phi_bins": 25}, {"metric": "l1"}):
        assert measure.is_up_to_date(out, measure.wanted(configured_pi0(**drawn)), files) \
            == (True, ""), drawn
    assert measure.is_up_to_date(out, measure.wanted(configured_pi0(ancestor_pdg=[15, 24])),
                                 files) == (False, "the settings of pi0 changed")


def test_the_pi0_settings_are_checked():
    assert measure.pi0_binning("pi0", {}) == {"mass": (0.09, 0.18, 18),
                                              "energy": (-6.0, 6.0, 48),
                                              "theta": (-0.3, 0.3, 40),
                                              "phi": (-0.3, 0.3, 40)}
    assert measure.pi0_binning("pi0", {"phi_range": [-1, 2], "theta_bins": 7})["phi"] == (
        -1.0, 2.0, 40)
    for settings, message in (
            ({"candidate_collection": None}, "observables.pi0.candidate_collection should be "
                                             "the name of a collection"),
            ({"reverse_link_collection": 3}, "should be the name of a collection, or null"),
            ({"ancestor_pdg": 15}, "observables.pi0.ancestor_pdg should be a list of PDG codes"),
            ({"reconstructed_energy": "sum"}, "reconstructed_energy should be fitted or photons"),
            ({"mass_range": [0.18, 0.09]}, "observables.pi0.mass_range should be [first, last]"),
            ({"energy_range": [-6]}, "energy_range should be [first, last], the lowest and "
                                     "highest edges of the bins in GeV"),
            ({"theta_range": [0.3, -0.3]}, "observables.pi0.theta_range should be [first, last], "
                                           "the lowest and highest edges of the bins in degrees"),
            ({"phi_range": "0.3"}, "observables.pi0.phi_range should be [first, last]"),
            ({"mass_bins": 0}, "observables.pi0.mass_bins should be a number of bins"),
            ({"phi_bins": 2.5}, "observables.pi0.phi_bins should be a number of bins")):
        with pytest.raises(measure.ObservableError) as error:
            configured_pi0(**settings)
        assert message in str(error.value), (settings, str(error.value))


def test_a_missing_pi0_collection_says_what_to_set(pi0_file, tmp_path):
    for settings, message in (
            ({"candidate_collection": "GammaGammaParticles"},
             "has no GammaGammaParticles.energy (does it have the collection GammaGammaParticles? "
             "set observables.pi0.candidate_collection)"),
            ({"link_collection": "MCTruthLinks"},
             "has no MCTruthLinks.weight (does it have the links MCTruthLinks? set "
             "observables.pi0.link_collection)")):
        with pytest.raises(measure.ObservableError) as error:
            measured(configured_pi0(**settings), pi0_file, str(tmp_path / "geant4.npz"))
        assert message in str(error.value), str(error.value)


def test_relations_and_links_must_point_to_the_collections_set():
    """When the file's metadata names the collections (podio's table)."""
    class Stub(object):
        path = "rec.root"

        def __init__(self, table):
            self.collection_ids = lambda: table

    table = {"PandoraPFOs": 2, "MCParticles": 1, "DistilledPFOs": 7}
    measure._points_to(Stub(table), np.array([2, 2]), "PandoraPFOs", "the links", "pi0",
                       "link_collection")
    with pytest.raises(measure.ObservableError) as error:
        measure._points_to(Stub(table), np.array([2, 7]), "PandoraPFOs",
                           "the particles of GammaGammaCandidatePi0s", "pi0",
                           "particle_collection")
    assert str(error.value) == ("the particles of GammaGammaCandidatePi0s in rec.root points to "
                                "DistilledPFOs, not to PandoraPFOs; set "
                                "observables.pi0.particle_collection")
    # Without the table (files not written by podio), or the IDs, nothing to check
    measure._points_to(Stub({}), np.array([7]), "PandoraPFOs", "x", "pi0", "y")
    measure._points_to(Stub(table), None, "PandoraPFOs", "x", "pi0", "y")


def test_the_best_link_is_of_the_highest_weight_the_first_of_equal_ones():
    links = [(5.0, 0, 10), (9.0, 0, 11), (9.0, 0, 12), (1.0, 1, 10)]
    assert measure.best_links(links, by_reco=True) == {0: 11, 1: 10}
    assert measure.best_links(links, by_reco=False) == {10: 0, 11: 0, 12: 0}
