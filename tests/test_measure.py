"""Measuring the reconstructed energy of each event, on a small EDM4hep-style
file written here (a podio "events" tree, as tests/fakes/bin/k4run writes)."""

import os

import numpy as np
import pytest

from ddml_bench import measure

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
        "radial_profile, longitudinal_profile, energy, intrinsic_angle")


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
    particle, its momentum, [(energy, position)] of its ECAL barrel hits)."""
    import awkward as ak
    import uproot
    columns = {"MCParticles.generatorStatus": [[status] for status, _, _ in events],
               "MCParticles.mass": [[0.0] for _ in events],
               "EcalBarrelCollectionRec.energy": [[e for e, _ in hits] for _, _, hits in events],
               "EcalBarrelCollectionRec.cellID": [[0] * len(hits) for _, _, hits in events]}
    for i, c in enumerate("xyz"):
        columns["MCParticles.vertex." + c] = [[0.0] for _ in events]
        columns["MCParticles.momentum." + c] = [[float(p[i])] for _, p, _ in events]
        columns["EcalBarrelCollectionRec.position." + c] = [
            [float(position[i]) for _, position in hits] for _, _, hits in events]
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


def test_changing_how_the_residuals_are_drawn_needs_no_measuring_again(showers, tmp_path):
    out = str(tmp_path / "geant4.npz")
    measured(configured_angles(), showers, out)
    files = {"s": [showers]}
    for drawn in ({"max_residual": [10, 2]}, {"n_bins": 40}, {"metric": "l1"}):
        assert measure.is_up_to_date(out, measure.wanted(configured_angles(**drawn)), files) \
            == (True, ""), drawn
    for changed, why in (({"hit_fractions": [1.0, 0.1]},
                          "geant4.npz has no intrinsic_angle__s__dtheta_top10"),
                         ({"hit_collections": ["EcalBarrelCollectionRec",
                                               "EcalEndcapsCollectionRec"]},
                          "the settings of intrinsic_angle changed")):
        assert measure.is_up_to_date(out, measure.wanted(configured_angles(**changed)), files) \
            == (False, why)


def test_the_intrinsic_angle_settings_are_checked():
    assert measure.residual_binning("intrinsic_angle", {"hit_fractions": [1.0, 0.04, 0.025],
                                                        "max_residual": [10, 5, 5]}) == {
        "all": (-10, 10, 100), "top4": (-5, 5, 100), "top2.5": (-5, 5, 100)}
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
