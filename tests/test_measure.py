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
    assert str(error.value).endswith("radial_profile, longitudinal_profile, energy")
