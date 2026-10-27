"""A stand-in for DDML's steering file, for the tests: as DDML's
scripts/ddsim_steer.py, without its settings, which the fakes don't use. It
sets up the fast simulation of the presets on the command line, as DDML's
does."""

from DDSim.DD4hepSimulation import DD4hepSimulation

from ddml import ddml_physics, get_fastsim_configuration

SIM = DD4hepSimulation()
SIM.compactFile = ""
SIM.random.seed = None

presets, record_calo_entry = get_fastsim_configuration()
SIM.physics.setupUserPhysics(ddml_physics(presets, record_calo_entry))
