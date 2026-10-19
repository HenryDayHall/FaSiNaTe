"""As DDML's python/ddml/configs.py, for the presets the tests use."""

from .model import ModelConfig

PHOTONS = frozenset({"gamma"})

CC3_BARREL_PY_INTERFACE = ModelConfig(
    plugin="CaloCloudsTwoAngleModelPolyhedraBarrelPyEmbeddedModel/BarrelModelPython",
    plugin_properties={"PythonModule": "cc3_sf_2a_wrapper", "EntryPoint": "run_inference"},
    applicable_particles=PHOTONS,
    triggers={"gamma": 10.0},
)
