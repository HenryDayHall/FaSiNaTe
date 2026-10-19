"""As DDML's python/ddml/model.py, without the geometry."""


class ModelConfig(object):
    def __init__(self, plugin, plugin_properties, applicable_particles, triggers=None,
                 correct_angles=False, is_hadron=False):
        self.plugin = plugin
        self.plugin_properties = plugin_properties
        self.applicable_particles = applicable_particles
        self.triggers = triggers or {}
        self.correct_angles = correct_angles
        self.is_hadron = is_hadron
