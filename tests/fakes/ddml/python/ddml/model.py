"""As DDML's python/ddml/model.py (python-inference branch)."""

from dataclasses import dataclass
from typing import Any, Mapping

from .geometry import PluginGeometry


@dataclass(frozen=True)
class ModelConfig:
    plugin: str
    plugin_properties: Mapping[str, Any]
    geometry: PluginGeometry
    applicable_particles: frozenset
    triggers: Mapping[str, float]
    correct_angles: bool = False
    is_hadron: bool = False
