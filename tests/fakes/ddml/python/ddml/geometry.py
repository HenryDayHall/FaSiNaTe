"""As DDML's python/ddml/geometry.py (python-inference branch)."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class DetectorGeometry:
    detector: str
    region: str
    symmetry: Optional[int] = None


@dataclass(frozen=True)
class PluginGeometry:
    ecal: DetectorGeometry
    hcal: Optional[DetectorGeometry] = None
