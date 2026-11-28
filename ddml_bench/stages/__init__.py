"""The stages of the pipeline, in order: the stage registry.

  preflight    each model in models/, checked before it simulates   (preflight.py)
  generate     each input of each scenario                          (generate.py)
  geant4       each input, simulated with Geant4                     (simulate.py)
  models       each input, simulated with each model                 (simulate.py)
  reconstruct  each simulation, reconstructed                        (reconstruct.py)
  observables  each simulator's reconstructions, measured            (observables.py)
  plot         every measurement, plotted against Geant4             (plot.py)

base.py says what a stage declares. The graph (ddml_bench/graph.py) asks each
stage, in this order, for its tasks, so a stage can use the tasks of the
stages before it.
"""

from .base import Stage, StageError
from .generate import Generate
from .observables import Observables
from .plot import Plot
from .preflight import Preflight
from .reconstruct import Reconstruct
from .simulate import Geant4, Models

STAGES = (Preflight(), Generate(), Geant4(), Models(), Reconstruct(), Observables(), Plot())
STAGE_NAMES = tuple(stage.name for stage in STAGES)
BY_NAME = dict((stage.name, stage) for stage in STAGES)

__all__ = ["BY_NAME", "STAGES", "STAGE_NAMES", "Stage", "StageError"]
