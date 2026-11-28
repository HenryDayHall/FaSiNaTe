# Models

How a model is described in `models/`, and the models there. The README's [How to add a model](../README.md#how-to-add-a-model) has the short version.

## Adding a model

Add a YAML file to `models/`; its file name becomes the model's name in commands and output folders. Every file there is commented, e.g. [`models/caloclouds3_paper.yaml`](../models/caloclouds3_paper.yaml). It records:

- `name`: its label in the figures;
- `representation`: how the model describes a shower, e.g. `point_cloud`;
- `fixed_at_training`: what the model has no input for, so where it is valid: the detector (and the compact file to simulate it in) and the particle;
- `context`: what the model is conditioned on for each shower, e.g. energy and direction, with units and training ranges;
- `ddml`: how DDML runs it, one of
  - `presets`: DDML's presets (from `python/ddml/configs.py` in DDML), passed to `ddsim --ml-model`;
  - `model_configs`: for a model DDML has no preset for, or a preset with something changed, DDML's model configurations written out ([`ddml_bench/ddml_configs.py`](../ddml_bench/ddml_configs.py)). Each is a preset's name, or the preset it starts from (`preset`), if any, and the fields of DDML's `ModelConfig` it sets: `plugin` (`<factory>/<instance name>`, the factories declared in DDML's `src/MLModelActions.cc`), `geometry` (`ILD_BARREL` or `ILD_ENDCAP` from DDML's `configs.py`, or `{ecal: {detector, region, symmetry}}`), `plugin_properties` (e.g. `ModelPath`, relative to DDML's `scripts/` folder as in DDML's presets, or `PythonModule`, the module an embedded-Python plugin runs, which can be [the model's own](#a-models-own-inference-code); set over a preset's one by one), `applicable_particles`, `triggers` (`{particle: GeV}`), `correct_angles` and `is_hadron`. Without a preset, `plugin`, `geometry`, `applicable_particles` and `triggers` are needed. [`unused_models/caloclouds3_libtorch.yaml`](../unused_models/caloclouds3_libtorch.yaml) changes a preset's weights, [`models/caloclouds3_recompiled_5GeV.yaml`](../models/caloclouds3_recompiled_5GeV.yaml) its trigger, and [`models/l2lflows.yaml`](../models/l2lflows.yaml) writes a configuration out whole;

  then the weights file, `weights`, relative to the DDML clone, or to this repository with `weights_in: fasinate`, with its md5 sum, `weights_md5` (the check of the environment makes sure that the plugins load that file), and `particles`, the particles DDML gives the model, if they aren't just the one it was trained on;
- `closure` (optional): the model this one is the [closure test](#closure-tests) of, `of`, and whether their events should be the same where both are simulated with the same seeds, `identical`;
- `plot`: its colour and line style in the figures (each model needs a distinct line style as well as a colour, so curves can be told apart without colour).

The weights are found in the DDML clone, where building DDML downloads DDML's own into its `models/` folder, so put new weights there too, or, with `ddml.weights_in: fasinate`, in this repository, e.g. in `models/weights/` (under GitHub's 100 MB per file), which a model with [its own inference code](#a-models-own-inference-code) can load from there. The model then takes part in stages 3–6 automatically, once it passes its [preflight](stages.md#0-check-the-models) (`./preflight_models.sh <model>`), which every run makes before simulating it. That needs `ddml.weights` and `ddml.weights_md5`, and a model's own code formatted with Black.

### A model's own inference code

A model can run inference code of its own in place of DDML's: a Python module in `models/`, which a configuration names as the `PythonModule` of one of DDML's embedded-Python plugins, as [`models/caloclouds3_recompiled.yaml`](../models/caloclouds3_recompiled.yaml) has DDML's `CC3_BARREL_PY_INTERFACE` run [`models/caloclouds3_recompiled.py`](../models/caloclouds3_recompiled.py), which loads weights kept in this repository (`weights_in: fasinate`):

```yaml
ddml:
  model_configs:
    - preset: CC3_BARREL_PY_INTERFACE
      plugin_properties:
        PythonModule: caloclouds3_recompiled
  weights_in: fasinate
  weights: models/weights/CC3_recompiled.pt
```

The module keeps DDML's contract for embedded Python, `run_inference(inputs)`, as DDML's own `python/examples/cc3_sf_2a_wrapper.py` does. `scripts/ddsim_steer_models.py` puts `models/` first on ddsim's Python path, so the plugin imports the module from there, and nothing needs to be set before running `ddsim`: the module can read what it needs from the model's description beside it, for the weights, the file `ddml.weights` names and whose md5 sum is checked: in this repository, beside `models/`, with `weights_in: fasinate`, or otherwise in the DDML clone, at `$DDML_SOURCE_DIR` (which `setup.sh` sets). The check of the environment says where each plugin's module is found, and stops a run where a module of `models/` would stand in for a preset's own without the description naming it. Each simulation's log records the sha256 of the model's description (as read, so not its comments, label, plot style or closure test) and of each module of its own code as `ddsim` started, so the [manifest](published.md#publishing-a-data-folder) of the published measurements has them, and published measurements are fetched only for a model that is still the same. A module is hashed alone, so keep a model's own code in its one module.

## The models

Each model in [`models/`](../models/) runs through DDML, as DDML's presets do: photons above the trigger (10 GeV, unless the description sets another) entering the ECal regions of its presets go to the model, and Geant4 simulates everything else. They are DDML's generative models, with DDML's own weights, which building DDML downloads (and `L2LFlowsx9.pt`, which DDML's build leaves out, `scripts/build_ddml.sh` fetches as DDML would, by its md5 sum: `ddml.extra_models`), or with weights kept in this repository, in `models/weights/`:

| Model | What it is | How DDML runs it | Backend | Weights |
|---|---|---|---|---|
| `caloclouds3_paper` | CaloClouds3, a point-cloud model: the checkpoint of the CaloClouds3 paper | presets `CC3_BARREL` and `CC3_ENDCAP` | LibTorch | `CC3_paper_checkpoint.pt` |
| `caloclouds3_recompiled` | CaloClouds3, its TorchScript file regenerated from the weights | its configuration: presets `CC3_BARREL_PY_INTERFACE` and `CC3_ENDCAP_PY_INTERFACE`, running [its own code](#a-models-own-inference-code), `models/caloclouds3_recompiled.py` | embedded Python | `models/weights/CC3_recompiled.pt`, in this repository |
| `caloclouds3_recompiled_5GeV` | the same, for photons down to 5 GeV | the same, with the triggers at 5 GeV | embedded Python | the same |
| `l2lflows` | ConvL2LFlows, a layer-to-layer normalising flow on a grid three times finer than the cells | its configuration | LibTorch | `L2LFlowsx9.pt` |
| `bibae_two_angle` | a BIB-AE on a regular grid, conditioned on the energy and both angles | preset `BIBAE_BARREL` | LibTorch | `BIBAE_Two_Angle_Full_PP_cut.pt` |

More descriptions are kept in [`unused_models/`](../unused_models/), out of the runs; move one into `models/` to run it:

| Model | What it is | How DDML runs it | Backend | Weights |
|---|---|---|---|---|
| `caloclouds3_libtorch` | CaloClouds3 compiled: the closure test of `caloclouds3` (published measurements) | preset `CC3_BARREL`, with `caloclouds3`'s weights | LibTorch | `CC3_SF_2A.pt` |
| `bibae_one_angle` | a BIB-AE on a regular grid, conditioned on the energy and the polar angle | its configuration | LibTorch | `BIBAE_Full_PP_cut.pt` |
| `gan_libtorch` | a GAN on a regular grid, at normal incidence, conditioned on the energy | its configuration | LibTorch | `francisca_gan_jit.pt` |
| `gan_onnx` | the same network: `gan_libtorch`'s closure test | its configuration | ONNX Runtime | `francisca_gan.onnx` |
| `par04_vae_onnx` | the VAE of Geant4's example Par04 | its configuration | ONNX Runtime | `Generator.onnx` |

Each file says what the model was trained on and where that comes from. A few things to bear in mind when comparing them:

- The CaloClouds3 models also run DDML's endcap presets (`CC3_ENDCAP`, `CC3_ENDCAP_PY_INTERFACE`); the others run in the barrel alone.
- The plugins of CaloClouds3, ConvL2LFlows and the two-angle BIB-AE have a trigger of their own that leaves the ends of the barrel and the joins between its staves to Geant4; those of the GAN, the one-angle BIB-AE and the Par04 VAE (in `unused_models/`) don't.
- The Par04 VAE tests DDML's ONNX path through the whole chain; it isn't a surrogate for ILD. It was trained on electrons in Par04's own cylindrical calorimeter, and DDML conditions it on the angle from the normal to the layers, where the example trained it on the angle from the cylinder's axis.
- Left out: the `LoadHDF5` plugins replay showers from a file rather than generating them, and PionClouds is of hadron showers.

### Closure tests

A model can be the closure test of another: the same model, through another of DDML's inference backends. Both pairs below are in `unused_models/`. `caloclouds3_libtorch` is `caloclouds3`'s, its weights through DDML's plugin compiled against LibTorch instead of DDML's embedded Python, and `gan_onnx` is `gan_libtorch`'s, the same network exported to ONNX (its weights are those of `francisca_gan_jit.pt`, number for number). Its description names the other (`closure`), and `plot_observables.sh` (so `pipeline.sh` too) then draws the two against each other alone, the other as the reference, in `plots/closure/<model>/`, with the metrics between them, and writes `plots/closure.csv`, a row for each scenario:

- `seeds`: `same` where each of the two simulated each input with the same ddsim seed (the scenario's `simulation_seeds`), `ddsim's own` where ddsim chose them, `different` otherwise;
- `identical`: whether everything the two measured on the scenario is the same, number for number;
- `expected`: `identical` where the description says the events should be (`identical: true`) and the seeds are the same, and `statistical` otherwise; `passed`, for those expected identical, whether they are. A failure is printed too.

Both CaloClouds3 backends run the same TorchScript model on the same LibTorch library, but the model draws its random numbers from LibTorch's own generator, which PyTorch seeds anew in every process, not from ddsim's seed. So two runs give other showers even for the same events and seeds, and the closure test of the embedded Python is statistical: the figures and metrics of `closure/caloclouds3_libtorch/` show whether the distributions agree, in every scenario. The two GANs draw their latent vectors from Geant4's random numbers, so with the same seeds (`di_photon`) they are given the same inputs, but ONNX Runtime and LibTorch round their arithmetic differently, so they too need only agree statistically. `identical: true` is for a pair that should give the same events, number for number, with the same seeds; `passed` then says whether they do. Both of a pair should be simulated with the same software (`key4hep.release`, `ddml.commit`); a run that fetches the published measurements of `caloclouds3` says where its software differs from theirs.
