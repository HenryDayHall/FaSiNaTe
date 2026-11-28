"""Fixtures for the tests: a copy of the repository with a configuration of its
own, and a fake software stack for the environment the pipeline runs in.

The copy (so that the tests never touch the real data/ or configs/local.yaml)
is configured by test.yaml: a small single_photon scenario of two energies and
a small di_photon one of one energy simulated with two seeds, its input whole
(Bench.with_segments() puts the inputs in segments), the tau_pairs scenario
disabled (Bench.with_tau_pairs() enables it, with inputs of its own),
a data folder of its own, no container, ILDConfig cloned from a local
repository and no patches.
Its "DDML build" is a folder with the files setup.sh and the
stages' environment checks look for, and an env file that puts the fakes in
tests/fakes on PATH and PYTHONPATH: ddsim, k4run, DDML's Python package and
pyLCIO, and slurm's sbatch and squeue. So every script runs as it does on the
cluster, through scripts/run_in_environment.sh and setup.sh, with only the
programs at the ends faked.
"""

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

import pytest
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAKES = os.path.join(REPO, "tests", "fakes")
sys.path.insert(0, REPO)

# What the copy of the repository has (and of models/, only BENCH_MODELS)
COPIED = ["ddml_bench", "scripts", "configs", "ILD_Particle_Gun.py",
          "ILD_Di_Photon_Gun.py", "setup.sh",
          "pipeline.sh", "slurm_pipeline.sh", "slurm_simulate_geant4.sh", "generate_input.sh",
          "simulate_geant4.sh", "simulate_models.sh", "preflight_models.sh", "reconstruct_simulation.sh",
          "observables_from_reconstruction.sh", "plot_observables.sh", "low_cost.sh"]
# A model of DDML's presets alone, with its weights in DDML's build
BENCH_MODELS = ["caloclouds3_paper"]
# The plugins DDML declares (src/MLModelActions.cc), with every backend built
DDML_PLUGINS = [
    "RegularGridGANPolyhedraBarrelONNXModel", "RegularGridGANEndcapONNXModel",
    "Par04ExampleVAEPolyhedraBarrelONNXModel", "Par04ExampleVAEEndcapONNXModel",
    "RegularGridGANPolyhedraBarrelTorchModel", "RegularGridGANEndcapTorchModel",
    "RegularGridBIBAEPolyhedraBarrelTorchModel", "RegularGridBIBAEEndcapTorchModel",
    "RegularGridTwoAngleBIBAEModelPolyhedraBarrelTorchModel",
    "RegularGridTwoAngleBIBAEModelEndcapTorchModel",
    "CaloCloudsTwoAngleModelPolyhedraBarrelTorchModel", "CaloCloudsTwoAngleModelEndcapTorchModel",
    "L2LFlowsModelPolyhedraBarrelTorchModel", "L2LFlowsModelEndcapTorchModel",
    "CaloCloudsTwoAngleModelPolyhedraBarrelPyEmbeddedModel",
    "CaloCloudsTwoAngleModelEndcapPyEmbeddedModel",
    "LoadHDF5RegularGridTwoAngleBIBAEModelPolyhedraBarrel",
    "LoadHDF5RegularGridTwoAngleBIBAEModelEndcap",
    "LoadHDF5PionCloudsPCHadronModelPolyhedraBarrel", "LoadHDF5PionCloudsPCHadronModelEndcap"]
COMPACT = "ILD/compact/ILD_l5_o1_v02/ILD_l5_o1_v02.xml"
CELL_ID_ENCODING = "system:5,module:3,stave:4,tower:4,layer:6,wafer:6,slice:4,cellX:32:-16,cellY:-16"
GIT = ["git", "-c", "user.name=tests", "-c", "user.email=tests@example.org",
       "-c", "commit.gpgsign=false", "-c", "init.defaultBranch=master"]

TEST_CONFIG = """\
paths:
  software_dir: {software}
  data_dir: {data}
ddml:
  update: false
  download_models: false
key4hep:
  setup_script: {tmp}/cvmfs/key4hep/setup.sh
  release: "2026-04-08"
container:
  enabled: false
detector_configs:
  ildconfig:
    repository: {tmp}/ILDConfig.git
patches:
  marlintrkprocessors:
    enabled: false
scenarios:
  single_photon:
    input:
      energies: [20, 80]
      n_events: 6
  di_photon:
    input:
      energies: [20]
      n_events: 30
      segments: 1
    simulation_seeds: [1, 2]
  tau_pairs:
    enabled: false
observables:
  longitudinal_profile:
    cell_id_encoding: "{encoding}"
published:
  enabled: false
preflight:
  # (whichever Black the tests have: the newest for Python 3.9 is older)
  black: null
"""

# The tau_pairs scenario of the tests (Bench.with_tau_pairs()): two inputs made
# "elsewhere", each simulated with its own seed
TAU_INPUTS = ["taus_a.slcio", "taus_b.slcio"]
TAU_SEEDS = [42, 1764]
TAU_CONFIG = """\
scenarios:
  tau_pairs:
    enabled: true
    input:
      files: [{files}]
      segments: {segments}
    simulation_seeds: [{seeds}]
"""
# The segments of the inputs of di_photon and tau_pairs (Bench.with_segments())
SEGMENTS_CONFIG = """\
scenarios:
  di_photon:
    input:
      segments: {segments}
  tau_pairs:
    input:
      segments: {segments}
"""

ENV_FILE = """\
# A fake DDML build for the tests (tests/conftest.py)
# build stamp: {stamp}
_fasinate_k4_release=2026-04-08
_fasinate_k4_platform=x86_64-almalinux9-gcc14.2.0-opt
export DDML_SOURCE_DIR={ddml}
export DDML_INSTALL_DIR={ddml}/install
export DDML_COMMIT=0123456789abcdef0123456789abcdef01234567
export DDML_KEY4HEP_RELEASE=2026-04-08
export KEY4HEP_STACK={tmp}/key4hep-stack/setup.sh
export k4geo_DIR={tmp}/k4geo
export PATH={bin}:$PATH
export PYTHONPATH={ddml}/python:{ddml}/python/examples:{python}${{PYTHONPATH:+:$PYTHONPATH}}
export LD_LIBRARY_PATH={ddml}/install/lib${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}
export FAKE_CONTROL={tmp}/fake-control
unset _fasinate_k4_release _fasinate_k4_platform
return 0
"""


def _write(path, text=""):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as stream:
        stream.write(text)


PI0_MASS = 0.1349768


def tau_events(n_events, seed):
    """Made-up e+e- -> tau+tau- events, as the fake LCIO files hold them: their
    MC particles with their parents and daughters (indices in the event). The
    tau- decays to a nu_tau and a rho-, which decays to a pi- and a pi0 into two
    photons, of exactly the pi0's mass; the tau+ to an anti-nu_tau and a pi+,
    or in every third event to a rho+ and so a second pi0."""
    import numpy as np
    rng = np.random.default_rng(seed)
    events = []

    def particle(pdg, status, momentum, mass=0.0, parents=()):
        return {"PDG": pdg, "generatorStatus": status, "mass": mass, "charge": 0.0,
                "momentum": [float(p) for p in momentum], "vertex": [0.0, 0.0, 0.0],
                "parents": list(parents), "daughters": []}

    def decay(particles, parent, pdg_pi0, direction, energy):
        """A pi0 of the energy along the direction, and its two photons."""
        pi0 = len(particles)
        momentum = direction * np.sqrt(energy ** 2 - PI0_MASS ** 2)
        particles.append(particle(pdg_pi0, 2, momentum, PI0_MASS, [parent]))
        particles[parent]["daughters"].append(pi0)
        # The photons share its energy, at the opening angle of its mass
        share = rng.uniform(0.25, 0.75)
        energies = np.array([share, 1 - share]) * energy
        # In its rest frame, back to back at an angle to its flight, boosted
        gamma = energy / PI0_MASS
        beta = np.sqrt(1 - 1 / gamma ** 2)
        cos_star = (energies[0] - energies[1]) / (beta * energy)
        sin_star = np.sqrt(max(1 - cos_star ** 2, 0.0))
        across = np.cross(direction, [0.0, 0.0, 1.0])
        across /= np.linalg.norm(across)
        half = PI0_MASS / 2
        for sign in (1, -1):
            along = gamma * (sign * half * cos_star + beta * half)
            transverse = sign * half * sin_star
            photon = along * direction + transverse * across
            particles.append(particle(22, 1, photon, 0.0, [pi0]))
            particles[pi0]["daughters"].append(len(particles) - 1)

    for number in range(n_events):
        cos_theta = rng.uniform(-0.7, 0.7)
        phi = rng.uniform(-np.pi, np.pi)
        sin_theta = np.sqrt(1 - cos_theta ** 2)
        axis = np.array([sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta])
        particles = [particle(15, 2, 124.9 * axis, 1.777), particle(-15, 2, -124.9 * axis, 1.777)]
        for tau, sign, with_pi0 in ((0, 1, True), (1, -1, number % 3 == 0)):
            nu = rng.uniform(0.1, 0.4)
            particles.append(particle(16 * sign, 1, sign * nu * 125 * axis, 0.0, [tau]))
            particles[tau]["daughters"].append(len(particles) - 1)
            if not with_pi0:
                particles.append(particle(211 * -sign, 1, sign * (1 - nu) * 125 * axis, 0.1396,
                                          [tau]))
                particles[tau]["daughters"].append(len(particles) - 1)
                continue
            rho = len(particles)
            particles.append(particle(-213 * sign, 2, sign * (1 - nu) * 125 * axis, 0.775, [tau]))
            particles[tau]["daughters"].append(rho)
            pi0_energy = (1 - nu) * 125 * rng.uniform(0.3, 0.7)
            direction = sign * axis + rng.normal(0, 0.01, 3)
            decay(particles, rho, 111, direction / np.linalg.norm(direction), pi0_energy)
            particles.append(particle(211 * -sign, 1, sign * ((1 - nu) * 125 - pi0_energy) * axis,
                                      0.1396, [rho]))
            particles[rho]["daughters"].append(len(particles) - 1)
        events.append({"event": number, "particles": particles})
    return events


def write_tau_input(path, n_events, seed):
    """A fake LCIO file of tau pair events (tau_events())."""
    lines = [json.dumps({"run": {"made": "elsewhere, by the tests"}})]
    lines += [json.dumps(event) for event in tau_events(n_events, seed)]
    _write(path, "\n".join(lines) + "\n")


class Bench(object):
    """A copy of the repository, its configuration and its fake environment."""

    def __init__(self, tmp):
        self.tmp = tmp
        self.repo = os.path.join(tmp, "repo")
        self.software = os.path.join(tmp, "software")
        self.data = os.path.join(tmp, "data")
        self.bin = os.path.join(tmp, "bin")
        self.slurm_state = os.path.join(tmp, "fake-slurm")
        self.ddml = ddml = os.path.join(self.software, "DDML")

        # The repository, without anyone's local settings
        os.makedirs(self.repo)
        ignore = shutil.ignore_patterns("__pycache__", "local.yaml")
        for name in COPIED:
            source = os.path.join(REPO, name)
            target = os.path.join(self.repo, name)
            if os.path.isdir(source):
                shutil.copytree(source, target, ignore=ignore)
            else:
                shutil.copy2(source, target)

        # The fake DDML build, the models with their weights, and the detector
        shutil.copytree(os.path.join(FAKES, "ddml"), ddml)
        os.makedirs(os.path.join(self.repo, "models"))
        for name in BENCH_MODELS:
            self.add_model(name)
        _write(os.path.join(ddml, "install", "bin", "thisDDML.sh"), "# fake\n")
        _write(os.path.join(ddml, "install", "lib", "libDDML.so"), "fake\n")
        _write(os.path.join(ddml, "install", "lib", "libDDML.components"),
               "".join("v2::libDDML.so:%s\n" % plugin for plugin in DDML_PLUGINS))
        _write(os.path.join(tmp, "k4geo", COMPACT), "<lccdd/>\n")

        # ILDConfig, as a repository to clone
        source = os.path.join(tmp, "ILDConfig-source")
        _write(os.path.join(source, "StandardConfig", "production", "ILDReconstruction.py"),
               "# fake\n")
        _write(os.path.join(source, "StandardConfig", "production", "Config", "settings.py"),
               "# fake\n")
        subprocess.check_call(GIT + ["init", "-q", source])
        subprocess.check_call(GIT + ["-C", source, "add", "-A"])
        subprocess.check_call(GIT + ["-C", source, "commit", "-q", "-m", "fake ILDConfig"])
        subprocess.check_call(GIT + ["clone", "-q", "--bare", source,
                                     os.path.join(tmp, "ILDConfig.git")])

        # The fake programs, and this Python as the environment's python3
        shutil.copytree(os.path.join(FAKES, "bin"), self.bin)
        for name in os.listdir(self.bin):
            os.chmod(os.path.join(self.bin, name), 0o755)
        os.symlink(sys.executable, os.path.join(self.bin, "python3"))
        os.symlink(sys.executable, os.path.join(self.bin, "python"))

        self.config = os.path.join(tmp, "test.yaml")
        _write(self.config, TEST_CONFIG.format(software=self.software, data=self.data, tmp=tmp,
                                               encoding=CELL_ID_ENCODING))
        self.env = dict((key, value) for key, value in os.environ.items()
                        if not key.startswith(("FASINATE_", "DDML_", "FAKE_"))
                        and key not in ("KEY4HEP_STACK", "k4geo_DIR"))
        self.env.update(FASINATE_CONFIG=self.config, FASINATE_PYTHON=sys.executable,
                        PATH=self.bin + os.pathsep + self.env.get("PATH", ""),
                        FAKE_SLURM_DIR=self.slurm_state)
        stamp = self.setting("FASINATE_DDML_BUILD_STAMP")
        env_file = os.path.join(self.software, "env", "ddml.sh")
        _write(env_file, ENV_FILE.format(stamp=stamp, ddml=ddml, tmp=tmp, bin=self.bin,
                                         python=os.path.join(FAKES, "python")))

    def add_model(self, name, text=None, folder="models"):
        """Put the repository's <folder>/<name>.yaml (models/ unless said, or
        unused_models/ for a model set aside), or text, in the copy's models/,
        with fake weights where it keeps them, in the fake DDML build or, with
        ddml.weights_in: fasinate, in the copy of the repository, whose md5 sum
        it is then given, and the modules of <folder> that its DDML model
        configurations run (its own inference code: plugin_properties
        PythonModule)."""
        if text is None:
            with open(os.path.join(REPO, folder, name + ".yaml")) as stream:
                text = stream.read()
        ddml = yaml.safe_load(text).get("ddml") or {}
        weights = ddml.get("weights")
        if weights:
            path = os.path.join(self.repo if ddml.get("weights_in") == "fasinate" else self.ddml,
                                weights)
            if not os.path.exists(path):
                _write(path, "fake weights: %s\n" % weights)
            md5 = hashlib.md5(open(path, "rb").read()).hexdigest()
            # (quoted or not, as yaml.safe_dump writes it)
            text = re.sub(r"""weights_md5: *["']?[0-9a-f]{32}["']?""", 'weights_md5: "%s"' % md5,
                          text)
        _write(os.path.join(self.repo, "models", name + ".yaml"), text)
        for entry in ddml.get("model_configs") or []:
            module = ((entry.get("plugin_properties") or {}).get("PythonModule")
                      if isinstance(entry, dict) else None)
            source = os.path.join(REPO, folder, "%s.py" % module) if module else None
            if source and os.path.isfile(source):
                shutil.copy2(source, os.path.join(self.repo, "models", module + ".py"))

    def description(self, name):
        """The copy's models/<name>.yaml, as read: what a test expects of a
        model comes from there, so that changing a model's presets or
        configurations doesn't break the tests."""
        with open(os.path.join(self.repo, "models", name + ".yaml")) as stream:
            return yaml.safe_load(stream)

    def setting(self, name):
        """A setting from scripts/fasinate_config.py --shell, by its variable."""
        output = subprocess.check_output(
            [sys.executable, os.path.join(self.repo, "scripts", "fasinate_config.py"), "--shell"],
            env=self.env, universal_newlines=True)
        for line in output.splitlines():
            if line.startswith("export %s=" % name):
                return line.split("=", 1)[1].strip("'")
        raise KeyError(name)

    def run(self, *command, **options):
        """Run a script of the copy, from its root; returns (status, output).
        With check (the default), fail the test unless it succeeds."""
        check = options.pop("check", True)
        env = dict(self.env, **options.pop("env", {}))
        result = subprocess.run([os.path.join(self.repo, command[0])] + list(command[1:]),
                                cwd=self.repo, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, universal_newlines=True)
        if check and result.returncode != 0:
            pytest.fail("%s exited with %d:\n%s" % (" ".join(command), result.returncode,
                                                     result.stdout))
        return result.returncode, result.stdout

    def run_fake(self, name, *args):
        """Run one of the fakes, e.g. sbatch --drain."""
        subprocess.check_call([os.path.join(self.bin, name)] + list(args), env=self.env,
                              stdout=subprocess.DEVNULL)

    def fail_ddsim(self, pattern=None):
        """Make the fake ddsim fail on "<simulator> <input file name>" matching
        the shell pattern; with None, not any more."""
        path = os.path.join(self.tmp, "fake-control", "ddsim_fail")
        if pattern is None:
            if os.path.exists(path):
                os.remove(path)
        else:
            _write(path, pattern + "\n")

    def with_tau_pairs(self, inputs=TAU_INPUTS, events=12, write=True):
        """Enable the tau_pairs scenario, with these inputs, whole, each
        simulated with its own seed (TAU_SEEDS), and (with write) write them
        into its folder, of so many events each; returns the configuration file
        it adds after the others."""
        override = os.path.join(self.tmp, "tau_pairs.yaml")
        seeds = ", ".join(str(seed) for seed in TAU_SEEDS[:len(inputs)])
        _write(override, TAU_CONFIG.format(files=", ".join(inputs), seeds=seeds, segments=1))
        self.env["FASINATE_CONFIG"] = self.env["FASINATE_CONFIG"] + ":" + override
        if write:
            for number, name in enumerate(inputs):
                write_tau_input(self.path("input_physics_samples", name), events, number)
        return override

    def with_segments(self, segments=3):
        """Put the inputs of di_photon and tau_pairs in so many segments
        each, as the default configuration does; returns the configuration
        file it adds after the others."""
        override = os.path.join(self.tmp, "segments.yaml")
        _write(override, SEGMENTS_CONFIG.format(segments=segments))
        if override not in self.env["FASINATE_CONFIG"].split(":"):
            self.env["FASINATE_CONFIG"] = self.env["FASINATE_CONFIG"] + ":" + override
        return override

    def path(self, *parts):
        return os.path.join(self.data, *parts)

    def files(self, *parts):
        """The files in a folder of the data, without hidden ones."""
        folder = self.path(*parts)
        if not os.path.isdir(folder):
            return []
        return sorted(name for name in os.listdir(folder) if not name.startswith("."))

    def slurm_jobs(self):
        path = os.path.join(self.slurm_state, "jobs.json")
        if not os.path.exists(path):
            return []
        with open(path) as stream:
            return sorted(json.load(stream)["jobs"].values(), key=lambda job: int(job["id"]))


@pytest.fixture
def bench(tmp_path):
    return Bench(str(tmp_path))


@pytest.fixture
def config_env(bench, monkeypatch):
    """This process configured as the bench is (for the graph and layout tests,
    which run the harness in-process), its models too."""
    from ddml_bench import models
    for key in list(os.environ):
        if key.startswith(("FASINATE_", "DDML_")) or key == "KEY4HEP_STACK":
            monkeypatch.delenv(key)
    monkeypatch.setenv("FASINATE_CONFIG", bench.config)
    monkeypatch.setattr(models, "MODELS_DIR", os.path.join(bench.repo, "models"))
    return bench
