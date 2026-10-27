"""Fixtures for the tests: a copy of the repository with a configuration of its
own, and a fake software stack for the environment the pipeline runs in.

The copy (so that the tests never touch the real data/ or configs/local.yaml)
is configured by test.yaml: a small single_photon scenario of two energies and
a small di_photon one of one energy simulated with two seeds, a data folder of
its own, no container, ILDConfig cloned from a local repository and no
patches. Its models/ has caloclouds3 alone, which the tests are written
around; Bench.add_model() adds others. Its "DDML build" is a folder with the
files setup.sh and the stages' environment checks look for (every plugin DDML
declares, and fake weights for the models), and an env file that puts the
fakes in tests/fakes on PATH and PYTHONPATH: ddsim, k4run, DDML's Python
package, DD4hep's DDSim and g4units, pyLCIO, and slurm's sbatch and squeue. So
every script runs as it does on the cluster, through
scripts/run_in_environment.sh and setup.sh, with only the programs at the
ends faked.
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
          "simulate_geant4.sh", "simulate_models.sh", "reconstruct_simulation.sh",
          "observables_from_reconstruction.sh", "plot_observables.sh"]
BENCH_MODELS = ["caloclouds3"]
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
    simulation_seeds: [1, 2]
observables:
  longitudinal_profile:
    cell_id_encoding: "{encoding}"
published:
  enabled: false
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

    def add_model(self, name, text=None):
        """Put the repository's models/<name>.yaml (or text) in the copy's
        models/, with fake weights in the fake DDML build, whose md5 sum it is
        then given."""
        if text is None:
            with open(os.path.join(REPO, "models", name + ".yaml")) as stream:
                text = stream.read()
        weights = (yaml.safe_load(text).get("ddml") or {}).get("weights")
        if weights:
            path = os.path.join(self.ddml, weights)
            if not os.path.exists(path):
                _write(path, "fake weights: %s\n" % weights)
            md5 = hashlib.md5(open(path, "rb").read()).hexdigest()
            text = re.sub(r'weights_md5: "[0-9a-f]{32}"', 'weights_md5: "%s"' % md5, text)
        _write(os.path.join(self.repo, "models", name + ".yaml"), text)

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
