"""Stage 4: reconstruct every simulation with ILD's standard reconstruction,
one task per simulation.

ILDConfig's ILDReconstruction.py runs under Gaudi (k4run), as in the Key4hep
tutorial (key4hep-doc, tutorials/gaudi_ild_reco), in the detector the
simulation was simulated in: the model's fixed_at_training.detector.compact_file,
or geant4.compact. It runs in a folder of links to ILDConfig's
StandardConfig/production, since it finds its configuration relative to the
folder it runs in, and writes hidden files, renamed once k4run succeeds (the
one that marks it done last), so the outputs that exist are complete.

ILDConfig isn't part of Key4hep releases; setup.sh clones it to
detector_configs.ildconfig.source_dir with scripts/fetch_detector_configs.sh.
Arguments for ILDReconstruction.py go in reconstruction.extra_args;
--noBeamCalReco is always added, and --usingParticleGun for the scenarios of
a particle gun (the particle_gun and di_photon_gun inputs).
"""

import glob
import os
import shutil
import socket
import subprocess
import sys
import time

from .. import process
from ..layout import GEANT4, GIVEN_FILES, REC_SUFFIX, REC_SUFFIX_LCIO_ONLY
from .base import Stage, StageError, compact_problems, resolve_compact

# The script in ILDConfig, and the folder it runs in
ILDCONFIG_PRODUCTION = os.path.join("StandardConfig", "production")
RECO_SCRIPT = "ILDReconstruction.py"
# Arguments of ILDReconstruction.py that the stage sets itself
MANAGED_ARGS = ("--inputFiles", "--outputFileBase", "--compactFile", "--detectorModel",
                "--num-events", "--evt-max", "-n")


def say(message):
    print("reconstruct: " + message, file=sys.stderr)


def extra_args(config):
    """ILDReconstruction.py's arguments from reconstruction.extra_args, checked."""
    extra = config["reconstruction"]["extra_args"]
    if extra is None:
        extra = []
    if not isinstance(extra, list) or any(isinstance(a, (dict, list)) for a in extra):
        raise StageError("reconstruction.extra_args should be a list of arguments")
    args = ["true" if a is True else "false" if a is False else str(a) for a in extra]
    for arg in args:
        if arg.split("=", 1)[0] in MANAGED_ARGS:
            raise StageError("reconstruction.extra_args can't include %s: the reconstruct stage "
                             "sets it" % arg)
    return args


def done_suffix(args):
    """What the output that marks a reconstruction done ends in."""
    lcio = None
    for i, arg in enumerate(args):
        if arg == "--lcioOutput" and i + 1 < len(args):
            lcio = args[i + 1]
        elif arg.startswith("--lcioOutput="):
            lcio = arg.split("=", 1)[1]
    return REC_SUFFIX_LCIO_ONLY if lcio == "only" else REC_SUFFIX


def reco_args(extra, particle_gun):
    """Arguments for ILDReconstruction.py, besides the input, output and detector."""
    args = list(extra)
    if not {"--runBeamCalReco", "--noBeamCalReco"} & set(args):
        args.append("--noBeamCalReco")
    if particle_gun and "--usingParticleGun" not in args:
        args.append("--usingParticleGun")
    return args


class Reconstruct(Stage):
    name = "reconstruct"
    # Made for the measurements
    intermediate = True

    def add_tasks(self, graph):
        layout = graph.layout
        for simulation in graph.simulations:
            self._add(graph, layout.reconstruction_base(simulation["scenario"],
                                                        simulation["simulator"],
                                                        simulation["input_file"],
                                                        simulation["seed"]),
                      simulation["output"], [simulation["key"]], simulation["scenario"],
                      simulation["simulator"], simulation["name"], simulation["input_file"],
                      simulation["seed"])
        # Simulations given directly: a task of the graph if they are its
        # simulations, and one of their own if not
        for path in graph.plan["given_simulations"]:
            scenario, simulator, base = layout.given_simulation(path)
            key = graph.writers.get(base + graph.plan["reconstruction"]["done_suffix"])
            if key is None:
                task = self._add(graph, base, path, [], scenario, simulator,
                                 os.path.basename(base), None)
            else:
                task = graph.by_key[key]
                if os.path.realpath(task["inputs"][0]) != os.path.realpath(path):
                    raise StageError("%s and %s would both be reconstructed to %s"
                                     % (task["inputs"][0], path, task["output"]))
            graph.given.add(task["key"])

    def _add(self, graph, base, simulation, needs, scenario, simulator, name, input_file,
             seed=None):
        plan = graph.plan
        settings = plan["reconstruction"]
        compact = settings["compact"] or plan["compact"].get(simulator)
        if compact is None:
            compact = plan["compact"][GEANT4]
            if simulator != GIVEN_FILES:
                say("warning: %s isn't a model in models/ or geant4, so its simulations are "
                    "reconstructed in geant4.compact, %s (use --compact to choose)"
                    % (simulator, compact))
        params = {
            "base": base,
            "compact": compact,
            # The whole simulation: the sample was chosen when it was simulated
            "events": -1,
            "args": reco_args(settings["extra_args"], scenario in settings["particle_gun"]),
            "done_suffix": settings["done_suffix"],
            "ildconfig_dir": settings["ildconfig_dir"],
        }
        task = graph.add(self.name, base + settings["done_suffix"], needs, scenario=scenario,
                         simulator=simulator, name=name, inputs=[simulation],
                         input_file=input_file, params=params, seed=seed)
        graph.reconstructions.setdefault((simulator, scenario), []).append(task)
        return task

    def check_environment(self, tasks):
        problems = []
        if not shutil.which("k4run"):
            problems.append("k4run isn't on PATH; is this the environment from setup.sh?")
        for folder in sorted({task["params"]["ildconfig_dir"] for task in tasks}):
            script = os.path.join(folder, RECO_SCRIPT)
            if not os.path.isfile(script):
                problems.append("%s not found; get ILDConfig with "
                                "scripts/fetch_detector_configs.sh" % script)
        problems += compact_problems(sorted({task["params"]["compact"] for task in tasks}),
                                     "the reconstruction")
        return problems

    def run(self, task):
        params = task["params"]
        base, done_file, simulation = params["base"], task["output"], task["inputs"][0]
        folder, name = os.path.split(base)
        partial = os.path.join(folder, ".%s.%s.%d" % (name, process.host(), os.getpid()))
        work, log = partial + ".work", base + ".log"
        suffix = params["done_suffix"]
        ildconfig_dir = params["ildconfig_dir"]
        compact = resolve_compact(params["compact"])
        if compact is None:
            raise StageError("k4geo_DIR isn't set, so the compact file %s can't be found"
                             % params["compact"])
        command = (["k4run", os.path.join(work, RECO_SCRIPT), "--inputFiles=" + simulation,
                    "--outputFileBase=" + partial, "--compactFile=" + compact,
                    "--num-events=%d" % params["events"]] + list(params["args"]))
        try:
            commit = subprocess.check_output(["git", "-C", ildconfig_dir, "rev-parse", "HEAD"],
                                             stderr=subprocess.DEVNULL,
                                             universal_newlines=True).strip()
        except (OSError, subprocess.CalledProcessError):
            commit = "unknown"

        os.makedirs(folder, exist_ok=True)
        with open(log, "w") as stream:
            for line in [
                    "# ddml-bench reconstruct: %s on %s" % (task["simulator"], task["scenario"]),
                    "# input     %s" % simulation,
                    "# output    %s and the rest, once k4run succeeds" % done_file,
                    "# started   %s on %s" % (process.now(), socket.gethostname()),
                    "# Key4hep   %s" % (os.environ.get("DDML_KEY4HEP_RELEASE") or "unknown"),
                    "# ILDConfig %s, in %s" % (commit, ildconfig_dir),
                    "# patches   %s (patches in the configuration)"
                    % (os.environ.get("FASINATE_PATCHES_APPLIED") or "none"),
                    "# command   (in a folder linking to %s) %s"
                    % (ildconfig_dir, process.quote(command))]:
                stream.write(line + "\n")
        print("reconstruct:   log: %s" % log)

        def clean_up():
            shutil.rmtree(work, ignore_errors=True)
            process.remove(*glob.glob(glob.escape(partial) + "_*"))

        with process.interruptible():
            try:
                # ILDReconstruction.py finds its configuration relative to the
                # folder it runs in, so it runs in a folder of links to ILDConfig's,
                # which also keeps what it writes there apart from other runs
                shutil.rmtree(work, ignore_errors=True)
                os.makedirs(work)
                for entry in sorted(os.listdir(ildconfig_dir)):
                    if not entry.startswith("."):
                        os.symlink(os.path.join(ildconfig_dir, entry), os.path.join(work, entry))
                start = time.time()
                status = process.run_logged(command, work, log)
            except process.Interrupted:
                clean_up()
                process.append(log, "# INTERRUPTED %s" % process.now())
                raise
        elapsed = int(round(time.time() - start))
        shutil.rmtree(work, ignore_errors=True)

        why = None
        if status != 0:
            why = "k4run exited with status %d" % status
        elif not os.path.isfile(partial + suffix) or os.path.getsize(partial + suffix) == 0:
            why = "k4run wrote no %s" % suffix.lstrip("_")
        else:
            # The outputs, keeping the one that marks it done for last
            for path in sorted(glob.glob(glob.escape(partial) + "_*")):
                if path != partial + suffix:
                    try:
                        os.replace(path, base + path[len(partial):])
                    except OSError as error:
                        why = "could not rename %s: %s" % (path, error.strerror)
            if why is None:
                try:
                    os.replace(partial + suffix, done_file)
                except OSError as error:
                    why = "could not rename %s to %s: %s" % (partial + suffix, done_file,
                                                             error.strerror)
        if why is None:
            process.append(log, "# finished  %s, after %d s" % (process.now(), elapsed))
            print("reconstruct:   done in %d s" % elapsed)
            return
        clean_up()
        process.append(log, "# FAILED    %s: %s" % (process.now(), why))
        raise StageError("%s; the end of its log, %s:\n%s" % (why, log, process.log_tail(log)))

    def describe(self, task):
        return "the reconstruction of the %s simulation of %s" % (task["simulator"],
                                                                  task["name"])

    def label(self, task):
        return "%s %s/%s" % (task["simulator"], task["scenario"], task["name"])

    def job_name(self, task):
        return "reco-" + task["simulator"]

    def log_name(self, task):
        return [self.name, task["simulator"], task["scenario"], task["name"]]
