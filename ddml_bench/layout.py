"""Where every file of the benchmark goes: the one module that names them.

Everything is written under paths.data_dir:

  <folder>/<input>.slcio (and .h5)                            the scenarios' inputs,
      each scenario's in the folder its input.folder names: the guns' in
      input_single_particle_gun, and the physics samples, made elsewhere, in
      input_physics_samples
  sim_output/<scenario>/<simulator>/<input><sample><ending>   each input simulated,
      with <input><sample>.log; the ending is .edm4hep.root for Geant4 (both
      its samples, geant4 and geant4_2) and .root for the models, EDM4hep all
  rec_output/<scenario>/<simulator>/<input><sample>_REC.edm4hep.root
      each simulation reconstructed, with _AIDA.root, _PfoAnalysis.root and .log
  observables/<simulator><sample>.npz                         each simulator's measurements
  observables/plots/<observable>_<profile>__<scenario><sample>.<format>
  observables/plots/metrics<sample>.csv                       the figures and metrics,
  observables/plots/fits<sample>.csv                          and the energy figures' fits
  observables/plots/pareto_<js|wasserstein>_<node>__<scenario><sample>.<format>
  observables/plots/pareto<sample>.csv                        the Pareto fronts of the metrics
      against the time per event in timing/summary.csv
  timing/                                                     the timing benchmark's results,
      which ddml_timing/plan.py names
  timing/summary.csv                                          and their table

<sample> is empty for the full inputs, and _n<N> for a test on the first N
events of each (-n N). It is set once, when an input is simulated: the
simulation of the first N events of <input> is <input>_n<N>, and everything
made from it carries that name on, so -n N means the same sample at every
stage. A scenario whose inputs are simulated with ddsim seeds of its own
(simulation_seeds) adds _simseed<S> to <sample> for the seed S, likewise: the
simulation of <input> with seed 2 is <input>_simseed2, and of its first N
events <input>_n<N>_simseed2. (The inputs' own names may have _seed<S>, the
gun's seed, which this isn't.) The second Geant4 sample, geant4_2, has a seed
of its own for every simulation, the first's plus 1000000 (or 1000000 where
the first has ddsim's own), so its names always end in _simseed<S>: beside
the first's <input>_simseed2 it has <input>_simseed1000002, and beside the
first's <input> (ddsim's own seed), <input>_simseed1000000.

A scenario whose inputs are in segments (input.segments: N) has each of its
inputs' events in N files of their own, the i-th segment of <whole> named
<whole>_seg<i>of<N>: the first events of the whole in the first, and so on,
as evenly as they go (segment_bounds()). Each segment is an input like any
other, so its simulations and everything made from them are named after it,
e.g. <whole>_seg2of3_simseed1_REC.edm4hep.root. Files of the whole inputs
that are split into segments (scripts/split_data.py) go to UNSPLIT_DIR,
which is hidden, so nothing of the harness looks there.

Inputs given directly (simulate_*.sh --input) are simulated as the scenario
"custom"; simulations given directly to reconstruct_simulation.sh are
reconstructed where they belong if they are in sim_output, and to
rec_output/custom/files otherwise.

This module only needs the standard library, so that it works anywhere, e.g.
to plot .npz files copied to a laptop.
"""

import glob
import os
import re

# The folders of the inputs: the guns' (the default), and the physics samples'
INPUT_DIR = "input_single_particle_gun"
PHYSICS_INPUT_DIR = "input_physics_samples"
INPUT_FOLDERS = (INPUT_DIR, PHYSICS_INPUT_DIR)
SIM_DIR = "sim_output"
REC_DIR = "rec_output"
OBSERVABLES_DIR = "observables"
PLOTS_DIR = "plots"
# The timing benchmark's results (ddml_timing/, which names what goes in it)
TIMING_DIR = "timing"
# Their table, in TIMING_DIR (ddml_timing/summary.py), which the plots' Pareto
# fronts take the time per event from (ddml_bench/pareto.py)
TIMING_SUMMARY = "summary.csv"
# The folders the harness keeps for itself, which no inputs may go in
RESERVED_FOLDERS = (SIM_DIR, REC_DIR, OBSERVABLES_DIR, "slurm", TIMING_DIR)

# The reference simulator, which no model may be named after
GEANT4 = "geant4"
# Its second sample, which no model may be named after either: every input
# simulated with Geant4 again, with other ddsim seeds (second_sample_seed()),
# so that its distance from the first is how far apart two Geant4 samples of
# the same events are, Geant4's own (the Pareto fronts draw Geant4 at it)
SECOND_GEANT4 = "geant4_2"
# Its simulations' ddsim seeds are the first's plus this, so the scenarios'
# own seeds are below it (ddml_bench/scenarios.py)
SECOND_SEED_OFFSET = 1000000
# The scenario that inputs given directly are simulated as
CUSTOM_SCENARIO = "custom"
# rec_output/custom/files: simulations given directly from outside sim_output
GIVEN_FILES = "files"

# ddsim writes EDM4hep for a .root output file. Geant4's simulations have
# always been named .edm4hep.root, and the models' .root.
SIM_ENDINGS = {GEANT4: ".edm4hep.root", SECOND_GEANT4: ".edm4hep.root"}
MODEL_SIM_ENDING = ".root"
# The endings a finished simulation can have, longest first
SIMULATION_ENDINGS = (".edm4hep.root", ".root")

# The reconstructed file that marks a reconstruction done: EDM4hep, unless
# ILDReconstruction.py is told to write only LCIO (--lcioOutput only)
REC_SUFFIX = "_REC.edm4hep.root"
REC_SUFFIX_LCIO_ONLY = "_REC.slcio"

# Between the parts of .npz keys and of figure names
SEP = "__"

# What the names of the simulations of a ddsim seed have, before the seed
SEED_TAG = "_simseed"
# What the names of an input's segments have, before their number
SEGMENT_TAG = "_seg"
# Where scripts/split_data.py moves the files of the whole inputs it splits
# into segments, and of everything made from them, each at its path in the
# data folder: hidden, so that nothing of the harness or the manifests sees it
UNSPLIT_DIR = ".unsplit"

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+-]*$")
_SAMPLE = re.compile(r"_n\d+$")


class LayoutError(Exception):
    pass


def check_name(name, what):
    """name, if it can be a folder name (scenarios and simulators become folders)."""
    if not isinstance(name, str) or not _NAME.match(name):
        raise LayoutError("%s names become folder names, so use only letters, digits and _.+- "
                          "in them, starting with a letter or digit, not %r" % (what, name))
    return name


def check_input_folder(folder, what):
    """folder, if it can hold inputs: a folder of the data folder (one name, not
    a path), and not one of the harness's own."""
    check_name(folder, what)
    if folder in RESERVED_FOLDERS:
        raise LayoutError("%s can't be %s, which the pipeline writes its own files to"
                          % (what, folder))
    return folder


def sample_suffix(events):
    """What the names of the files of a sample end in: nothing for the full
    inputs, _n<N> for the first N events of each."""
    if events is None:
        return ""
    if not isinstance(events, int) or isinstance(events, bool) or events < 1:
        raise LayoutError("the number of events must be at least 1, not %r" % (events,))
    return "_n%d" % events


def seed_suffix(seed):
    """What the names of the simulation of an input with a ddsim seed, and of
    everything made from it, end in: nothing without a seed of its own (None),
    _simseed<S> with seed S."""
    if seed is None:
        return ""
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise LayoutError("a simulation seed must be a whole number, at least 0, not %r"
                          % (seed,))
    return "%s%d" % (SEED_TAG, seed)


def check_segments(segments, what):
    """The number of segments an input's events are in: a whole number, at
    least 1 (1 for the whole input in one file)."""
    if not isinstance(segments, int) or isinstance(segments, bool) or segments < 1:
        raise LayoutError("%s should be a whole number, at least 1 (the number of files each "
                          "input's events are in), not %r" % (what, segments))
    return segments


def segment_suffix(index, count):
    """What the name of the index-th of an input's count segments (from 1)
    ends in, after the whole input's: _seg<index>of<count>; nothing for a
    whole input (count 1)."""
    check_segments(count, "the number of segments")
    if count == 1:
        return ""
    if not isinstance(index, int) or isinstance(index, bool) or not 1 <= index <= count:
        raise LayoutError("there's no segment %r of %d" % (index, count))
    return "%s%dof%d" % (SEGMENT_TAG, index, count)


def segment_bounds(n_events, count):
    """[(first, last + 1)] of the events of each of count segments of
    n_events, in order: the first events in the first segment, and as many in
    each as can be, the first ones taking one more where they don't divide."""
    check_segments(count, "the number of segments")
    if not isinstance(n_events, int) or isinstance(n_events, bool) or n_events < count:
        raise LayoutError("%r events can't be split into %d segments of at least one event each"
                          % (n_events, count))
    size, extra = divmod(n_events, count)
    bounds, start = [], 0
    for index in range(count):
        stop = start + size + (1 if index < extra else 0)
        bounds.append((start, stop))
        start = stop
    return bounds


def second_sample_of(reference):
    """The simulator that is a second sample of the reference, simulated as
    it is: SECOND_GEANT4 of Geant4, None of any other."""
    return SECOND_GEANT4 if reference == GEANT4 else None


def second_sample_seed(seed):
    """The ddsim seed of the second Geant4 sample's simulation of an input
    that the first simulates with seed: seed + SECOND_SEED_OFFSET, or the
    offset alone where the first has ddsim's own (None), so that each of its
    simulations has a seed of its own, which no simulation of the first
    has."""
    return (seed or 0) + SECOND_SEED_OFFSET


def input_stem(path):
    """An input's name, which everything made from it is named after."""
    return os.path.splitext(os.path.basename(path))[0]


def simulation_stem(file_name):
    """A simulation's name without its ending, or None if the file isn't a
    finished simulation (unfinished ones are hidden)."""
    if file_name.startswith("."):
        return None
    for ending in SIMULATION_ENDINGS:
        if file_name.endswith(ending) and len(file_name) > len(ending):
            return file_name[:-len(ending)]
    return None


def simulations_in(directory):
    """The finished simulations in a folder, sorted."""
    if not os.path.isdir(directory):
        return []
    return [os.path.join(directory, name) for name in sorted(os.listdir(directory))
            if simulation_stem(name) and os.path.isfile(os.path.join(directory, name))]


def figure_base(folder, observable, profile, scenario, suffix=""):
    """A figure's path, without the ending of its format."""
    return os.path.join(folder, "%s_%s%s%s%s" % (observable, profile, SEP, scenario, suffix))


def timing_summary_of(observables_dir):
    """The timing benchmark's table of the data folder whose observables folder
    is observables_dir: <data_dir>/timing/summary.csv, beside it."""
    return os.path.join(os.path.dirname(os.path.abspath(observables_dir)), TIMING_DIR,
                        TIMING_SUMMARY)


def measurement_files(folder, suffix=""):
    """[(simulator, .npz file)] of one sample's measurements in a folder, by name."""
    found = []
    for path in sorted(glob.glob(os.path.join(glob.escape(folder), "*%s.npz" % suffix))):
        name = os.path.basename(path)[:-len(suffix + ".npz")]
        # Hidden files are unfinished; without a suffix, leave out the test samples
        if not name or name.startswith(".") or (suffix == "" and _SAMPLE.search(name)):
            continue
        found.append((name, path))
    return found


class Layout(object):
    """The paths of one sample, the full inputs or the first N events of each
    (events=N), under a data folder."""

    def __init__(self, data_dir, events=None):
        self.data_dir = data_dir
        self.events = events
        self.suffix = sample_suffix(events)

    # The inputs

    @property
    def input_dir(self):
        """The guns' folder of inputs, the default."""
        return self.inputs(INPUT_DIR)

    def inputs(self, folder):
        """A folder of inputs (a scenario's input.folder)."""
        return os.path.join(self.data_dir, folder)

    # The simulations

    @property
    def sim_dir(self):
        return os.path.join(self.data_dir, SIM_DIR)

    def sample_name(self, input_file, seed=None):
        """The name of an input's simulation in this sample (with the ddsim
        seed, if it has one of its own), which everything made from that
        simulation is named after too."""
        return input_stem(input_file) + self.suffix + seed_suffix(seed)

    def simulation(self, scenario, simulator, input_file, seed=None):
        ending = SIM_ENDINGS.get(simulator, MODEL_SIM_ENDING)
        return os.path.join(self.sim_dir, scenario, simulator,
                            self.sample_name(input_file, seed) + ending)

    def simulation_log(self, scenario, simulator, input_file, seed=None):
        return os.path.join(self.sim_dir, scenario, simulator,
                            self.sample_name(input_file, seed) + ".log")

    # The reconstructions

    @property
    def rec_dir(self):
        return os.path.join(self.data_dir, REC_DIR)

    def reconstruction_base(self, scenario, simulator, input_file, seed=None):
        """What a reconstruction's outputs are named after: <base>_REC.edm4hep.root,
        <base>_AIDA.root, <base>_PfoAnalysis.root and <base>.log."""
        return os.path.join(self.rec_dir, scenario, simulator,
                            self.sample_name(input_file, seed))

    def given_simulation(self, path):
        """(scenario, simulator, reconstruction base) of a simulation file given
        directly: where it belongs if it is in sim_output, and custom/files if not."""
        real = os.path.realpath(path)
        stem = simulation_stem(os.path.basename(real))
        if stem is None:
            raise LayoutError("%s isn't a finished simulation: an EDM4hep file named *.root or "
                              "*.edm4hep.root (not hidden)" % path)
        parts = os.path.relpath(real, os.path.realpath(self.sim_dir)).split(os.sep)
        if len(parts) == 3 and parts[0] != os.pardir:
            scenario, simulator = parts[0], parts[1]
        else:
            scenario, simulator = CUSTOM_SCENARIO, GIVEN_FILES
        return scenario, simulator, os.path.join(self.rec_dir, scenario, simulator, stem)

    # The measurements, and their plots

    @property
    def observables_dir(self):
        return os.path.join(self.data_dir, OBSERVABLES_DIR)

    def observables(self, simulator):
        return os.path.join(self.observables_dir, simulator + self.suffix + ".npz")

    def measurements(self):
        """[(simulator, .npz file)] of this sample measured so far."""
        return measurement_files(self.observables_dir, self.suffix)

    @property
    def plots_dir(self):
        return os.path.join(self.observables_dir, PLOTS_DIR)

    @property
    def metrics(self):
        return os.path.join(self.plots_dir, "metrics%s.csv" % self.suffix)

    @property
    def timing_summary(self):
        """The timing benchmark's table (made by timing_summary.sh)."""
        return os.path.join(self.data_dir, TIMING_DIR, TIMING_SUMMARY)

    # Records of files, e.g. in the .npz files

    def relative(self, path):
        """A path in the data folder relative to it, so that a record of it holds
        wherever the data is copied to; any other path as it is."""
        relative = os.path.relpath(path, self.data_dir)
        if relative == os.pardir or relative.startswith(os.pardir + os.sep):
            return path
        return relative

    def absolute(self, path):
        """The path a record made by relative() stands for."""
        return path if os.path.isabs(path) else os.path.join(self.data_dir, path)
