"""Splitting what was made from whole inputs into the segments of those
inputs: scripts/split_data.py.

With input.segments: N, the pipeline takes the events of each of a scenario's
inputs in N files, the segments, <whole>_seg<i>of<N> (ddml_bench/layout.py),
each an input of its own, so that each simulation and reconstruction is a job
of 1/N of the events. What was made from the whole inputs before is the same
events, in the same order, so this splits it rather than having the pipeline
make it all again: of each whole input of a scenario with segments,

  <folder>/<whole>.slcio            the input (LCIO), event by event, into the
  <folder>/<whole>.h5               segments the configuration names, and the
                                    gun's truth beside it likewise
  sim_output/<scenario>/<simulator>/<whole><seed>.edm4hep.root (or .root)
  rec_output/<scenario>/<simulator>/<whole><seed>_REC.edm4hep.root (_REC.slcio)
                                    each simulation and reconstruction (EDM4hep,
                                    podio's frames; LCIO), into the same
                                    segments: ddsim and the reconstruction keep
                                    the events of their input, in order, so the
                                    events of segment i are those of the input's
                                    segment i
  ...<whole><seed>.log              for each segment, a log with the header of
                                    the whole job's (where and when it ran, the
                                    software, the command), which the manifests
                                    read, and a line saying which of its events
                                    the segment has; no time taken, as the job
                                    wasn't the segment's
  ..._AIDA.root, _PfoAnalysis.root  histograms of the whole job, kept whole

of every simulator whose folder is there, in models/ or not. Each segment is
written to a hidden file beside where it goes, checked against the whole
(its number of events, every category of frames, and some of its events,
spread over it, collection by collection and value by value), and only then
put in its place. Once a file's segments are all there, the file moves to
<data_dir>/.unsplit/, at the same path (layout.UNSPLIT_DIR, which nothing of
the harness looks in), with the logs and the files kept whole: nothing is
deleted. Delete that folder once the pipeline has measured the segments again
(below); its logs keep the times the whole jobs took (scripts/plot_step_times.py
--data-dir <data_dir>/.unsplit).

Every file made from an input must have as many events as the input; one that
doesn't, or whose segments fail a check, is left as it is, with the rest of
its simulation or reconstruction, and said so. Segments there already (from
the pipeline, or a run of this that stopped) are checked rather than made, so
this can simply be run again; it does what is left to do. Test samples
(_n<N>) are left alone.

Without --apply it only says what it would do. The measurements
(observables/<simulator>.npz) record the reconstructions they were made
from, so once those are split they are out of date, and the pipeline measures
them again from the segments, which gives the same numbers: --apply first
copies them and the metrics to .unsplit/observables/, to compare.

It needs the environment from setup.sh (pyLCIO, podio, h5py, and uproot for
the check of the values), and goes there itself (scripts/run_in_environment.sh)
when it isn't run there.
"""

import argparse
import errno
import glob
import importlib
import os
import re
import shutil
import sys
import time

from . import process
from .config import REPO_DIR, ConfigError, load_config
from .layout import (OBSERVABLES_DIR, PLOTS_DIR, REC_DIR, SEED_TAG, SEGMENT_TAG, SIM_DIR,
                     UNSPLIT_DIR, Layout, LayoutError, input_stem, measurement_files,
                     segment_bounds)

SCRIPT = os.path.join(REPO_DIR, "scripts", "split_data.py")
# What --only takes for the inputs, as well as the simulators
INPUTS = "inputs"

# The kinds of files: split by event (LCIO, the guns' truth, EDM4hep), a log,
# or kept whole
LCIO, TRUTH, EDM4HEP, LOG, KEPT = "LCIO", "truth", "EDM4hep", "log", "kept"
# The files of a simulation, and of a reconstruction, after the name of the
# whole input: the seed, if any, then the ending
_SEED = r"(?P<seed>%s\d+)?" % re.escape(SEED_TAG)
SIM_FILES = re.compile(r"^%s(?P<ending>\.edm4hep\.root|\.root|\.log)$" % _SEED)
REC_FILES = re.compile(r"^%s(?P<ending>_REC\.edm4hep\.root|_REC\.slcio|_AIDA\.root"
                       r"|_PfoAnalysis\.root|\.log)$" % _SEED)
KINDS = {".edm4hep.root": EDM4HEP, ".root": EDM4HEP, "_REC.edm4hep.root": EDM4HEP,
         "_REC.slcio": LCIO, "_AIDA.root": KEPT, "_PfoAnalysis.root": KEPT, ".log": LOG}
# The first file that names a test sample, after the whole input's name
_TEST_SAMPLE = re.compile(r"^_n\d+")
# How the hidden files written before they are checked start
PARTIAL = ".split."
# The end of a job's log (stages/simulate.py, stages/reconstruct.py)
_ENDED = re.compile(r"^# (finished|FAILED|INTERRUPTED)\b", re.M)
# The lines of a log's header that are the job's end, not its start
_HEADER_SKIP = re.compile(r"^# (finished|FAILED|INTERRUPTED|segment|whole log)\b")


class SplitError(Exception):
    pass


def say(message=""):
    print(message)
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# What there is to split


class Item(object):
    """A file of a whole input's: split by event into the segments' files
    (kind LCIO, TRUTH or EDM4HEP), a log (LOG, segments the segments' logs),
    or kept whole (KEPT, no segments)."""

    def __init__(self, path, kind, segments=None):
        self.path = path
        self.kind = kind
        self.segments = segments


class Sample(object):
    """A whole input, or one of its simulations or reconstructions: its files,
    which go together."""

    def __init__(self, what, name):
        self.what = what  # "input", or the simulator
        self.name = name  # <whole><seed>
        self.items = []

    def split_by_event(self):
        return [item for item in self.items if item.kind in (LCIO, TRUTH, EDM4HEP)]


class Chain(object):
    """A whole input of a scenario, its segments, and what was made from it
    there is to split (samples), and the files of its name left alone
    (others: test samples)."""

    def __init__(self, scenario, whole, segments):
        self.scenario = scenario
        self.whole = whole
        self.segments = segments
        self.samples = []
        self.others = []

    @property
    def name(self):
        return input_stem(self.whole)


def find_chains(config, data_dir, scenarios=(), only=()):
    """The chains of the scenarios whose inputs are in segments (or those
    named), with what of each there is in the data folder: its inputs, unless
    only names neither them (INPUTS) nor nothing, and the simulations and
    reconstructions of every simulator (or those named in only)."""
    from .scenarios import scenario_inputs
    groups = scenario_inputs(config, Layout(data_dir))
    known = [group["scenario"] for group in groups]
    for name in scenarios:
        if name not in known:
            raise SplitError("there is no scenario '%s'; the configuration has: %s"
                             % (name, ", ".join(known)))
    chains = []
    for group in groups:
        scenario = group["scenario"]
        if scenarios and scenario not in scenarios:
            continue
        wholes = {}
        for entry in group["inputs"]:
            if "whole" in entry:
                wholes.setdefault(entry["whole"], []).append(entry["file"])
        if scenarios and not wholes:
            raise SplitError("the inputs of the scenario %s aren't in segments "
                             "(scenarios.%s.input.segments)" % (scenario, scenario))
        for whole, segments in wholes.items():
            chain = Chain(scenario, whole, segments)
            stem = input_stem(whole)
            stems = [input_stem(path) for path in segments]
            if not only or INPUTS in only:
                sample = Sample("input", stem)
                sample.items.append(Item(whole, LCIO, segments))
                truth = os.path.splitext(whole)[0] + ".h5"
                truths = [os.path.splitext(path)[0] + ".h5" for path in segments]
                if os.path.exists(truth) or any(os.path.exists(path) for path in truths):
                    sample.items.append(Item(truth, TRUTH, truths))
                chain.samples.append(sample)
            for top, files in ((SIM_DIR, SIM_FILES), (REC_DIR, REC_FILES)):
                base = os.path.join(data_dir, top, scenario)
                names = sorted(os.listdir(base)) if os.path.isdir(base) else []
                for simulator in names:
                    folder = os.path.join(base, simulator)
                    if simulator.startswith(".") or not os.path.isdir(folder) \
                            or (only and simulator not in only):
                        continue
                    found = {}
                    for name in sorted(os.listdir(folder)):
                        if name.startswith(".") or not name.startswith(stem):
                            continue
                        rest = name[len(stem):]
                        match = files.match(rest)
                        if match is None:
                            # Of a test sample, say; not the segments'
                            if _TEST_SAMPLE.match(rest) or (
                                    rest.startswith("_")
                                    and not rest.startswith(SEGMENT_TAG)):
                                chain.others.append(os.path.join(folder, name))
                            continue
                        seed, ending = match.group("seed") or "", match.group("ending")
                        sample = found.setdefault(seed, Sample(simulator, stem + seed))
                        kind = KINDS[ending]
                        sample.items.append(Item(
                            os.path.join(folder, name), kind,
                            None if kind == KEPT else
                            [os.path.join(folder, other + seed + ending) for other in stems]))
                    chain.samples += [found[seed] for seed in sorted(found)]
            chains.append(chain)
    return chains


# ---------------------------------------------------------------------------
# The formats: how to count the events of a file, write those of a segment,
# and check a segment against the whole


def picks(n_events):
    """The events of a segment compared with the whole's: its first, middle
    and last."""
    return sorted({0, n_events // 2, n_events - 1})


def _same(first, second):
    """Whether two values from uproot (to_list()) are the same, NaN as NaN."""
    if isinstance(first, float) and isinstance(second, float):
        return first == second or (first != first and second != second)
    if isinstance(first, (list, tuple)) and isinstance(second, (list, tuple)):
        return len(first) == len(second) and all(_same(a, b) for a, b in zip(first, second))
    if isinstance(first, dict) and isinstance(second, dict):
        return sorted(first) == sorted(second) and all(_same(first[k], second[k])
                                                       for k in first)
    return first == second


class Lcio(object):
    """LCIO files (pyLCIO): the inputs, and the reconstructions' _REC.slcio."""

    def __init__(self):
        from pyLCIO import EVENT, IOIMPL
        self.EVENT, self.IOIMPL = EVENT, IOIMPL

    def reader(self, path):
        reader = self.IOIMPL.LCFactory.getInstance().createLCReader()
        reader.open(path)
        return reader

    def count(self, path):
        reader = self.reader(path)
        try:
            return int(reader.getNumberOfEvents())
        finally:
            reader.close()

    def write(self, path, start, stop, out):
        """Events start to stop - 1 of path into out, after its run headers,
        as the guns write them (one, before the events)."""
        writer = self.IOIMPL.LCFactory.getInstance().createLCWriter()
        writer.open(out, self.EVENT.LCIO.WRITE_NEW)
        try:
            reader = self.reader(path)
            try:
                run = reader.readNextRunHeader()
                while run:
                    writer.writeRunHeader(run)
                    run = reader.readNextRunHeader()
            finally:
                reader.close()
            reader = self.reader(path)
            try:
                if start:
                    reader.skipNEvents(start)
                for number in range(start, stop):
                    event = reader.readNextEvent()
                    if not event:
                        raise SplitError("%s ended after %d events" % (path, number))
                    writer.writeEvent(event)
            finally:
                reader.close()
        finally:
            writer.close()

    @staticmethod
    def describe(event):
        """An event's numbers, its collections and their sizes, and of its MC
        particles their PDG codes, generator statuses and momenta."""
        found = [int(event.getRunNumber()), int(event.getEventNumber())]
        for name in sorted(str(name) for name in event.getCollectionNames()):
            collection = event.getCollection(name)
            found.append((name, int(collection.getNumberOfElements())))
            try:
                if str(collection.getTypeName()) == "MCParticle":
                    particles = []
                    for p in collection:
                        # (a const double[3], which can't be iterated over)
                        momentum = p.getMomentum()
                        particles.append((int(p.getPDG()), int(p.getGeneratorStatus()),
                                          tuple(round(float(momentum[i]), 6) for i in range(3))))
                    found.append(tuple(particles))
            except Exception:  # what this pyLCIO can't give
                pass
        return found

    def events_at(self, path, indices):
        """{index: describe()} of the events of path at these indices."""
        found, position = {}, 0
        reader = self.reader(path)
        try:
            for index in sorted(set(indices)):
                if index > position:
                    reader.skipNEvents(index - position)
                event = reader.readNextEvent()
                if not event:
                    break
                found[index] = self.describe(event)
                position = index + 1
        finally:
            reader.close()
        return found

    def check(self, path, start, stop, segment):
        count = self.count(segment)
        if count != stop - start:
            return "it has %d events, not %d" % (count, stop - start)
        chosen = picks(stop - start)
        mine = self.events_at(segment, chosen)
        theirs = self.events_at(path, [start + index for index in chosen])
        for index in chosen:
            if mine.get(index) != theirs.get(start + index):
                return "its event %d isn't event %d of the whole" % (index, start + index)
        return None


class Podio(object):
    """EDM4hep files (podio's frames): the simulations, and the
    reconstructions' _REC.edm4hep.root. Frames are copied as podio-merge-files
    copies them, those of the events of the segment and every frame of the
    other categories (the run's, the file's metadata)."""

    EVENTS = "events"

    def __init__(self):
        try:
            # EDM4hep's dictionaries, to read its collections
            importlib.import_module("edm4hep")
        except ImportError:
            pass
        from podio import reading, root_io
        self.reading, self.root_io = reading, root_io

    def reader(self, path):
        reader = self.reading.get_reader(path)
        categories = list(reader.categories)
        if self.EVENTS not in categories:
            raise SplitError("%s has no %s (its categories: %s)"
                             % (path, self.EVENTS, ", ".join(categories) or "none"))
        return reader

    def writer(self, reader, path):
        rntuple = getattr(self.root_io, "RNTupleReader", None)
        if rntuple is not None and isinstance(reader, rntuple):
            return self.root_io.RNTupleWriter(path)
        if not isinstance(reader, self.root_io.Reader):
            raise SplitError("%s isn't a file of podio frames (%s)"
                             % (path, type(reader).__name__))
        return self.root_io.Writer(path)

    @staticmethod
    def finish(writer):
        # (The writers of podio before 1.6 have no finish(), but theirs)
        done = getattr(writer, "finish", None)
        if done is not None:
            done()
        else:
            writer._writer.finish()  # pylint: disable=protected-access

    def count(self, path):
        return len(self.reader(path).get(self.EVENTS))

    def write(self, path, start, stop, out):
        reader = self.reader(path)
        writer = self.writer(reader, out)
        try:
            events = reader.get(self.EVENTS)
            for index in range(start, stop):
                writer.write_frame(events[index], self.EVENTS)
            for category in reader.categories:
                if category != self.EVENTS:
                    for frame in reader.get(category):
                        writer.write_frame(frame, category)
        finally:
            self.finish(writer)

    @staticmethod
    def describe(frame):
        """A frame's collections and their sizes, and its parameters with their
        values."""
        parameters = []
        for name in sorted(getattr(frame, "parameters", ())):
            try:
                value = frame.get_parameter(name)
            except Exception:  # of several types under one name
                value = None
            parameters.append((name, [str(v) for v in value] if isinstance(value, list)
                               else str(value)))
        return ([(name, int(frame.get(name).size()))
                 for name in sorted(frame.getAvailableCollections())], parameters)

    def check(self, path, start, stop, segment):
        whole, part = self.reader(path), self.reader(segment)
        if sorted(part.categories) != sorted(whole.categories):
            return "its categories are %s, not %s" % (", ".join(sorted(part.categories)),
                                                      ", ".join(sorted(whole.categories)))
        for category in whole.categories:
            wanted = stop - start if category == self.EVENTS else len(whole.get(category))
            got = len(part.get(category))
            if got != wanted:
                return "it has %d frames of %s, not %d" % (got, category, wanted)
        mine, theirs = part.get(self.EVENTS), whole.get(self.EVENTS)
        chosen = picks(stop - start)
        for index in chosen:
            if self.describe(mine[index]) != self.describe(theirs[start + index]):
                return ("its event %d isn't event %d of the whole (its collections, their "
                        "sizes or its parameters differ)" % (index, start + index))
        # Value by value, the one in the middle (reading every branch of an
        # event takes a while)
        return same_entries(path, start, segment, chosen[len(chosen) // 2:][:1], self.EVENTS)


def same_entries(path, start, segment, chosen, tree):
    """Why the entries chosen of the segment's tree aren't entries start +
    chosen of the whole's, value by value, as uproot reads them (branch by
    branch: a podio file's are its collections' members), or None. Branches
    uproot can't read are passed over; without uproot, or for files that
    aren't ROOT's, nothing is compared."""
    try:
        import uproot
    except ImportError:
        return None
    try:
        whole, part = uproot.open(path), uproot.open(segment)
    except Exception:  # not a ROOT file
        return None
    with whole, part:
        # Everything of the whole is in the segment: nothing of the file
        # that isn't podio's frames is left behind
        lost = sorted(set(whole.keys(recursive=False, cycle=False))
                      - set(part.keys(recursive=False, cycle=False)))
        if lost:
            return "it hasn't the whole's %s" % ", ".join(lost)
        if tree not in whole or tree not in part:
            return None
        first, second = whole[tree], part[tree]
        theirs = set(second.keys())
        mine = [name for name in first.keys() if not first[name].branches]
        if [name for name in mine if name not in theirs]:
            return "it hasn't the branches of the whole: %s" % ", ".join(
                sorted(name for name in mine if name not in theirs)[:5])
        for name in mine:
            for index in chosen:
                try:
                    a = first[name].array(entry_start=start + index,
                                          entry_stop=start + index + 1, library="ak")
                    b = second[name].array(entry_start=index, entry_stop=index + 1,
                                           library="ak")
                except Exception:  # a type uproot can't read
                    break
                if not _same(a.to_list(), b.to_list()):
                    return "%s of its event %d isn't that of event %d of the whole" % (
                        name, index, start + index)
    return None


class Truth(object):
    """The guns' truth (.h5): every dataset of an entry per event, split."""

    def __init__(self):
        import h5py
        self.h5py = h5py

    def per_event(self, stream):
        return [(name, item) for name, item in stream.items()
                if isinstance(item, self.h5py.Dataset) and item.shape]

    def count(self, path):
        with self.h5py.File(path, "r") as stream:
            lengths = sorted({item.shape[0] for _, item in self.per_event(stream)})
        if len(lengths) != 1:
            raise SplitError("%s has datasets of %s entries, not one per event of one number "
                             "of events" % (path, ", ".join(str(n) for n in lengths) or "no"))
        return lengths[0]

    def write(self, path, start, stop, out):
        with self.h5py.File(path, "r") as source, self.h5py.File(out, "w") as target:
            for key, value in source.attrs.items():
                target.attrs[key] = value
            per_event = dict(self.per_event(source))
            for name, item in source.items():
                if name in per_event:
                    copy = target.create_dataset(name, data=item[start:stop])
                    for key, value in item.attrs.items():
                        copy.attrs[key] = value
                else:
                    source.copy(item, target, name=name)

    def check(self, path, start, stop, segment):
        import numpy as np
        with self.h5py.File(path, "r") as whole, self.h5py.File(segment, "r") as part:
            if sorted(whole.keys()) != sorted(part.keys()):
                return "its datasets are %s, not %s" % (", ".join(sorted(part.keys())),
                                                        ", ".join(sorted(whole.keys())))
            for name, item in self.per_event(whole):
                if not np.array_equal(item[start:stop], part[name][...],
                                      equal_nan=item.dtype.kind in "fc"):
                    return "its %s isn't that of events %d to %d of the whole" % (
                        name, start, stop - 1)
        return None


FORMATS = {LCIO: (Lcio, "pyLCIO"), EDM4HEP: (Podio, "podio"), TRUTH: (Truth, "h5py")}


class Formats(dict):
    """The formats, each set up when first needed."""

    def __missing__(self, kind):
        made, module = FORMATS[kind]
        try:
            self[kind] = made()
        except ImportError as error:
            raise SplitError("splitting %s files needs %s, which this environment doesn't have "
                             "(%s); run this in the environment from setup.sh"
                             % (kind, module, error))
        return self[kind]


# ---------------------------------------------------------------------------
# Splitting


class Splitter(object):
    """Splits chains, or says what it would do (apply False)."""

    def __init__(self, data_dir, apply):
        self.data_dir = data_dir
        self.apply = apply
        self.formats = Formats()
        self.problems = []
        self.done = {"split": 0, "there": 0, "moved": 0, "bytes": 0}

    def relative(self, path):
        return os.path.relpath(path, self.data_dir)

    def backup(self, path):
        """Where a whole file goes: .unsplit/<its path in the data folder>."""
        return os.path.join(self.data_dir, UNSPLIT_DIR, self.relative(path))

    def move_away(self, path):
        target = self.backup(path)
        if os.path.exists(target):
            raise SplitError("%s is in %s already, so %s can't go there; move one of them away"
                             % (self.relative(target), UNSPLIT_DIR, self.relative(path)))
        os.makedirs(os.path.dirname(target), exist_ok=True)
        try:
            os.rename(path, target)
        except OSError as error:
            if error.errno == errno.EXDEV:
                raise SplitError("%s is on another filesystem than %s, so it would have to be "
                                 "copied there; move it away yourself"
                                 % (path, os.path.join(self.data_dir, UNSPLIT_DIR)))
            raise SplitError("can't move %s to %s: %s" % (path, target, error.strerror))
        self.done["moved"] += 1

    def count(self, path, kind):
        try:
            return self.formats[kind].count(path)
        except SplitError:
            raise
        except Exception as error:  # what the format's library raises
            raise SplitError("can't read %s: %s: %s" % (self.relative(path),
                                                       type(error).__name__, error))

    def input_events(self, chain):
        """The events of a chain's input, and what they are counted from:
        the whole input, or its segments if it is split; (None, None) without
        either."""
        if os.path.exists(chain.whole):
            return self.count(chain.whole, LCIO), "its input"
        if chain.segments and all(os.path.exists(path) for path in chain.segments):
            counts = [self.count(path, LCIO) for path in chain.segments]
            total = sum(counts)
            wanted = [stop - start for start, stop in segment_bounds(total, len(counts))]
            if counts != wanted:
                raise SplitError("the segments of its input have %s events, not %s"
                                 % (", ".join(map(str, counts)), ", ".join(map(str, wanted))))
            return total, "its input's segments"
        return None, None

    def split_item(self, item, n_events, bounds):
        """Split a file into its segments, checking each (and those there
        already), then move the file away; what was done, or would be, in
        words."""
        fmt = self.formats[item.kind]
        size = os.path.getsize(item.path)
        count = self.count(item.path, item.kind)
        if count != n_events:
            raise SplitError("it has %d events, not %d as its input" % (count, n_events))
        there = []
        for (start, stop), target in zip(bounds, item.segments):
            if os.path.exists(target):
                why = fmt.check(item.path, start, stop, target)
                if why:
                    raise SplitError("its segment %s is there already, but isn't its events %d "
                                     "to %d (%s); move that away" % (
                                         os.path.basename(target), start, stop - 1, why))
                there.append(target)
        note = "%s%s" % (size_text(size), ", with %d of its segments there already, checked"
                         % len(there) if there else "")
        self.done["bytes"] += size
        if not self.apply:
            return "%s: to split" % note
        began = time.time()
        for (start, stop), target in zip(bounds, item.segments):
            if target in there:
                continue
            partial = os.path.join(os.path.dirname(target), "%s%s.%d.%s" % (
                PARTIAL, process.host(), os.getpid(), os.path.basename(target)))
            try:
                with process.interruptible():
                    fmt.write(item.path, start, stop, partial)
                    why = fmt.check(item.path, start, stop, partial)
                if why:
                    raise SplitError("its events %d to %d, written to %s, aren't them: %s"
                                     % (start, stop - 1, os.path.basename(partial), why))
                os.replace(partial, target)
            except (SplitError, process.Interrupted):
                raise
            except Exception as error:  # what the format's library raises
                raise SplitError("splitting it failed: %s: %s" % (type(error).__name__, error))
            finally:
                process.remove(partial)
        self.move_away(item.path)
        self.done["split"] += 1
        self.done["there"] += len(there)
        return "%s: split, in %d s" % (note, round(time.time() - began))

    def segment_logs(self, sample, item, n_events, bounds):
        """A log for each segment: the whole job's header, and which of its
        events the segment has."""
        head, _ = log_header(item.path)
        data = [other.path for other in sample.split_by_event()]
        made = ", whose output was %s" % self.relative(data[0]) if data else ""
        for index, ((start, stop), target) in enumerate(zip(bounds, item.segments), 1):
            if os.path.exists(target):
                continue
            lines = head + [
                "# segment   %d of %d: events %d to %d, %d of the %d of the job above%s, "
                "split into segments by scripts/split_data.py on %s" % (
                    index, len(bounds), start, stop - 1, stop - start, n_events, made,
                    process.now()),
                "# whole log %s" % self.relative(self.backup(item.path))]
            with open(target, "w") as stream:
                stream.write("\n".join(lines) + "\n")

    def split_others(self, sample, n_events, bounds):
        """A sample's log (a log for each segment, if its job finished: the
        others' aren't any segment's) and the files kept whole, which then move
        away; [(the file's ending, what of it)]."""
        notes = []
        for item in sample.items:
            if item.kind not in (LOG, KEPT) or not os.path.exists(item.path):
                continue
            ending = os.path.basename(item.path)[len(sample.name):]
            if item.kind == KEPT:
                notes.append((ending, "kept whole"))
            elif not bounds:
                notes.append((ending, "of nothing here to split"))
            elif log_header(item.path)[1]:
                notes.append((ending, "a log for each segment"))
                if self.apply:
                    self.segment_logs(sample, item, n_events, bounds)
            else:
                notes.append((ending, "of a job that didn't finish"))
            if self.apply:
                self.move_away(item.path)
        return notes

    def split_chain(self, chain):
        """Split what there is of a chain, saying what; True if nothing went
        wrong."""
        say("%s: %s" % (chain.scenario, chain.name))
        try:
            n_events, counted = self.input_events(chain)
        except SplitError as error:
            return self.problem(chain, None, "its input: %s" % error)
        there = [item for sample in chain.samples for item in sample.items
                 if os.path.exists(item.path)]
        if not there:
            say("  nothing of it to split%s" % (": split already" if n_events else ""))
        if n_events is None and there:
            data = [item for sample in chain.samples for item in sample.split_by_event()
                    if os.path.exists(item.path)]
            if data:
                try:
                    n_events = self.count(data[0].path, data[0].kind)
                except SplitError as error:
                    return self.problem(chain, None, str(error))
                counted = "%s, as its input isn't here, whole or in segments" % self.relative(
                    data[0].path)
        bounds = []
        if n_events is not None and there:
            try:
                bounds = segment_bounds(n_events, len(chain.segments))
            except LayoutError as error:
                return self.problem(chain, None, str(error))
            say("  %d events, counted from %s, in %d segments of %s" % (
                n_events, counted, len(bounds),
                ", ".join(str(stop - start) for start, stop in bounds)))
        good = True
        for sample in chain.samples:
            ok, said = True, False
            for item in sample.split_by_event():
                if not os.path.exists(item.path):
                    continue
                try:
                    say("  %s, %s" % (self.relative(item.path),
                                      self.split_item(item, n_events, bounds)))
                    said = True
                except SplitError as error:
                    ok = good = self.problem(chain, sample, "%s: %s" % (
                        self.relative(item.path), error))
                    break
            if not ok:
                say("    (the rest of its files are left as they are too)")
                continue
            try:
                notes = self.split_others(sample, n_events, bounds)
            except (SplitError, OSError) as error:
                good = self.problem(chain, sample, str(error))
                continue
            if notes:
                where = "%s to %s" % ("moved" if self.apply else "then", UNSPLIT_DIR)
                text = list_of(notes)
                if said:
                    say("    and its %s: %s" % (text, where))
                else:
                    say("  %s: its %s: %s" % (os.path.join(self.relative(os.path.dirname(
                        sample.items[0].path)), sample.name), text, where))
        if chain.others:
            say("  left alone: %s" % ", ".join(sorted(os.path.basename(path)
                                                     for path in chain.others)))
        return good

    def problem(self, chain, sample, why):
        self.problems.append("%s %s%s: %s" % (
            chain.scenario, chain.name,
            " (%s)" % sample.what if sample is not None and sample.what != "input" else "", why))
        say("  PROBLEM: %s" % why)
        return False


def list_of(notes):
    """[(file, what)] in words: files of the same what together."""
    groups = []
    for name, what in notes:
        if groups and groups[-1][1] == what:
            groups[-1][0].append(name)
        else:
            groups.append(([name], what))
    return ", ".join("%s (%s)" % (" and ".join(names), what) for names, what in groups)


def log_header(path):
    """(The lines of a job's log before its program's output, but its end,
    and whether the job finished)."""
    head = []
    with open(path, errors="replace") as stream:
        for line in stream:
            if not line.startswith("#") or len(head) >= 40:
                break
            line = line.rstrip("\n")
            if not _HEADER_SKIP.match(line):
                head.append(line)
    with open(path, "rb") as stream:
        stream.seek(0, os.SEEK_END)
        stream.seek(max(0, stream.tell() - 16384))
        tail = stream.read().decode("utf-8", "replace")
    ends = _ENDED.findall(tail)
    return head, bool(ends) and ends[-1] == "finished"


def size_text(size):
    for unit, scale in (("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if size >= scale:
            return "%.1f %s" % (size / scale, unit)
    return "%d bytes" % size


def keep_measurements(data_dir):
    """Copy the measurements (observables/*.npz) and the metrics
    (observables/plots/*.csv) to .unsplit/, where they aren't already, to
    compare with those measured again from the segments."""
    copied = []
    folder = os.path.join(data_dir, OBSERVABLES_DIR)
    for pattern in ("*.npz", os.path.join(PLOTS_DIR, "*.csv")):
        for path in sorted(glob.glob(os.path.join(glob.escape(folder), pattern))):
            target = os.path.join(data_dir, UNSPLIT_DIR, os.path.relpath(path, data_dir))
            if os.path.basename(path).startswith(".") or os.path.exists(target):
                continue
            os.makedirs(os.path.dirname(target), exist_ok=True)
            shutil.copy2(path, target)
            copied.append(target)
    return copied


def unmade_reconstructions(chains, data_dir):
    """[(.npz file, scenario, how many)] of the measurements in observables/
    whose reconstructions of a scenario split here aren't there, whole or in
    segments: once split, those measurements are out of date, and the
    pipeline can measure them again only by making those again."""
    try:
        import numpy as np
    except ImportError:
        return []
    by_name = {}  # (scenario, whole input's name): segments' names
    for chain in chains:
        by_name[(chain.scenario, chain.name)] = [input_stem(path) for path in chain.segments]
    layout = Layout(data_dir)
    found = []
    for simulator, path in measurement_files(os.path.join(data_dir, OBSERVABLES_DIR)):
        try:
            with np.load(path, allow_pickle=False) as data:
                recorded = dict((scenario, [str(f) for f in data["%s__files" % scenario]])
                                for scenario in {s for s, _ in by_name}
                                if "%s__files" % scenario in data.files)
        except (OSError, ValueError):
            continue
        for scenario, files in recorded.items():
            missing = 0
            for relative in files:
                if os.path.isabs(relative) and os.sep + REC_DIR + os.sep in relative:
                    relative = relative[relative.rindex(os.sep + REC_DIR + os.sep) + 1:]
                rec = layout.absolute(relative)
                if os.path.exists(rec):
                    continue
                folder, name = os.path.split(rec)
                segments = None
                for (other, whole), stems in by_name.items():
                    rest = name[len(whole):]
                    if other == scenario and name.startswith(whole) and REC_FILES.match(rest):
                        segments = [os.path.join(folder, stem + rest) for stem in stems]
                if not segments or not all(os.path.exists(p) for p in segments):
                    missing += 1
            if missing:
                found.append((os.path.relpath(path, data_dir), scenario, missing))
    return found


def run(config, data_dir, scenarios=(), only=(), apply=False):
    """Split the data folder's whole files of the scenarios with segments
    (or say what would be split); the exit status."""
    chains = find_chains(config, data_dir, scenarios, only)
    if not chains:
        say("split_data: no scenario of the configuration has its inputs in segments "
            "(input.segments), so there is nothing to split")
        return 0
    say("split_data: %s what was made from the whole inputs of %s into the segments of the "
        "inputs (input.segments), in %s%s" % (
            "splitting" if apply else "would split", ", ".join(dict.fromkeys(
                chain.scenario for chain in chains)), data_dir,
            "" if not only else ", only %s" % ", ".join(only)))
    if not apply:
        say("(nothing is changed without --apply)")
    folders = {name for chain in chains for top in (SIM_DIR, REC_DIR)
               for name in (os.listdir(os.path.join(data_dir, top, chain.scenario))
                            if os.path.isdir(os.path.join(data_dir, top, chain.scenario)) else [])}
    for name in only:
        if name != INPUTS and name not in folders:
            say("warning: --only %s: no simulator of that name has a folder in %s or %s of "
                "these scenarios" % (name, SIM_DIR, REC_DIR))
    say()
    leftovers = sorted(glob.glob(os.path.join(glob.escape(data_dir), "**", PARTIAL + "*"),
                                 recursive=True))
    splitter = Splitter(data_dir, apply)
    if apply:
        copied = keep_measurements(data_dir)
        if copied:
            say("copied %d measurement and metrics file(s) to %s, to compare with those "
                "measured from the segments\n" % (len(copied), os.path.join(
                    UNSPLIT_DIR, OBSERVABLES_DIR)))
    for chain in chains:
        try:
            splitter.split_chain(chain)
        except process.Interrupted as interrupted:
            say("\nsplit_data: interrupted; what was split is in place, and running this "
                "again does the rest")
            return interrupted.status
    say()
    done = splitter.done
    if apply and (done["split"] or done["moved"]):
        say("split_data: split %d file(s) (%s)%s; moved %d file(s) of the whole inputs, and "
            "made from them, to %s" % (
                done["split"], size_text(done["bytes"]),
                ", with %d of their segments there already, checked" % done["there"]
                if done["there"] else "", done["moved"], os.path.join(data_dir, UNSPLIT_DIR)))
    elif apply:
        say("split_data: nothing left to split")
    elif done["bytes"]:
        say("split_data: would split %s of files, writing about as much again; the whole "
            "files then go to %s (moved, not copied: their space is freed when you delete it)"
            % (size_text(done["bytes"]), os.path.join(data_dir, UNSPLIT_DIR)))
    else:
        say("split_data: nothing to split")
    for path, scenario, missing in unmade_reconstructions(chains, data_dir):
        say("warning: %s was measured on %d reconstruction(s) of %s that aren't here, whole or "
            "in segments; once split it is out of date, and the pipeline would make those "
            "again to measure it" % (path, missing, scenario))
    if leftovers:
        say("warning: %d hidden file(s) of a split that stopped (or that is running now), which "
            "you can delete once no split is running: %s"
            % (len(leftovers), ", ".join(os.path.relpath(p, data_dir) for p in leftovers[:5])))
    if splitter.problems:
        say("split_data: %d problem(s), whose files are left as they were:" % len(
            splitter.problems))
        for line in splitter.problems:
            say("  " + line)
        return 1
    if apply:
        say("split_data: next, ./slurm_pipeline.sh --dry-run (or pipeline.sh) should have "
            "only the measurements and the plots to do; compare the new metrics with those "
            "in %s, then delete %s" % (os.path.join(UNSPLIT_DIR, OBSERVABLES_DIR, PLOTS_DIR),
                                       os.path.join(data_dir, UNSPLIT_DIR)))
    else:
        say("split_data: run again with --apply to split")
    return 0


def main(argv=None):
    from .cli import UsageError, use_config_files
    from .environment import in_environment, run_in_environment
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="split_data.py",
        description="Split what was made from whole inputs (the inputs, simulations, "
                    "reconstructions and logs) into the segments the configuration names "
                    "for them (input.segments), moving the whole files to <data_dir>/.unsplit.",
        epilog="See ddml_bench/split.py for what it does.")
    parser.add_argument("-c", "--config", action="append", default=[], metavar="FILE",
                        help="merge FILE over the configuration (repeatable)")
    parser.add_argument("-s", "--scenario", action="append", default=[], metavar="NAME",
                        help="only this scenario (repeatable; default: every scenario whose "
                             "inputs are in segments)")
    parser.add_argument("--only", action="append", default=[], metavar="WHAT",
                        help="only the inputs (%s), or the simulations and reconstructions of "
                             "this simulator (repeatable; default: all of them)" % INPUTS)
    parser.add_argument("--apply", action="store_true",
                        help="split (without it, it only says what it would do)")
    args = parser.parse_args(argv)
    try:
        use_config_files(args.config)
        if not in_environment():
            forwarded = sum((["-s", name] for name in args.scenario), [])
            forwarded += sum((["--only", what] for what in args.only), [])
            forwarded += ["--apply"] if args.apply else []
            return run_in_environment(forwarded, entry=SCRIPT)
        config = load_config()
        return run(config, config["paths"]["data_dir"], args.scenario, args.only, args.apply)
    except (ConfigError, LayoutError, SplitError, UsageError) as error:
        print("split_data: %s" % error, file=sys.stderr)
        return 1
