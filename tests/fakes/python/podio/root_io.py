"""podio.root_io, for the tests: frames of the files the fakes write as
EDM4hep, read and written back as podio does it, category by category.

  the fake ddsim's simulations   JSON: an "events" frame for each event, its
                                 number a parameter (as an event's header is
                                 in EDM4hep), and a "metadata" frame of the
                                 rest (simulator, detector, seed, models)
  the fake k4run's REC files     an "events" tree, written with uproot: a
                                 frame for each entry, its collections the
                                 names before the "." of the branches

A Writer writes them back as they were read, when it finishes.
"""

import json


class Collection(object):
    def __init__(self, size):
        self._size = size

    def size(self):
        return self._size


class Frame(object):
    def __init__(self, kind, data):
        self.kind = kind  # "json", "metadata" or "root"
        self.data = data
        self.parameters = ("EventNumber",) if kind == "json" and "event" in data else ()

    def get_parameter(self, name):
        if name not in self.parameters:
            raise KeyError(name)
        return self.data["event"]

    def _sizes(self):
        if self.kind == "json":
            return {"MCParticles": len(self.data["particles"])}
        if self.kind == "metadata":
            return {}
        sizes = {}
        for field in self.data.fields:
            sizes.setdefault(field.split(".")[0], len(self.data[field][0]))
        return sizes

    def getAvailableCollections(self):
        return tuple(sorted(self._sizes()))

    def get(self, name):
        sizes = self._sizes()
        if name not in sizes:
            raise KeyError(name)
        return Collection(sizes[name])


class FrameCategoryIterator(object):
    def __init__(self, frames):
        self._frames = frames

    def __len__(self):
        return len(self._frames)

    def __getitem__(self, entry):
        return self._frames[entry]

    def __iter__(self):
        return iter(self._frames)


class Reader(object):
    def __init__(self, filenames):
        path = filenames if isinstance(filenames, str) else filenames[0]
        with open(path, "rb") as stream:
            json_file = stream.read(1) == b"{"
        if json_file:
            with open(path) as stream:
                data = json.load(stream)
            events = data.pop("events")
            self._frames = {"events": [Frame("json", event) for event in events],
                            "metadata": [Frame("metadata", data)]}
        else:
            import uproot
            with uproot.open(path) as stream:
                tree = stream["events"]
                # (not the counts of the jagged branches, which uproot adds)
                counts = {tree[name].count_branch.name for name in tree.keys()
                          if getattr(tree[name], "count_branch", None) is not None}
                arrays = tree.arrays([name for name in tree.keys() if name not in counts],
                                     library="ak")
            self._frames = {"events": [Frame("root", arrays[i:i + 1])
                                       for i in range(len(arrays))]}
        self._categories = tuple(self._frames)

    @property
    def categories(self):
        return self._categories

    def get(self, category):
        return FrameCategoryIterator(self._frames[category])


class Writer(object):
    def __init__(self, filename):
        self._path = filename
        self._frames = {}
        self._done = False

    def write_frame(self, frame, category, collections=None):
        self._frames.setdefault(category, []).append(frame)

    def finish(self):
        if self._done:
            return
        self._done = True
        events = self._frames.get("events", [])
        if any(frame.kind == "root" for frame in events):
            import awkward as ak
            import uproot
            merged = ak.concatenate([frame.data for frame in events])
            columns = dict((name, merged[name]) for name in merged.fields)
            with uproot.recreate(self._path) as stream:
                stream.mktree("events", dict((name, column.type)
                                             for name, column in columns.items()))
                stream["events"].extend(columns)
            return
        data = {}
        for frame in self._frames.get("metadata", []):
            data.update(frame.data)
        data["events"] = [frame.data for frame in events]
        with open(self._path, "w") as stream:
            json.dump(data, stream)


class RNTupleReader(object):
    """None of the fakes' files are RNTuples."""


class RNTupleWriter(Writer):
    pass
