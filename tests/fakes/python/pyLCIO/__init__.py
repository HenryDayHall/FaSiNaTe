"""A stand-in for pyLCIO, for the tests: just what ILD_Particle_Gun.py,
ILD_Di_Photon_Gun.py and ddml_bench/split.py use.

The "LCIO" files it writes are JSON, one line per record: the run header,
then each event's MC particles. tests/fakes/bin/ddsim reads them, and so does
its reader, which gives the records as the classes the guns write them with,
so that what it reads can be written again, as with LCIO.
"""

import json


class _Parameters(object):
    def __init__(self):
        self.values = {}

    def setValue(self, key, value):
        self.values[key] = value if isinstance(value, (int, str)) else float(value)


class LCRunHeaderImpl(object):
    def __init__(self):
        self._parameters = _Parameters()

    def parameters(self):
        return self._parameters

    @classmethod
    def read(cls, record):
        run = cls()
        run._parameters.values = dict(record["run"])
        return run


class MCParticleImpl(object):
    def __init__(self):
        self.record = {}

    def setGeneratorStatus(self, value):
        self.record["generatorStatus"] = int(value)

    def setMass(self, value):
        self.record["mass"] = float(value)

    def setPDG(self, value):
        self.record["PDG"] = int(value)

    def setMomentum(self, value):
        self.record["momentum"] = [float(v) for v in value]

    def setCharge(self, value):
        self.record["charge"] = float(value)

    def setVertex(self, value):
        self.record["vertex"] = [float(v) for v in value]

    def setEndpoint(self, value):
        self.record["endpoint"] = [float(v) for v in value]

    def getPDG(self):
        return self.record["PDG"]

    def getGeneratorStatus(self):
        return self.record["generatorStatus"]

    def getMomentum(self):
        return list(self.record["momentum"])


class LCCollectionVec(object):
    def __init__(self, kind):
        self.kind = kind
        self.elements = []

    def addElement(self, element):
        self.elements.append(element)

    def getNumberOfElements(self):
        return len(self.elements)

    def getTypeName(self):
        return self.kind

    def __iter__(self):
        return iter(self.elements)


class LCEventImpl(object):
    def __init__(self):
        self.number = 0
        self.collections = {}

    def setEventNumber(self, number):
        self.number = int(number)

    def addCollection(self, collection, name):
        self.collections[name] = collection

    def getEventNumber(self):
        return self.number

    def getRunNumber(self):
        return 0

    def getCollectionNames(self):
        return list(self.collections)

    def getCollection(self, name):
        return self.collections[name]

    @classmethod
    def read(cls, record):
        """An event as the writer wrote it: its MC particles, of any fields."""
        event = cls()
        event.setEventNumber(record["event"])
        collection = LCCollectionVec("MCParticle")
        for fields in record["particles"]:
            particle = MCParticleImpl()
            particle.record = dict(fields)
            collection.addElement(particle)
        event.addCollection(collection, "MCParticle")
        return event


class LCWriter(object):
    def open(self, path, mode):
        self.stream = open(path, "w")

    def writeRunHeader(self, run):
        self.stream.write(json.dumps({"run": run.parameters().values}) + "\n")

    def writeEvent(self, event):
        particles = [element.record for collection in event.collections.values()
                     for element in collection.elements]
        self.stream.write(json.dumps({"event": event.number, "particles": particles}) + "\n")

    def close(self):
        self.stream.close()


class LCReader(object):
    """Reads the records of a file in order, as LCIO's reader does: each
    readNext... reads on to the next record of its kind."""

    def open(self, path):
        with open(path) as stream:
            self.records = [json.loads(line) for line in stream if line.strip()]
        self.position = 0

    def getNumberOfEvents(self):
        return sum(1 for record in self.records if "event" in record)

    def _next(self, kind):
        while self.position < len(self.records):
            record = self.records[self.position]
            self.position += 1
            if kind in record:
                return record
        return None

    def readNextRunHeader(self):
        record = self._next("run")
        return LCRunHeaderImpl.read(record) if record else None

    def readNextEvent(self):
        record = self._next("event")
        return LCEventImpl.read(record) if record else None

    def skipNEvents(self, n):
        for _ in range(n):
            if self._next("event") is None:
                break

    def close(self):
        self.records = []


class LCFactory(object):
    @classmethod
    def getInstance(cls):
        return cls()

    def createLCWriter(self):
        return LCWriter()

    def createLCReader(self):
        return LCReader()


class _Namespace(object):
    def __init__(self, **items):
        self.__dict__.update(items)


EVENT = _Namespace(LCIO=_Namespace(WRITE_NEW=1, MCPARTICLE="MCParticle"))
IOIMPL = _Namespace(LCFactory=LCFactory)
IMPL = _Namespace(LCRunHeaderImpl=LCRunHeaderImpl, LCCollectionVec=LCCollectionVec,
                  LCEventImpl=LCEventImpl, MCParticleImpl=MCParticleImpl)
