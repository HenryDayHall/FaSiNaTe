"""A stand-in for pyLCIO, for the tests: just what ILD_Particle_Gun.py uses.

The "LCIO" files it writes are JSON, one line per record: the run header,
then each event's MC particles. tests/fakes/bin/ddsim reads them.
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


class LCCollectionVec(object):
    def __init__(self, kind):
        self.kind = kind
        self.elements = []

    def addElement(self, element):
        self.elements.append(element)


class LCEventImpl(object):
    def __init__(self):
        self.number = 0
        self.collections = {}

    def setEventNumber(self, number):
        self.number = int(number)

    def addCollection(self, collection, name):
        self.collections[name] = collection


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


class LCFactory(object):
    @classmethod
    def getInstance(cls):
        return cls()

    def createLCWriter(self):
        return LCWriter()


class _Namespace(object):
    def __init__(self, **items):
        self.__dict__.update(items)


EVENT = _Namespace(LCIO=_Namespace(WRITE_NEW=1, MCPARTICLE="MCParticle"))
IOIMPL = _Namespace(LCFactory=LCFactory)
IMPL = _Namespace(LCRunHeaderImpl=LCRunHeaderImpl, LCCollectionVec=LCCollectionVec,
                  LCEventImpl=LCEventImpl, MCParticleImpl=MCParticleImpl)
