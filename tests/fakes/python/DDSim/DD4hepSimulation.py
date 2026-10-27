"""A stand-in for DD4hep's DDSim.DD4hepSimulation, for the tests: a steering
file can give it any setting (SIM.<group>.<name> = value), and the physics
setup it is given (SIM.physics.setupUserPhysics) is kept, for the fake ddsim
to look at."""


class _Settings(object):
    def __init__(self):
        object.__setattr__(self, "_values", {})

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return self._values.setdefault(name, _Settings())

    def __setattr__(self, name, value):
        self._values[name] = value

    def setupUserPhysics(self, function):
        self._values["user_physics"] = function


class DD4hepSimulation(_Settings):
    pass
