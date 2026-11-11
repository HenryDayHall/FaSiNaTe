"""podio.reading, for the tests: a reader for any file of the fakes."""

from . import root_io


def get_reader(filenames):
    return root_io.Reader(filenames)
