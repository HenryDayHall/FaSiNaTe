#!/usr/bin/env python3
"""Fetch models of DDML's that its build doesn't download (ddml.extra_models),
as DDML's build fetches its own: by the md5 sum in DDML's models/<file>.md5,
from the URL template that has {md5} in it, checked against that sum, into
DDML's models/ folder. A model that is there with that sum is left as it is.
scripts/build_ddml.sh runs this after building DDML; it needs only the
standard library.

  fetch_ddml_models.py --url TEMPLATE --models-dir DIR [--dry-run] FILE ...
"""

import argparse
import hashlib
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from ddml_bench.fetch import FetchError, download  # noqa: E402

_MD5 = re.compile(r"^[0-9a-f]{32}$")


def say(message):
    print("fetch_ddml_models: " + message)
    sys.stdout.flush()


def md5_of(path):
    digest = hashlib.md5()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(names, models_dir, url, dry_run=False):
    """Fetch each model named into models_dir; returns the problems."""
    problems = []
    for name in names:
        if os.path.basename(name) != name or not name:
            problems.append("%r should be the name of a file in DDML's models/" % name)
            continue
        sum_file = os.path.join(models_dir, name + ".md5")
        try:
            with open(sum_file) as stream:
                md5 = stream.read().strip()
        except OSError:
            problems.append("%s isn't there: DDML has no model %s to fetch (ddml.extra_models)"
                            % (sum_file, name))
            continue
        if not _MD5.match(md5):
            problems.append("%s should hold an md5 sum, not %r" % (sum_file, md5))
            continue
        target = os.path.join(models_dir, name)
        if os.path.isfile(target) and md5_of(target) == md5:
            say("%s is there, md5 %s" % (target, md5))
            continue
        source = url.format(md5=md5)
        if dry_run:
            say("would fetch %s from %s" % (target, source))
            continue
        say("fetching %s from %s" % (target, source))
        try:
            size = download(source, target, md5=md5)
        except FetchError as error:
            problems.append(str(error))
            continue
        say("fetched %s, %.1f MB, md5 %s" % (target, size / 1e6, md5))
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("names", nargs="*", metavar="FILE")
    parser.add_argument("--url", required=True, help="the URL template, with {md5}")
    parser.add_argument("--models-dir", required=True, help="DDML's models/ folder")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if "{md5}" not in args.url:
        parser.error("--url should have {md5} in it, where the model's md5 sum goes")
    problems = fetch(args.names, args.models_dir, args.url, args.dry_run)
    for problem in problems:
        say("error: " + problem)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
