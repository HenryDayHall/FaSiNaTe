#!/usr/bin/env python3
"""Print FaSiNaTe's merged configuration: see ddml_bench/config.py.

  python3 scripts/fasinate_config.py [--config FILE] [--set SECTION.KEY=VALUE]
                                     [--get SECTION.KEY | --shell]

scripts/load_config.sh and the build scripts run this with --shell.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from ddml_bench.config import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
