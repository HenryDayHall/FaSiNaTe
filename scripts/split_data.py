#!/usr/bin/env python3
"""Split what was made from whole inputs into the segments of the inputs that
the configuration now names (input.segments): the inputs, simulations,
reconstructions and their logs, each file event by event, checked, with the
whole files moved to <data_dir>/.unsplit. Without --apply it only says what
it would do.

  python3 scripts/split_data.py                 # what it would split
  python3 scripts/split_data.py --apply         # split
  python3 scripts/split_data.py --apply --only inputs
  python3 scripts/split_data.py --apply --only geant4

Run it in the repository the pipeline runs from, before the pipeline, with
the same configuration (--config, or FASINATE_CONFIG): it goes into the
environment from setup.sh itself. See ddml_bench/split.py.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from ddml_bench.split import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
