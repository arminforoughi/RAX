"""Put ``src/`` on the path so the suite runs from a clone, with no install step.

``pip install -e .`` is the supported way to use this repo, but a contributor whose
first action is ``pytest`` should not be met with an ImportError — and CI should be
testing the working tree rather than whatever happens to be installed in the
environment.
"""

import pathlib
import sys

SRC = pathlib.Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
