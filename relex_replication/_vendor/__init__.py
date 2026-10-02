"""Third-party source snapshots, kept byte-identical to upstream (see each NOTICE.md)."""

import sys
from pathlib import Path

# BFCL and Google IFEval import themselves by top-level package name; exposing
# this directory lets their files stay unmodified.
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))
