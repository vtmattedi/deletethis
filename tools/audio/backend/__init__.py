"""Local backend for the INMP441 detector.

The ESP32 stays simple: Serial for debug, one TCP socket for binary
audio. This package is the only thing that connects to it, and it
owns reception, feature extraction, classification, runtime config
and event recording. The browser only displays what it is told.
"""

import sys
from pathlib import Path

# The existing tools live one directory up and are imported as-is
# rather than copied or refactored.
_TOOLS = str(Path(__file__).resolve().parent.parent)

if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)
