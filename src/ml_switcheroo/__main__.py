"""
Entry point for module execution (``python -m ml_switcheroo``).

This module delegates execution to the CLI handler in ``ml_switcheroo.cli.__main__``.
"""

import sys
from ml_switcheroo.cli.__main__ import main

if __name__ == "__main__":
  sys.exit(main())
