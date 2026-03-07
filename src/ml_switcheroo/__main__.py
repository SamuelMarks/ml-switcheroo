"""
Entry point for module execution (``python -m ml_switcheroo``).

This module delegates execution to the CLI handler in ``ml_switcheroo.cli.__main__``.
"""

import sys  # pragma: no cover
from ml_switcheroo.cli.__main__ import main  # pragma: no cover

if __name__ == "__main__":  # pragma: no cover
  sys.exit(main())  # pragma: no cover
