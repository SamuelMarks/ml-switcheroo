"""
Internal Standards Definition (Deprecated).

This file previously held the hardcoded "Gold Standard" definitions.
These have been migrated to JSON files in the `src/ml_switcheroo/semantics/` directory.

- k_array_api.json (Math)
- k_neural_net.json (Neural)
- k_framework_extras.json (Extras)

This file remains as an empty placeholder to prevent ImportErrors in
legacy code or tests that import `MATH_OPS` etc.
"""

import warnings as _warnings

# Emit warning on import to catch laggards
_warnings.warn(
  "Importing from standards_internal is deprecated. Use SemanticsManager or file_loader.",
  DeprecationWarning,
  stacklevel=2,
)

# Empty containers for backward compatibility
MATH_OPS = {}
NEURAL_OPS = {}
EXTRAS_OPS = {}
INTERNAL_OPS = {}
