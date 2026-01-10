"""
Wrapper for the Python Backend (Legacy Compat).

Re-exports ``PythonBackend`` as ``GraphSynthesizer``.
"""

from ml_switcheroo.compiler.backends.python import PythonBackend as _Backend
from ml_switcheroo.compiler.backends.python import ClassBodyReplacer


class GraphSynthesizer(_Backend):
  """Legacy wrapper for PythonBackend."""

  pass


__all__ = ["GraphSynthesizer", "ClassBodyReplacer"]
