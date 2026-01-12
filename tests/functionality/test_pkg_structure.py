"""
Tests for Package integrity and enumerations.
Renamed from test_structure.py to avoid collision with core pass tests.
"""

from ml_switcheroo.enums import SemanticTier


def test_semantic_tier_enum():
  """Ensure we can import and use the Enums defined in Step 1."""
  assert SemanticTier.ARRAY_API == "array"


def test_dependency_check():
  """Ensure required libs are installed."""
  import libcst
  import rich
  import griffe

  assert libcst.LIBCST_VERSION
  assert rich.inspect
  assert griffe.inspect
