"""Test suite for the Pkg Structure module."""

from ml_switcheroo_ir.schema.ghost import SemanticTier


def test_semantic_tier_enum():
  """Verifies the behavior of semantic tier enum."""
  assert SemanticTier.ARRAY_API == "array"


def test_dependency_check():
  """Verifies the behavior of dependency check."""
  import libcst
  import rich

  assert libcst.LIBCST_VERSION
  assert rich.inspect is not None
