"""Test suite for the Pkg Structure module."""

from ml_switcheroo_ir.schema.ghost import SemanticTier


def test_semantic_tier_enum() -> None:
  """Verify semantic tier enum produces the expected state."""
  assert SemanticTier.ARRAY_API.value == "array"


def test_dependency_check() -> None:
  """Verify dependency check produces the expected state."""
  import libcst
  import rich

  assert libcst.LIBCST_VERSION
  assert "." in libcst.LIBCST_VERSION
  assert len(libcst.LIBCST_VERSION) >= 3
  assert rich.inspect is not None
  assert callable(rich.inspect)
