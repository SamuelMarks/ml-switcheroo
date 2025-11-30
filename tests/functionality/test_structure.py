from ml_switcheroo.enums import SupportedEngine, SemanticTier


def test_enums_are_accessible():
  """Ensure we can import and use the Enums defined in Step 1."""
  assert SupportedEngine.TORCH == "torch"
  assert SupportedEngine.JAX == "jax"
  assert SemanticTier.ARRAY_API == "array"


def test_dependency_check():
  """Ensure required libs are installed."""
  import libcst
  import rich
  import griffe

  assert libcst.LIBCST_VERSION
  assert rich.inspect
  assert griffe.inspect
