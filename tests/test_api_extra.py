"""Tests for extra functionalities of the API transformer."""

from ml_switcheroo.core.rewriter.passes.api import ApiTransformer


def test_api_transformer_version_exceptions(monkeypatch):
  """Test that exceptions during version checking are handled gracefully."""
  import importlib.metadata

  class MockSemantics:
    """Mock semantics."""

    def get_framework_config(self, fw):
      """Get framework config."""
      return {}

  class DummyContext:
    """Dummy context."""

    def __init__(self):
      """Init."""
      self.target_fw = "flax_nnx"
      self.semantics = MockSemantics()

  p = ApiTransformer(context=DummyContext())

  def mock_version(pkg):
    """Mock version."""
    raise Exception("Mocked Exception")

  monkeypatch.setattr(importlib.metadata, "version", mock_version)

  res = p.check_version_constraints("1.0", "2.0")
  assert res is None
