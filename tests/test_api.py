"""Tests for extra functionalities of the API transformer."""

import typing


from ml_switcheroo.core.rewriter.passes.api import ApiTransformer


def test_api_transformer_version_exceptions(monkeypatch: typing.Any) -> None:
  """Test that exceptions during version checking are handled gracefully.

  Args:
      monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture.
  """
  import importlib.metadata

  class MockSemantics:
    """Mock semantics."""

    def get_framework_config(self, fw: typing.Any) -> typing.Any:
      """Get framework config.

      Args:
          fw (str): Target framework string.

      Returns:
          Dict[str, Any]: Configuration dictionary.
      """
      return {}

  class DummyContext:
    """Dummy context."""

    def __init__(self) -> None:
      """Init."""
      self.target_fw: typing.Any = "flax_nnx"
      self.semantics: typing.Any = MockSemantics()

  p: typing.Any = ApiTransformer(context=DummyContext())

  def mock_version(pkg: typing.Any) -> typing.Any:
    """Mock version.

    Args:
        pkg (str): Package string.

    Raises:
        Exception: Mock exception.

    Returns:
        str: Version string.
    """
    raise Exception("Mocked Exception")

  monkeypatch.setattr(importlib.metadata, "version", mock_version)

  res: typing.Any = p.check_version_constraints("1.0", "2.0")
  assert res is None
