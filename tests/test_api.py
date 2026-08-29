"""Tests for extra functionalities of the API transformer."""

from typing import Any, Dict, Optional

import pytest

from ml_switcheroo.core.rewriter.passes.api import ApiTransformer


def test_api_transformer_version_exceptions(monkeypatch: pytest.MonkeyPatch) -> None:
  """Test that exceptions during version checking are handled gracefully.

  Args:
      monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture.
  """
  import importlib.metadata

  class MockSemantics:
    """Mock semantics."""

    def get_framework_config(self, fw: str) -> Dict[str, Any]:
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
      self.target_fw: str = "flax_nnx"
      self.semantics: MockSemantics = MockSemantics()

  p: ApiTransformer = ApiTransformer(context=DummyContext())  # type: ignore

  def mock_version(pkg: str) -> str:
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

  res: Optional[bool] = p.check_version_constraints("1.0", "2.0")
  assert res is None
