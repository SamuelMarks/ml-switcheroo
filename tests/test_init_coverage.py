"""Test suite for the Init Coverage module."""

from unittest.mock import MagicMock, patch

import pytest

import ml_switcheroo
from ml_switcheroo.semantics.manager import SemanticsManager


def test_convert_success() -> None:
  """Converts successfully."""
  with patch("ml_switcheroo.ASTEngine") as MockEngine:
    MockEngine.return_value.run.return_value = MagicMock(success=True, code="test")
    result: str = ml_switcheroo.convert("code", source="torch", target="jax")
    assert result == "test"


def test_convert_with_semantics() -> None:
  """Converts with semantics."""
  manager: SemanticsManager = SemanticsManager()
  with patch("ml_switcheroo.ASTEngine") as MockEngine:
    MockEngine.return_value.run.return_value = MagicMock(success=True, code="test")
    result: str = ml_switcheroo.convert("code", source="torch", target="jax", semantics=manager)
    assert result == "test"


def test_convert_failure() -> None:
  """Converts successfully handling failure."""
  with patch("ml_switcheroo.ASTEngine") as MockEngine:
    MockEngine.return_value.run.return_value = MagicMock(success=False, errors=["some error"])
    with pytest.raises(ValueError, match="Transpilation failed.*"):
      ml_switcheroo.convert("code", source="torch", target="jax")
