"""Test suite for the Init Gap module."""

from unittest.mock import MagicMock, patch

import pytest

import ml_switcheroo
from ml_switcheroo.semantics.manager import SemanticsManager


def test_init_convert_success() -> None:
  """Verifies the behavior of initialization convert successfully."""
  with patch("ml_switcheroo.ASTEngine") as MockEngine:
    with patch("ml_switcheroo.config.RuntimeConfig.load"):
      MockEngine.return_value.run.return_value = MagicMock(success=True, code="finalcode", errors=[])
      res: str = ml_switcheroo.convert("code")
      assert res == "finalcode"


def test_init_convert_failure() -> None:
  """Verifies the behavior of initialization convert successfully handling failure."""
  with patch("ml_switcheroo.ASTEngine") as MockEngine:
    with patch("ml_switcheroo.config.RuntimeConfig.load"):
      MockEngine.return_value.run.return_value = MagicMock(success=False, errors=["boom"])
      with pytest.raises(ValueError, match="Transpilation failed:\nboom"):
        ml_switcheroo.convert("code")


def test_init_convert_semantics() -> None:
  """Verifies the behavior of initialization convert semantics."""
  sm: SemanticsManager = SemanticsManager()
  with patch("ml_switcheroo.ASTEngine") as MockEngine:
    with patch("ml_switcheroo.config.RuntimeConfig.load"):
      MockEngine.return_value.run.return_value = MagicMock(success=True, code="finalcode", errors=[])
      res: str = ml_switcheroo.convert("code", semantics=sm)
      assert res == "finalcode"
