import pytest
import ml_switcheroo
from ml_switcheroo.semantics.manager import SemanticsManager
from unittest.mock import patch, MagicMock


def test_convert_success():
  with patch("ml_switcheroo.ASTEngine") as MockEngine:
    MockEngine.return_value.run.return_value = MagicMock(success=True, code="test")
    result = ml_switcheroo.convert("code", source="torch", target="jax")
    assert result == "test"


def test_convert_with_semantics():
  manager = SemanticsManager()
  with patch("ml_switcheroo.ASTEngine") as MockEngine:
    MockEngine.return_value.run.return_value = MagicMock(success=True, code="test")
    result = ml_switcheroo.convert("code", source="torch", target="jax", semantics=manager)
    assert result == "test"


def test_convert_failure():
  with patch("ml_switcheroo.ASTEngine") as MockEngine:
    MockEngine.return_value.run.return_value = MagicMock(success=False, errors=["some error"])
    with pytest.raises(ValueError, match="Transpilation failed.*"):
      ml_switcheroo.convert("code", source="torch", target="jax")
