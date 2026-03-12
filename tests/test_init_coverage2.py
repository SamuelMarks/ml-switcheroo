import pytest
import ml_switcheroo
from unittest.mock import patch, MagicMock


def test_convert_success():
  with patch("ml_switcheroo.ASTEngine") as MockEngine:
    MockEngine.return_value.run.return_value = MagicMock(success=True, code="test")
    res = ml_switcheroo.convert("code", source="torch", target="jax")
    assert res == "test"


def test_convert_failure():
  with patch("ml_switcheroo.ASTEngine") as MockEngine:
    MockEngine.return_value.run.return_value = MagicMock(success=False, errors=["some error"])
    with pytest.raises(ValueError, match="Transpilation failed.*"):
      ml_switcheroo.convert("code", source="torch", target="jax")


def test_convert_with_semantics():
  with patch("ml_switcheroo.ASTEngine") as MockEngine:
    MockEngine.return_value.run.return_value = MagicMock(success=True, code="test2")
    res = ml_switcheroo.convert("code", source="torch", target="jax", semantics=MagicMock())
    assert res == "test2"
