import pytest


def test_convert_error():
  import ml_switcheroo
  from unittest.mock import patch, MagicMock
  from ml_switcheroo.core.conversion_result import ConversionResult

  with patch("ml_switcheroo.ASTEngine") as MockEngine:
    MockEngine.return_value.run.return_value = ConversionResult(
      success=True, code="finalcode", errors=[], trace_events=[]
    )
    assert ml_switcheroo.convert("code", source="torch", target="jax") == "finalcode"
