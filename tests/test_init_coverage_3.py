"""Test suite for the Init Coverage 3 module."""


def test_convert_error() -> None:
  """Converts correctly handling an error."""
  from unittest.mock import patch

  import ml_switcheroo
  from ml_switcheroo.core.conversion_result import ConversionResult

  with patch("ml_switcheroo.ASTEngine") as MockEngine:
    MockEngine.return_value.run.return_value = ConversionResult(
      success=True, code="finalcode", errors=[], trace_events=[]
    )
    assert ml_switcheroo.convert("code", source="torch", target="jax") == "finalcode"
