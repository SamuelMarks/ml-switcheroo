"""Test suite for code extractor dedent."""

from unittest.mock import patch

from ml_switcheroo.utils.code_extractor import CodeExtractor


def test_code_extractor_dedent() -> None:
  """Verify `CodeExtractor.extract_class` handles code extractor dedent as expected."""
  with patch("inspect.getsource", return_value="    class Foo:\n        pass\n"):
    res: str = CodeExtractor.extract_class(type("Foo", (), {}))
    assert res == "class Foo:\n    pass\n"
