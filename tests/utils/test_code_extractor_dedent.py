"""Docstring."""

from ml_switcheroo.utils.code_extractor import CodeExtractor
from unittest.mock import patch


def test_code_extractor_dedent():
  """Docstring."""
  with patch("inspect.getsource", return_value="    class Foo:\n        pass\n"):
    res = CodeExtractor.extract_class(type("Foo", (), {}))
    assert res == "class Foo:\n    pass\n"
