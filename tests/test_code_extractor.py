"""Docstring."""

import pytest
from ml_switcheroo.utils.code_extractor import CodeExtractor
from unittest.mock import patch


class DummyFuzzer:
  """Docstring."""

  def build_strategies(self):
    """Docstring."""
    pass


def test_extract_class():
  """Docstring."""
  with patch("inspect.getsource", return_value="class DummyFuzzer:\n    def build_strategies(self):\n        pass\n"):
    extracted = CodeExtractor.extract_class(DummyFuzzer)
    assert "class DummyFuzzer:" in extracted
    assert "build_strategies" in extracted


def test_extract_class_type_error():
  """Docstring."""
  with pytest.raises(TypeError):
    CodeExtractor.extract_class("not a class")


def test_extract_class_os_error():
  """Docstring."""
  with patch("inspect.getsource", side_effect=OSError("err")):
    with pytest.raises(OSError):
      CodeExtractor.extract_class(DummyFuzzer)


def test_normalize_harness_imports():
  """Docstring."""
  source = "class A: pass"
  reqs = ["math", "os.path"]
  code = CodeExtractor.normalize_harness_imports(source, reqs)
  assert "import math" in code
  assert "import os.path" in code
  assert "class A: pass" in code
