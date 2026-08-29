"""Docstring."""

from typing import List
from unittest.mock import patch

import pytest

from ml_switcheroo.utils.code_extractor import CodeExtractor


class DummyFuzzer:
  """Docstring."""

  def build_strategies(self) -> None:
    """Docstring."""
    pass


def test_extract_class() -> None:
  """Docstring."""
  with patch("inspect.getsource", return_value="class DummyFuzzer:\n    def build_strategies(self):\n        pass\n"):
    extracted: str = CodeExtractor.extract_class(DummyFuzzer)
    assert "class DummyFuzzer:" in extracted
    assert "build_strategies" in extracted


def test_extract_class_type_error() -> None:
  """Docstring."""
  with pytest.raises(TypeError):
    CodeExtractor.extract_class("not a class")  # type: ignore


def test_extract_class_os_error() -> None:
  """Docstring."""
  with patch("inspect.getsource", side_effect=OSError("err")):
    with pytest.raises(OSError):
      CodeExtractor.extract_class(DummyFuzzer)


def test_normalize_harness_imports() -> None:
  """Docstring."""
  source: str = "class A: pass"
  reqs: List[str] = ["math", "os.path"]
  code: str = CodeExtractor.normalize_harness_imports(source, reqs)
  assert "import math" in code
  assert "import os.path" in code
  assert "class A: pass" in code
