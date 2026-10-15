"""Test suite for code extractor."""

from typing import List
from unittest.mock import patch

import pytest

from ml_switcheroo.utils.code_extractor import CodeExtractor


class DummyFuzzer:
  """Test helper class DummyFuzzer."""

  def build_strategies(self) -> None:
    """Execute build strategies helper logic."""
    pass


def test_extract_class() -> None:
  """Verify extract class behavior."""
  with patch("inspect.getsource", return_value="class DummyFuzzer:\n    def build_strategies(self):\n        pass\n"):
    extracted: str = CodeExtractor.extract_class(DummyFuzzer)
    assert "class DummyFuzzer:" in extracted
    assert "build_strategies" in extracted


def test_extract_class_type_error() -> None:
  """Verify extract class type error behavior."""
  with pytest.raises(TypeError):
    CodeExtractor.extract_class("not a class")  # type: ignore


def test_extract_class_os_error() -> None:
  """Verify extract class os error behavior."""
  with patch("inspect.getsource", side_effect=OSError("err")):
    with pytest.raises(OSError):
      CodeExtractor.extract_class(DummyFuzzer)


def test_normalize_harness_imports() -> None:
  """Verify normalize harness imports behavior."""
  source: str = "class A: pass"
  reqs: List[str] = ["math", "os.path"]
  code: str = CodeExtractor.normalize_harness_imports(source, reqs)
  assert "import math" in code
  assert "import os.path" in code
  assert "class A: pass" in code
