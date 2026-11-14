"""Test suite for code extractor."""

from unittest.mock import patch

import pytest

from ml_switcheroo.utils.code_extractor import CodeExtractor


class DummyFuzzer:
  """Test double `DummyFuzzer` implementing `build_strategies` for code extractor tests."""

  def build_strategies(self) -> int:
    """Execute `build_strategies` on `DummyFuzzer` for code extractor tests.

    Returns:
        Integer strategy sentinel.
    """
    return 1


def test_extract_class() -> None:
  """Verify `CodeExtractor.extract_class` emits `class DummyFuzzer:` across code extractor inputs."""
  with patch("inspect.getsource", return_value="class DummyFuzzer:\n    def build_strategies(self):\n        pass\n"):
    extracted = CodeExtractor.extract_class(DummyFuzzer)
    assert "class DummyFuzzer:" in extracted
    assert "build_strategies" in extracted


def test_extract_class_type_error() -> None:
  """Verify `CodeExtractor.extract_class` raises `TypeError` during type error."""
  with pytest.raises(TypeError):
    CodeExtractor.extract_class("not a class")


def test_extract_class_os_error() -> None:
  """Verify `CodeExtractor.extract_class` raises `OSError` during os error."""
  with patch("inspect.getsource", side_effect=OSError("err")), pytest.raises(OSError, match="Could not get source"):
    CodeExtractor.extract_class(DummyFuzzer)


def test_normalize_harness_imports() -> None:
  """Verify `CodeExtractor.normalize_harness_imports` emits `import math` across code extractor inputs."""
  source = "class A: pass"
  reqs = ["math", "os.path"]
  code = CodeExtractor.normalize_harness_imports(source, reqs)
  assert "import math" in code
  assert "import os.path" in code
  assert "class A: pass" in code


def test_extract_function_body() -> None:
  """Verify extract_function_body extracts and indents method bodies via LibCST."""
  body = CodeExtractor.extract_function_body(DummyFuzzer.build_strategies, indent=4)
  assert "return 1" in body

  with patch("inspect.getsource", return_value="x = 1\n"), pytest.raises(ValueError, match="FunctionDef"):
    CodeExtractor.extract_function_body(DummyFuzzer.build_strategies)
