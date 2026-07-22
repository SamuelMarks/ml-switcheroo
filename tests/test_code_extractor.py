"""Test suite for the Code Extractor module."""

import pytest
from ml_switcheroo.utils.code_extractor import CodeExtractor


class MockClass:
  """Mock Class class for testing purposes."""

  def hello(self):
    """Mock implementation of hello."""
    return "world"


def test_extract_simple_class():
  """Extracts simple class."""
  extracted = CodeExtractor.extract_class(MockClass)
  assert "class MockClass" in extracted
  assert "def hello(self):" in extracted
  assert 'return "world"' in extracted


def test_indentation_dedent():
  """Verifies the behavior of indentation dedent."""

  class Nested:
    """Test suite for the Nested component."""

    def inner(self):
      """Helper to inner."""
      pass

  extracted = CodeExtractor.extract_class(Nested)
  lines = extracted.splitlines()
  assert lines[0].startswith("class Nested")
  assert not lines[0].startswith("    class")


def test_not_a_class_raises_error():
  """Verifies the behavior of not a class raises correctly handling an error."""
  with pytest.raises(TypeError):
    CodeExtractor.extract_class(lambda x: x)
  inst = MockClass()
  with pytest.raises(TypeError):
    CodeExtractor.extract_class(inst)


def test_normalize_imports_injection():
  """Verifies the behavior of normalize imports injection."""
  source = "class Foo: pass"
  mods = ["numpy", "random"]
  res = CodeExtractor.normalize_harness_imports(source, mods)
  assert "import numpy" in res
  assert "import random" in res
  assert "class Foo: pass" in res


def test_real_fuzzer_extraction():
  """Verifies the behavior of real fuzzer extraction."""
  from ml_switcheroo.testing.fuzzer import InputFuzzer

  extracted = CodeExtractor.extract_class(InputFuzzer)
  assert "class InputFuzzer" in extracted
  assert "build_strategies" in extracted
