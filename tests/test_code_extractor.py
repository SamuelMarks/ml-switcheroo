"""
Tests for the Code Extractor Utility.

Verifies that the extractor can retrieve source code from Python classes,
ensuring that the 'Split-Brain' fix is viable.
"""

import pytest
from ml_switcheroo.utils.code_extractor import CodeExtractor


# Mock Class for testing
class MockClass:
  """A sample class to extract."""

  def hello(self):
    return "world"


def test_extract_simple_class():
  """Verify basic class extraction."""
  extracted = CodeExtractor.extract_class(MockClass)
  assert "class MockClass" in extracted
  assert "def hello(self):" in extracted
  assert 'return "world"' in extracted


def test_indentation_dedent():
  """
  Verify that nested classes are extracted with proper dedent.
  """

  # Nested definition
  class Nested:
    def inner(self):
      pass

  extracted = CodeExtractor.extract_class(Nested)

  # The raw source lines usually contain the indentation from this test function.
  # We expect dedent to remove it.
  lines = extracted.splitlines()
  assert lines[0].startswith("class Nested")
  assert not lines[0].startswith("    class")


def test_not_a_class_raises_error():
  """Verify TypeError if passing a function or instance."""
  with pytest.raises(TypeError):
    CodeExtractor.extract_class(lambda x: x)

  inst = MockClass()
  with pytest.raises(TypeError):
    CodeExtractor.extract_class(inst)


def test_normalize_imports_injection():
  """Verify that imports are prepended correctly."""
  source = "class Foo: pass"
  mods = ["numpy", "random"]

  res = CodeExtractor.normalize_harness_imports(source, mods)
  assert "import numpy" in res
  assert "import random" in res
  assert "class Foo: pass" in res


def test_real_fuzzer_extraction():
  """
  Integration: Ensure we can extract the real InputFuzzer class.
  This is the primary use-case for the harness generator.
  """
  from ml_switcheroo.testing.fuzzer import InputFuzzer

  extracted = CodeExtractor.extract_class(InputFuzzer)
  assert "class InputFuzzer" in extracted
  assert "build_strategies" in extracted
