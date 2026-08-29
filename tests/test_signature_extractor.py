"""Test module."""

from ml_switcheroo.testing.signature_extractor import SignatureExtractor


def test_extract_first_function_name() -> None:
  """Docstring."""
  assert SignatureExtractor.extract_first_function_name("def foo(x): pass") == "foo"


def test_extract_first_function_name_exception() -> None:
  """Docstring."""
  assert SignatureExtractor.extract_first_function_name("def foo(x:") is None


def test_extract_first_function_name_multiple() -> None:
  """Docstring."""
  code: str = "def foo(): pass\ndef bar(): pass"
  assert SignatureExtractor.extract_first_function_name(code) == "foo"


def test_extract_first_function_name_none() -> None:
  """Docstring."""
  assert SignatureExtractor.extract_first_function_name("x = 1") is None
