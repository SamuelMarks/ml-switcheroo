"""Test suite for signature extractor."""

from ml_switcheroo.testing.signature_extractor import SignatureExtractor


def test_extract_first_function_name() -> None:
  """Verify extract first function name behavior."""
  assert SignatureExtractor.extract_first_function_name("def foo(x): pass") == "foo"


def test_extract_first_function_name_exception() -> None:
  """Verify extract first function name exception behavior."""
  assert SignatureExtractor.extract_first_function_name("def foo(x:") is None


def test_extract_first_function_name_multiple() -> None:
  """Verify extract first function name multiple behavior."""
  code: str = "def foo(): pass\ndef bar(): pass"
  assert SignatureExtractor.extract_first_function_name(code) == "foo"


def test_extract_first_function_name_none() -> None:
  """Verify extract first function name none behavior."""
  assert SignatureExtractor.extract_first_function_name("x = 1") is None
