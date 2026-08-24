"""Test module."""

from ml_switcheroo.testing.signature_extractor import SignatureExtractor


def test_extract_first_function_name():
  """Test element."""
  assert SignatureExtractor.extract_first_function_name("def foo(x): pass") == "foo"


def test_extract_first_function_name_exception():
  """Test element."""
  assert SignatureExtractor.extract_first_function_name("def foo(x:") is None


def test_extract_first_function_name_multiple():
  """Test element."""
  code = "def foo(): pass\ndef bar(): pass"
  assert SignatureExtractor.extract_first_function_name(code) == "foo"


def test_extract_first_function_name_none():
  """Test element."""
  assert SignatureExtractor.extract_first_function_name("x = 1") is None
