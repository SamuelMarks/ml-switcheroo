"""Test suite for the Signature Extractor module."""

import libcst as cst

from ml_switcheroo.testing.signature_extractor import FunctionDefVisitor, SignatureExtractor


def test_extract_first_function_name_simple() -> None:
  """Verify extract first function name simple behavior."""
  code: str = "def my_func(a, b):\n    pass"
  name: str | None = SignatureExtractor.extract_first_function_name(code)
  assert name == "my_func"


def test_extract_first_function_name_multi_line() -> None:
  """Verify extract first function name multi line behavior."""
  code: str = """
def complex_function(
    arg1: int,
    arg2: str = "default"
) -> bool:
    return True
"""
  name: str | None = SignatureExtractor.extract_first_function_name(code)
  assert name == "complex_function"


def test_extract_first_function_name_decorator() -> None:
  """Verify extract first function name decorator behavior."""
  code: str = """
# This is a comment
@dataclass
@pytest.mark.skip
def decorated_func():
    pass
"""
  name: str | None = SignatureExtractor.extract_first_function_name(code)
  assert name == "decorated_func"


def test_extract_first_function_name_commented_out() -> None:
  """Verify extract first function name commented out behavior."""
  code: str = """
# def commented_func():
#     pass

def actual_func():
    pass
"""
  name: str | None = SignatureExtractor.extract_first_function_name(code)
  assert name == "actual_func"


def test_extract_first_function_name_syntax_error() -> None:
  """Verify extract first function name syntax error behavior."""
  code: str = "def invalid_syntax("
  name: str | None = SignatureExtractor.extract_first_function_name(code)
  assert name is None


def test_extract_first_function_name_no_function() -> None:
  """Verify extract first function name no function behavior."""
  code: str = "a = 1 + 2\nprint(a)"
  name: str | None = SignatureExtractor.extract_first_function_name(code)
  assert name is None


def test_visitor_returns_false_second_time() -> None:
  """Verify visitor returns false second time behavior."""
  visitor: FunctionDefVisitor = FunctionDefVisitor()
  visitor.function_name = "already_set"
  node: cst.FunctionDef = getattr(cst.parse_module("def new_func(): pass"), "body")[0]
  assert visitor.visit_FunctionDef(node) is False
  assert visitor.function_name == "already_set"


def test_extract_first_function_name_another_syntax_error() -> None:
  """Verify extract first function name another syntax error behavior."""
  code: str = "this is not python"
  name: str | None = SignatureExtractor.extract_first_function_name(code)
  assert name is None
