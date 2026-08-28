"""Test suite for the Signature Extractor module."""

from ml_switcheroo.testing.signature_extractor import SignatureExtractor, FunctionDefVisitor
import libcst as cst
from typing import Optional


def test_extract_first_function_name_simple() -> None:
  """Test extracting a standard function."""
  code: str = "def my_func(a, b):\n    pass"
  name: Optional[str] = SignatureExtractor.extract_first_function_name(code)
  assert name == "my_func"


def test_extract_first_function_name_multi_line() -> None:
  """Test extracting a function with multi-line signature."""
  code: str = """
def complex_function(
    arg1: int,
    arg2: str = "default"
) -> bool:
    return True
"""
  name: Optional[str] = SignatureExtractor.extract_first_function_name(code)
  assert name == "complex_function"


def test_extract_first_function_name_decorator() -> None:
  """Test extracting a function with decorators and comments."""
  code: str = """
# This is a comment
@dataclass
@pytest.mark.skip
def decorated_func():
    pass
"""
  name: Optional[str] = SignatureExtractor.extract_first_function_name(code)
  assert name == "decorated_func"


def test_extract_first_function_name_commented_out() -> None:
  """Test that commented out functions are ignored."""
  code: str = """
# def commented_func():
#     pass

def actual_func():
    pass
"""
  name: Optional[str] = SignatureExtractor.extract_first_function_name(code)
  assert name == "actual_func"


def test_extract_first_function_name_syntax_error() -> None:
  """Test handling of invalid syntax."""
  code: str = "def invalid_syntax("
  name: Optional[str] = SignatureExtractor.extract_first_function_name(code)
  assert name is None


def test_extract_first_function_name_no_function() -> None:
  """Test when no function exists."""
  code: str = "a = 1 + 2\nprint(a)"
  name: Optional[str] = SignatureExtractor.extract_first_function_name(code)
  assert name is None


def test_visitor_returns_false_second_time() -> None:
  """Test visitor stops after first."""
  visitor: FunctionDefVisitor = FunctionDefVisitor()
  visitor.function_name = "already_set"
  node: cst.FunctionDef = getattr(cst.parse_module("def new_func(): pass"), "body")[0]
  assert visitor.visit_FunctionDef(node) is False
  assert visitor.function_name == "already_set"


def test_extract_first_function_name_another_syntax_error() -> None:
  """Test another syntax error case."""
  code: str = "this is not python"
  name: Optional[str] = SignatureExtractor.extract_first_function_name(code)
  assert name is None
