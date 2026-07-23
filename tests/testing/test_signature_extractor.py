"""Test suite for the Signature Extractor module."""

from ml_switcheroo.testing.signature_extractor import SignatureExtractor, FunctionDefVisitor
import libcst as cst


def test_extract_first_function_name_simple():
  """Test extracting a standard function."""
  code = "def my_func(a, b):\n    pass"
  name = SignatureExtractor.extract_first_function_name(code)
  assert name == "my_func"


def test_extract_first_function_name_multi_line():
  """Test extracting a function with multi-line signature."""
  code = """
def complex_function(
    arg1: int,
    arg2: str = "default"
) -> bool:
    return True
"""
  name = SignatureExtractor.extract_first_function_name(code)
  assert name == "complex_function"


def test_extract_first_function_name_decorator():
  """Test extracting a function with decorators and comments."""
  code = """
# This is a comment
@dataclass
@pytest.mark.skip
def decorated_func():
    pass
"""
  name = SignatureExtractor.extract_first_function_name(code)
  assert name == "decorated_func"


def test_extract_first_function_name_commented_out():
  """Test that commented out functions are ignored."""
  code = """
# def commented_func():
#     pass

def actual_func():
    pass
"""
  name = SignatureExtractor.extract_first_function_name(code)
  assert name == "actual_func"


def test_extract_first_function_name_syntax_error():
  """Test handling of invalid syntax."""
  code = "def invalid_syntax("
  name = SignatureExtractor.extract_first_function_name(code)
  assert name is None


def test_extract_first_function_name_no_function():
  """Test when no function exists."""
  code = "a = 1 + 2\nprint(a)"
  name = SignatureExtractor.extract_first_function_name(code)
  assert name is None


def test_visitor_returns_false_second_time():
  """Test visitor stops after first."""
  visitor = FunctionDefVisitor()
  visitor.function_name = "already_set"
  node = cst.parse_module("def new_func(): pass").body[0]
  assert visitor.visit_FunctionDef(node) is False
  assert visitor.function_name == "already_set"


def test_extract_first_function_name_another_syntax_error():
  """Test another syntax error case."""
  code = "this is not python"
  name = SignatureExtractor.extract_first_function_name(code)
  assert name is None
