"""Unit tests for `SignatureExtractor` and `FunctionDefVisitor` (`ml_switcheroo.testing.signature_extractor`)."""

import libcst as cst

from ml_switcheroo.testing.signature_extractor import FunctionDefVisitor, SignatureExtractor

_NL = chr(10)


def test_signature_extractor_first_function_and_edge_cases() -> None:
  """Verify `SignatureExtractor.extract_first_function_name` across multiline, decorated, commented, and invalid code."""
  assert SignatureExtractor.extract_first_function_name("def foo(x): pass") == "foo"
  assert SignatureExtractor.extract_first_function_name(f"def foo(): pass{_NL}def bar(): pass") == "foo"

  multiline = """
# def commented_func():
#     pass

@dataclass
@pytest.mark.skip
def complex_function(
    arg1: int,
    arg2: str = "default"
) -> bool:
    return True
"""
  assert SignatureExtractor.extract_first_function_name(multiline) == "complex_function"
  assert SignatureExtractor.extract_first_function_name(f"a = 1 + 2{_NL}print(a)") is None
  assert SignatureExtractor.extract_first_function_name("def invalid_syntax(") is None
  assert SignatureExtractor.extract_first_function_name("this is not python") is None

  visitor = FunctionDefVisitor()
  visitor.function_name = "already_set"
  fn_node = cst.ensure_type(cst.parse_module("def new_func(): pass").body[0], cst.FunctionDef)
  assert visitor.visit_FunctionDef(fn_node) is False
  assert visitor.function_name == "already_set"
