"""Test module."""

import libcst as cst
from ml_switcheroo.core.rewriter.passes.structure_helpers import StructuralTransformerHelpersMixin


class DummyTransformer(StructuralTransformerHelpersMixin):
  """Test element."""

  def _create_dotted_name(self, name):
    parts = name.split(".")
    node = cst.Name(parts[0])
    for p in parts[1:]:
      node = cst.Attribute(value=node, attr=cst.Name(p))
    return node


def test_structure_helpers():
  """Test element."""
  helper = DummyTransformer()
  func_def = cst.parse_module("def foo(self, a, b): pass").body[0]

  def get_code(node):
    return cst.Module(body=[node]).code

  # test _strip_argument_from_signature
  f2 = helper._strip_argument_from_signature(func_def, "a")
  assert " a," not in get_code(f2)

  # test _inject_argument_to_signature
  f3 = helper._inject_argument_to_signature(f2, "c", "int")
  assert "c: int" in get_code(f3)

  # check if first arg isn't self
  func_def2 = cst.parse_module("def foo(x): pass").body[0]
  f4 = helper._inject_argument_to_signature(func_def2, "c", None)
  assert "c, x" in get_code(f4)

  # test empty arg list
  func_def_empty = cst.parse_module("def foo(): pass").body[0]
  f_empty = helper._inject_argument_to_signature(func_def_empty, "c", None)
  assert "c" in get_code(f_empty)

  # test _apply_preamble
  f5 = helper._apply_preamble(func_def, ["print('hi')"])
  assert "print('hi')" in get_code(f5)

  # test _apply_preamble with syntax error
  helper._apply_preamble(func_def, ["print('hi'"])

  # test _convert_to_indented_block
  func_def3 = cst.parse_module("def foo(): return 1").body[0]
  f6 = helper._convert_to_indented_block(func_def3)
  assert isinstance(f6.body, cst.IndentedBlock)

  func_def4 = cst.parse_module("def foo():\n    return 1").body[0]
  helper._convert_to_indented_block(func_def4)

  # test _ensure_super_init
  f7 = helper._ensure_super_init(func_def)
  assert "super().__init__()" in get_code(f7)
  helper._ensure_super_init(f7)  # Shouldn't duplicate

  func_doc = cst.parse_module('def foo():\n    """Doc"""\n    pass').body[0]
  f8 = helper._ensure_super_init(func_doc)
  assert "super().__init__()" in get_code(f8)

  # test _strip_super_init
  f9 = helper._strip_super_init(f7)
  assert "super().__init__()" not in get_code(f9)

  helper._strip_super_init(func_def3)  # simple statement suite

  func_no_body = cst.parse_module("class A: pass").body[0]  # dummy node

  class A:
    pass

  A()
  helper._strip_super_init(func_no_body)

  # test _update_docstring
  f10 = helper._update_docstring(func_doc, [("rngs", None)])
  assert "rngs: Injected." in get_code(f10)

  func_no_doc = cst.parse_module("def foo():\n    pass").body[0]
  helper._update_docstring(func_no_doc, [("rngs", None)])

  func_simple_doc = cst.parse_module('def foo(): """Doc"""').body[0]
  helper._update_docstring(func_simple_doc, [("rngs", None)])

  f_invalid_doc = cst.parse_module("def foo():\n    1\n    pass").body[0]
  helper._update_docstring(f_invalid_doc, [("rngs", None)])

  helper._update_docstring(func_no_body, [("rngs", None)])

  # _is_super_init_call edge cases
  stmt1 = cst.parse_module("a = 1").body[0]
  assert not helper._is_super_init_call(stmt1)

  stmt2 = cst.parse_module("super().other()").body[0]
  assert not helper._is_super_init_call(stmt2)

  stmt3 = cst.parse_module("obj.__init__()").body[0]
  assert not helper._is_super_init_call(stmt3)

  # Not simple statement line
  stmt_complex = cst.parse_module("if True: pass").body[0]
  assert not helper._is_super_init_call(stmt_complex)
