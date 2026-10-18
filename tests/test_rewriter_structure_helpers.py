"""Test suite for rewriter structure helpers."""

import typing


import libcst as cst

from ml_switcheroo.core.rewriter.passes.structure_helpers import StructuralTransformerHelpersMixin


class DummyTransformer(StructuralTransformerHelpersMixin):
  """Test helper class DummyTransformer."""

  def _create_dotted_name(self, name: typing.Any) -> typing.Any:
    """Execute create dotted name helper logic.

    Returns:
        Computed test result or fixture value.
    """
    parts: typing.Any = name.split(".")
    node: typing.Any = cst.Name(parts[0])
    for p in parts[1:]:
      node = cst.Attribute(value=node, attr=cst.Name(p))
    return node


def test_structure_helpers() -> None:
  """Verify structure helpers behavior."""
  helper: typing.Any = DummyTransformer()
  func_def: typing.Any = getattr(cst.parse_module("def foo(self, a, b): pass"), "body")[0]

  def get_code(node: typing.Any) -> typing.Any:
    """Execute get code helper logic.

    Args:
        node: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return cst.Module(body=[node]).code

  # test _strip_argument_from_signature
  f2: typing.Any = helper._strip_argument_from_signature(func_def, "a")
  assert " a," not in get_code(f2)

  # test _inject_argument_to_signature
  f3: typing.Any = helper._inject_argument_to_signature(f2, "c", "int")
  assert "c: int" in get_code(f3)

  # check if first arg isn't self
  func_def2: typing.Any = getattr(cst.parse_module("def foo(x): pass"), "body")[0]
  f4: typing.Any = helper._inject_argument_to_signature(func_def2, "c", None)
  assert "c, x" in get_code(f4)

  # test empty arg list
  func_def_empty: typing.Any = getattr(cst.parse_module("def foo(): pass"), "body")[0]
  f_empty: typing.Any = helper._inject_argument_to_signature(func_def_empty, "c", None)
  assert "c" in get_code(f_empty)

  # test _apply_preamble
  f5: typing.Any = helper._apply_preamble(func_def, ["print('hi')"])
  assert "print('hi')" in get_code(f5)

  # test _apply_preamble with syntax error
  helper._apply_preamble(func_def, ["print('hi'"])

  # test _convert_to_indented_block
  func_def3: typing.Any = getattr(cst.parse_module("def foo(): return 1"), "body")[0]
  f6: typing.Any = helper._convert_to_indented_block(func_def3)
  assert isinstance(f6.body, cst.IndentedBlock)

  func_def4: typing.Any = getattr(cst.parse_module("def foo():\n    return 1"), "body")[0]
  helper._convert_to_indented_block(func_def4)

  # test _ensure_super_init
  f7: typing.Any = helper._ensure_super_init(func_def)
  assert "super().__init__()" in get_code(f7)
  helper._ensure_super_init(f7)  # Shouldn't duplicate

  func_doc: typing.Any = getattr(cst.parse_module('def foo():\n    """Doc"""\n    pass'), "body")[0]
  f8: typing.Any = helper._ensure_super_init(func_doc)
  assert "super().__init__()" in get_code(f8)

  # test _strip_super_init
  f9: typing.Any = helper._strip_super_init(f7)
  assert "super().__init__()" not in get_code(f9)

  helper._strip_super_init(func_def3)  # simple statement suite

  func_no_body: typing.Any = getattr(cst.parse_module("class A: pass"), "body")[0]

  class A:
    """Test helper class A."""

    pass

  A()
  helper._strip_super_init(func_no_body)

  # test _update_docstring
  f10: typing.Any = helper._update_docstring(func_doc, [("rngs", None)])
  assert "rngs: Injected." in get_code(f10)

  func_no_doc: typing.Any = getattr(cst.parse_module("def foo():\n    pass"), "body")[0]
  helper._update_docstring(func_no_doc, [("rngs", None)])

  func_simple_doc: typing.Any = getattr(cst.parse_module('def foo(): """Doc"""'), "body")[0]
  helper._update_docstring(func_simple_doc, [("rngs", None)])

  f_invalid_doc: typing.Any = getattr(cst.parse_module("def foo():\n    1\n    pass"), "body")[0]
  helper._update_docstring(f_invalid_doc, [("rngs", None)])

  helper._update_docstring(func_no_body, [("rngs", None)])

  # _is_super_init_call edge cases
  stmt1: typing.Any = getattr(cst.parse_module("a = 1"), "body")[0]
  assert not helper._is_super_init_call(stmt1)

  stmt2: typing.Any = getattr(cst.parse_module("super().other()"), "body")[0]
  assert not helper._is_super_init_call(stmt2)

  stmt3: typing.Any = getattr(cst.parse_module("obj.__init__()"), "body")[0]
  assert not helper._is_super_init_call(stmt3)

  # Not simple statement line
  stmt_complex: typing.Any = getattr(cst.parse_module("if True: pass"), "body")[0]
  assert not helper._is_super_init_call(stmt_complex)

  # _strip_argument_from_signature stripping the sole param (params becomes empty, len(params) > 0 is False)
  func_single_arg: typing.Any = getattr(cst.parse_module("def single(a): pass"), "body")[0]
  f_stripped_all: typing.Any = helper._strip_argument_from_signature(func_single_arg, "a")
  assert len(f_stripped_all.params.params) == 0

  # _fix_comma with trailing comma on last param
  func_trailing: typing.Any = getattr(cst.parse_module("def trailing(a, b,): pass"), "body")[0]
  f_fixed_trailing: typing.Any = helper._fix_comma(func_trailing, list(func_trailing.params.params))
  assert f_fixed_trailing.params.params[-1].comma == cst.MaybeSentinel.DEFAULT

  # _inject_stmts_to_body with empty existing body
  func_empty_body: typing.Any = getattr(cst.parse_module("def no_stmts():\n  pass"), "body")[0].with_changes(
    body=cst.IndentedBlock(body=[])
  )
  f_prepended: typing.Any = helper._inject_stmts_to_body(func_empty_body, [cst.parse_statement("x = 1")])
  assert len(f_prepended.body.body) == 1

  # Mock body without body attribute for _strip_super_init and _has_super_init
  class _MockBody:
    """Mock body without body attribute."""

  class _MockFunc:
    """Mock function with MockBody."""

    def __init__(self) -> None:
      """Initialize mock function."""
      self.body = _MockBody()

  mock_f: typing.Any = _MockFunc()
  assert helper._strip_super_init(mock_f) is mock_f
  assert helper._has_super_init(mock_f) is False

  # _update_docstring with empty body
  assert helper._update_docstring(func_empty_body, [("arg", "int")]) is func_empty_body

  # _update_docstring without triple double quotes (e.g. triple single quotes)
  func_single_quote_doc: typing.Any = getattr(cst.parse_module("def foo():\n    '''SingleDoc'''\n    pass"), "body")[0]
  assert helper._update_docstring(func_single_quote_doc, [("arg", "int")]) is func_single_quote_doc

  # _strip_docstring_arg coverage
  func_with_injected_doc: typing.Any = helper._update_docstring(func_doc, [("rngs", None)])
  assert "rngs: Injected." in get_code(func_with_injected_doc)
  f_pruned: typing.Any = helper._strip_docstring_arg(func_with_injected_doc, "rngs")
  assert "rngs: Injected." not in get_code(f_pruned)

  # _strip_docstring_arg edge cases
  assert helper._strip_docstring_arg(func_empty_body, "rngs") is func_empty_body
  assert helper._strip_docstring_arg(func_def3, "rngs") is func_def3
  assert helper._strip_docstring_arg(f_invalid_doc, "rngs") is f_invalid_doc
  assert helper._strip_docstring_arg(func_doc, "not_present") is func_doc

  # _strip_docstring_arg edge case for exactly matching argument name
  func_doc_exact: typing.Any = getattr(cst.parse_module('def foo():\n    """Doc\n    rngs:\n    """\n    pass'), "body")[
    0
  ]
  f_pruned_exact: typing.Any = helper._strip_docstring_arg(func_doc_exact, "rngs")
  assert "rngs:" not in get_code(f_pruned_exact)
