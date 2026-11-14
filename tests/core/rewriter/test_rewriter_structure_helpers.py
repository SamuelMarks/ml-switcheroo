"""Test suite for rewriter structure helpers."""

import libcst as cst

from ml_switcheroo.core.rewriter.passes.structure_helpers import StructuralTransformerHelpersMixin


class DummyTransformer(StructuralTransformerHelpersMixin):
  """Test double `DummyTransformer` used in rewriter structure helpers tests."""

  def _create_dotted_name(self, name: str) -> cst.BaseExpression:
    """Execute `name.split` to support create dotted name in rewriter structure helpers tests.

    Args:
        name: Target symbol or register identifier.

    Returns:
        Resulting `cst.BaseExpression` from create dotted name.
    """
    parts = name.split(".")
    node: cst.BaseExpression = cst.Name(parts[0])
    for p in parts[1:]:
      node = cst.Attribute(
        value=node,
        attr=cst.Name(p),
      )
    return node


def test_structure_helpers() -> None:
  """Verify `helper._strip_argument_from_signature` emits `c: int` for structure helpers."""
  helper = DummyTransformer()
  func_def = cst.parse_module("def foo(self, a, b): pass").body[0]
  assert isinstance(func_def, cst.FunctionDef)

  def get_code(node: cst.SimpleStatementLine | cst.BaseCompoundStatement) -> str:
    """Execute `Module` to support get code in rewriter structure helpers tests.

    Args:
        node: CST or AST node under inspection or transformation.

    Returns:
        Resulting `str` from get code.
    """
    return cst.Module(body=[node]).code

  # test _strip_argument_from_signature
  f2 = helper._strip_argument_from_signature(func_def, "a")
  assert " a," not in get_code(f2)

  # test _inject_argument_to_signature
  f3 = helper._inject_argument_to_signature(f2, "c", "int")
  assert "c: int" in get_code(f3)

  # check if first arg isn't self
  func_def2 = cst.parse_module("def foo(x): pass").body[0]
  assert isinstance(func_def2, cst.FunctionDef)
  f4 = helper._inject_argument_to_signature(func_def2, "c", None)
  assert "c, x" in get_code(f4)

  # test empty arg list
  func_def_empty = cst.parse_module("def foo(): pass").body[0]
  assert isinstance(func_def_empty, cst.FunctionDef)
  f_empty = helper._inject_argument_to_signature(func_def_empty, "c", None)
  assert "c" in get_code(f_empty)

  # test _apply_preamble
  f5 = helper._apply_preamble(func_def, ["print('hi')"])
  assert "print('hi')" in get_code(f5)

  # test _apply_preamble with syntax error
  helper._apply_preamble(func_def, ["print('hi'"])

  # test _convert_to_indented_block
  func_def3 = cst.parse_module("def foo(): return 1").body[0]
  assert isinstance(func_def3, cst.FunctionDef)
  f6 = helper._convert_to_indented_block(func_def3)
  assert isinstance(f6.body, cst.IndentedBlock)

  func_def4 = cst.parse_module("def foo():\n    return 1").body[0]
  assert isinstance(func_def4, cst.FunctionDef)
  helper._convert_to_indented_block(func_def4)

  # test _ensure_super_init
  f7 = helper._ensure_super_init(func_def)
  assert "super().__init__()" in get_code(f7)
  helper._ensure_super_init(f7)  # Shouldn't duplicate

  func_doc = cst.parse_module('def foo():\n    """Doc"""\n    pass').body[0]
  assert isinstance(func_doc, cst.FunctionDef)
  f8 = helper._ensure_super_init(func_doc)
  assert "super().__init__()" in get_code(f8)

  # test _strip_super_init
  f9 = helper._strip_super_init(f7)
  assert "super().__init__()" not in get_code(f9)

  helper._strip_super_init(func_def3)  # simple statement suite

  from unittest.mock import MagicMock

  mock_suite = MagicMock(spec=cst.BaseSuite)
  mock_f = MagicMock(spec=cst.FunctionDef)
  mock_f.body = mock_suite
  assert helper._strip_super_init(mock_f) is mock_f

  # test _update_docstring
  f10 = helper._update_docstring(func_doc, [("rngs", None)])
  assert "rngs: Injected." in get_code(f10)

  func_no_doc = cst.parse_module("def foo():\n    pass").body[0]
  assert isinstance(func_no_doc, cst.FunctionDef)
  helper._update_docstring(func_no_doc, [("rngs", None)])

  func_simple_doc = cst.parse_module('def foo(): """Doc"""').body[0]
  assert isinstance(func_simple_doc, cst.FunctionDef)
  helper._update_docstring(func_simple_doc, [("rngs", None)])

  f_invalid_doc = cst.parse_module("def foo():\n    1\n    pass").body[0]
  assert isinstance(f_invalid_doc, cst.FunctionDef)
  helper._update_docstring(f_invalid_doc, [("rngs", None)])

  assert helper._update_docstring(mock_f, [("rngs", None)]) is mock_f

  # _is_super_init_call edge cases
  stmt1 = cst.parse_module("a = 1").body[0]
  assert isinstance(stmt1, cst.SimpleStatementLine)
  assert not helper._is_super_init_call(stmt1)

  stmt2 = cst.parse_module("super().other()").body[0]
  assert isinstance(stmt2, cst.SimpleStatementLine)
  assert not helper._is_super_init_call(stmt2)

  stmt3 = cst.parse_module("obj.__init__()").body[0]
  assert isinstance(stmt3, cst.SimpleStatementLine)
  assert not helper._is_super_init_call(stmt3)

  # Not simple statement line
  stmt_complex = cst.parse_module("if True: pass").body[0]
  assert isinstance(stmt_complex, cst.If)
  assert not helper._is_super_init_call(stmt_complex)

  # _strip_argument_from_signature stripping the sole param (params becomes empty, len(params) > 0 is False)
  func_single_arg = cst.parse_module("def single(a): pass").body[0]
  assert isinstance(func_single_arg, cst.FunctionDef)
  f_stripped_all = helper._strip_argument_from_signature(func_single_arg, "a")
  assert len(f_stripped_all.params.params) == 0

  # _fix_comma with trailing comma on last param
  func_trailing = cst.parse_module("def trailing(a, b,): pass").body[0]
  assert isinstance(func_trailing, cst.FunctionDef)
  f_fixed_trailing = helper._fix_comma(func_trailing, list(func_trailing.params.params))
  assert f_fixed_trailing.params.params[-1].comma == cst.MaybeSentinel.DEFAULT

  # _inject_stmts_to_body with empty existing body
  func_empty_body = cst.ensure_type(
    cst.parse_module("def no_stmts():\n  pass").body[0],
    cst.FunctionDef,
  ).with_changes(body=cst.IndentedBlock(body=[]))
  f_prepended = helper._inject_stmts_to_body(func_empty_body, [cst.parse_statement("x = 1")])
  assert len(f_prepended.body.body) == 1

  # BaseSuite without IndentedBlock/SimpleStatementSuite for _has_super_init
  assert helper._has_super_init(mock_f) is False

  # _update_docstring with empty body
  assert helper._update_docstring(func_empty_body, [("arg", "int")]) is func_empty_body

  # _update_docstring without triple double quotes (e.g. triple single quotes)
  func_single_quote_doc = cst.parse_module("def foo():\n    '''SingleDoc'''\n    pass").body[0]
  assert isinstance(func_single_quote_doc, cst.FunctionDef)
  assert helper._update_docstring(func_single_quote_doc, [("arg", "int")]) is func_single_quote_doc

  # _strip_docstring_arg coverage
  func_with_injected_doc = helper._update_docstring(func_doc, [("rngs", None)])
  assert "rngs: Injected." in get_code(func_with_injected_doc)
  f_pruned = helper._strip_docstring_arg(func_with_injected_doc, "rngs")
  assert "rngs: Injected." not in get_code(f_pruned)

  # _strip_docstring_arg edge cases
  assert helper._strip_docstring_arg(func_empty_body, "rngs") is func_empty_body
  assert helper._strip_docstring_arg(func_def3, "rngs") is func_def3
  assert helper._strip_docstring_arg(f_invalid_doc, "rngs") is f_invalid_doc
  assert helper._strip_docstring_arg(func_doc, "not_present") is func_doc

  # _strip_docstring_arg edge case for exactly matching argument name
  func_doc_exact = cst.parse_module('def foo():\n    """Doc\n    rngs:\n    """\n    pass').body[0]
  assert isinstance(func_doc_exact, cst.FunctionDef)
  f_pruned_exact = helper._strip_docstring_arg(func_doc_exact, "rngs")
  assert "rngs:" not in get_code(f_pruned_exact)
