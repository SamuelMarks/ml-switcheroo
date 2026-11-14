"""Generator backend for creating physical Python test files.

It orchestrates the generation of PyTest-compatible files that verify
operations across multiple frameworks by using the semantic definitions.
"""

import ast
import pathlib
import typing

import libcst as cst
from pydantic import JsonValue

from ml_switcheroo.enums import RuntimeTensorOrValue
from ml_switcheroo.generated_tests.inputs import generate_input_value_code, parse_arg_def
from ml_switcheroo.generated_tests.runtime_builder import ensure_runtime_module
from ml_switcheroo.generated_tests.templates import get_template, is_static_arg

if typing.TYPE_CHECKING:
  import ml_switcheroo.semantics.manager

StmtNode: typing.TypeAlias = cst.SimpleStatementLine | cst.BaseCompoundStatement
SemanticsInputMap: typing.TypeAlias = typing.Mapping[
  str,
  typing.Mapping[str, RuntimeTensorOrValue] | JsonValue | RuntimeTensorOrValue,
]

_NL = chr(10)

_TYPE_CHECK_ASSERTIONS: dict[str, tuple[str, str]] = {
  "int": (
    "# Expected int",
    f"for fw, val in results.items():{_NL}    assert np.issubdtype(np.array(val).dtype, np.integer) or isinstance(val, int)",
  ),
  "bool": (
    "# Expected bool",
    f"for fw, val in results.items():{_NL}    assert np.issubdtype(np.array(val).dtype, bool) or isinstance(val, bool)",
  ),
  "Tensor": (
    "# Expected Array/Tensor",
    f"for fw, val in results.items():{_NL}    assert isinstance(val, (np.ndarray, np.generic))",
  ),
  "Array": (
    "# Expected Array/Tensor",
    f"for fw, val in results.items():{_NL}    assert isinstance(val, (np.ndarray, np.generic))",
  ),
}

_GRAD_TEMPLATES: dict[str, tuple[str, bool]] = {
  "jax": ("jax.grad(lambda {lam_sig}: jnp.sum({lam_call_body}))({input_0})", False),
  "torch": ("torch.func.grad(lambda {lam_sig}: torch.sum({lam_call_body}))({input_0})", True),
}


def _with_comments(stmt: StmtNode, *comments: str) -> StmtNode:
  """Attach leading comment lines onto ``stmt``.

  Args:
      stmt: Target LibCST statement node.
      *comments: Ordered comment lines (each beginning with ``'#'``).

  Returns:
      Updated statement node carrying ``leading_lines`` comments.
  """
  lines = list(stmt.leading_lines) + [cst.EmptyLine(comment=cst.Comment(c)) for c in comments]
  return stmt.with_changes(leading_lines=lines)


def _discover_existing_tests(out_file: pathlib.Path) -> set[str]:
  """Collect existing ``test_gen_*`` function names from ``out_file`` if present.

  Args:
      out_file: Candidate existing pytest file path.

  Returns:
      Set of existing generated test function identifiers.
  """
  existing_tests: set[str] = set()
  if not out_file.exists():
    return existing_tests
  original_content = out_file.read_text(encoding="utf-8")
  try:
    tree = ast.parse(original_content)
  except SyntaxError:
    return existing_tests
  for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name.startswith("test_gen_"):
      existing_tests.add(node.name)
  return existing_tests


def _collect_all_variants(semantics: SemanticsInputMap) -> list[str]:
  """Extract all referenced framework variant keys across ``semantics``.

  Args:
      semantics: Mapping of operation names to semantic definitions.

  Returns:
      Deduplicated list of framework identifier strings.
  """
  all_variants: set[str] = set()
  for op in semantics.values():
    if isinstance(op, typing.Mapping):
      raw_vars = op.get("variants", {})
      if isinstance(raw_vars, typing.Mapping):
        all_variants.update(str(k) for k in raw_vars)
  return list(all_variants)


def _build_input_section(
  args: list[dict[str, JsonValue]],
  *,
  is_void: bool,
) -> tuple[list[StmtNode], list[str], bool]:
  """Synthesize input initialization and result accumulator statements.

  Args:
      args: Normalized argument specifications for the operation.
      is_void: Whether the operation returns ``None``/``void``.

  Returns:
      Tuple of ``(statements, input_var_names, first_input_is_differentiable)``.
  """
  stmts: list[StmtNode] = []
  input_names: list[str] = []
  input_is_diff = False

  for idx, arg in enumerate(args):
    var_name = f"np_{arg['name']}"
    val_code = generate_input_value_code(str(arg["name"]), arg)
    stmts.append(cst.parse_statement(f"{var_name} = {val_code}"))
    input_names.append(var_name)
    if idx == 0 and arg["type"] in ("Array", "Tensor", "float"):
      input_is_diff = True

  stmts.append(cst.parse_statement("results = {}"))
  if not is_void:
    stmts.append(cst.parse_statement("results_grad = {}"))

  stmts[0] = _with_comments(stmts[0], "# 1. Inputs")
  return stmts, input_names, input_is_diff


def _build_grad_check_stmts(
  fw: str,
  api_to_call: str,
  call_args: list[str],
  to_numpy: str,
) -> list[StmtNode]:
  """Synthesize per-framework input gradient capture statements when supported.

  Args:
      fw: Framework identifier (e.g. ``'jax'``, ``'torch'``).
      api_to_call: Callable expression string invoked inside the gradient lambda.
      call_args: Converted framework input argument expressions.
      to_numpy: Format template converting framework tensors back to NumPy.

  Returns:
      Statements computing and storing ``results_grad[fw]``, or empty list if unsupported.
  """
  if fw not in _GRAD_TEMPLATES:
    return []
  lam_args = [f"a{i}" for i in range(len(call_args))]
  lam_sig = ", ".join(lam_args)
  lam_call_body = f"{api_to_call}({', '.join(lam_args)})"
  grad_tmpl, needs_to_numpy = _GRAD_TEMPLATES[fw]
  grad_line = grad_tmpl.format(lam_sig=lam_sig, lam_call_body=lam_call_body, input_0=call_args[0])
  grad_stmt = _with_comments(cst.parse_statement(f"grad_res = {grad_line}"), "# Gradient check")
  store_expr = to_numpy.format(res_var="grad_res") if needs_to_numpy else "grad_res"
  return [grad_stmt, cst.parse_statement(f"results_grad['{fw}'] = {store_expr}")]


def _build_comparison_section(
  *,
  return_type: str,
  is_void: bool,
  test_rtol: JsonValue | RuntimeTensorOrValue,
  test_atol: JsonValue | RuntimeTensorOrValue,
  exact_kw: str,
  check_grads: bool,
) -> list[StmtNode]:
  """Synthesize return-type, value-comparison, and gradient-comparison assertions.

  Args:
      return_type: Expected return type string from the operation spec.
      is_void: Whether the operation has void/None return semantics.
      test_rtol: Relative comparison tolerance.
      test_atol: Absolute comparison tolerance.
      exact_kw: Formatted ``exact=True`` or ``exact=False`` keyword argument.
      check_grads: Whether cross-framework gradient assertions should be emitted.

  Returns:
      List of assertion statements appended to the generated test body.
  """
  if is_void:
    return [
      _with_comments(
        cst.parse_statement("assert len(results) >= 2"),
        "# Comparison",
        "# Operation expected to return None / Void",
      )
    ]

  stmts: list[StmtNode] = []
  type_entry = _TYPE_CHECK_ASSERTIONS.get(return_type)
  if type_entry is not None:
    expected_comment, check_code = type_entry
    stmts.append(_with_comments(cst.parse_statement(check_code), "# Type Check", expected_comment))

  cmp_body: list[StmtNode] = [
    cst.parse_statement("vals = list(results.values())"),
    cst.parse_statement("ref = vals[0]"),
    cst.parse_statement(
      f"for val in vals[1:]:{_NL}    assert verify_results(ref, val, rtol={test_rtol}, atol={test_atol}, {exact_kw})"
    ),
  ]
  cmp_if = cst.If(
    test=cst.Comparison(
      left=cst.Call(func=cst.Name("len"), args=[cst.Arg(cst.Name("results"))]),
      comparisons=[cst.ComparisonTarget(operator=cst.GreaterThanEqual(), comparator=cst.Integer("2"))],
    ),
    body=cst.IndentedBlock(body=cmp_body),
  )
  stmts.append(_with_comments(cmp_if, "# Comparison"))

  if check_grads:
    grad_cmp: list[StmtNode] = [
      cst.parse_statement("g_vals = list(results_grad.values())"),
      cst.parse_statement("g_ref = g_vals[0]"),
      cst.parse_statement(
        f"for g_val in g_vals[1:]:{_NL}    np.testing.assert_allclose(g_ref, g_val, rtol=0.01, atol=0.001)"
      ),
    ]
    grad_if = cst.If(
      test=cst.Comparison(
        left=cst.Call(func=cst.Name("len"), args=[cst.Arg(cst.Name("results_grad"))]),
        comparisons=[cst.ComparisonTarget(operator=cst.GreaterThanEqual(), comparator=cst.Integer("2"))],
      ),
      body=cst.IndentedBlock(body=grad_cmp),
    )
    stmts.append(_with_comments(grad_if, "# Gradient Verification", "# Grad Check"))

  return stmts


class TestCaseGenerator:
  """Generate PyTest files for ML operators across frameworks (Torch, JAX, etc.).

  Handles argument constraints, type checking, gradient verification, and
  runtime environment setup via helper modules.
  """

  # Prevent pytest from trying to collect this class as a test suite
  __test__ = False

  def __init__(self, semantics_mgr: "ml_switcheroo.semantics.manager.SemanticsManager | None" = None) -> None:
    """Initialize the TestCaseGenerator.

    Args:
        semantics_mgr: Manager for semantics and templates.

    Returns:
        None
    """
    self.semantics_mgr = semantics_mgr

  def _ensure_runtime_module(self, out_dir: pathlib.Path, frameworks: list[str] | None = None) -> None:
    """Proxy request to runtime_builder.

    Args:
        out_dir: The directory where the runtime module should be created.
        frameworks: The collection or list of frameworks to configure.

    Returns:
        None
    """
    ensure_runtime_module(out_dir, frameworks, self.semantics_mgr)

  def _build_framework_block(
    self,
    fw: str,
    api: str,
    args: list[dict[str, JsonValue]],
    input_names: list[str],
    *,
    is_void: bool,
    check_grads: bool,
  ) -> StmtNode:
    """Construct the ``if <FW>_AVAILABLE:`` execution block for a single framework variant.

    Args:
        fw: Framework key string.
        api: Target framework callable path string.
        args: Normalized standard argument definitions.
        input_names: Generated NumPy input variable names.
        is_void: Whether the operation has void/None return semantics.
        check_grads: Whether gradient capture statements should be emitted.

    Returns:
        Guarded ``cst.If`` statement executing and recording the variant output.
    """
    tmpl = get_template(self.semantics_mgr, fw)
    conv_tmpl = tmpl.get("convert_input", "{np_var}")
    call_args = [conv_tmpl.format(np_var=aname) for aname in input_names]

    try_body: list[StmtNode] = []
    jit_tmpl = tmpl.get("jit_template")
    if jit_tmpl:
      try_body.append(cst.parse_statement(f"fn = {api}"))
      api_to_call = "fn"
    else:
      api_to_call = api

    fn_call_expr = f"{api_to_call}({', '.join(call_args)})"
    if jit_tmpl and "{fn}" in jit_tmpl:
      static_indices = [idx for idx, arg in enumerate(args) if is_static_arg(arg)]
      static_argnums = f"{tuple(static_indices)}" if static_indices else "None"
      jit_wrapper = jit_tmpl.format(fn=api_to_call, static_argnums=static_argnums)
      fn_call_expr = f"{jit_wrapper}({', '.join(call_args)})"

    try_body.append(cst.parse_statement(f"res = {fn_call_expr}"))
    to_numpy = tmpl.get("to_numpy", "{res_var}")
    res_conv = to_numpy.format(res_var="res")
    res_line = f"results['{fw}'] = {res_conv}" if not is_void else f"results['{fw}'] = res"
    try_body.append(cst.parse_statement(res_line))

    if check_grads:
      try_body.extend(_build_grad_check_stmts(fw, api_to_call, call_args, to_numpy))

    handler = cst.ExceptHandler(
      type=cst.Name("Exception"),
      name=cst.AsName(name=cst.Name("e")),
      body=cst.IndentedBlock(body=[cst.parse_statement(f"print(f'Skipping {fw} due to error: {{e}}')")]),
    )
    try_stmt = cst.Try(body=cst.IndentedBlock(body=try_body), handlers=[handler])
    if_stmt = cst.If(test=cst.Name(f"{fw.upper()}_AVAILABLE"), body=cst.IndentedBlock(body=[try_stmt]))
    return _with_comments(if_stmt, f"# Framework: {fw}")

  def _generate_op_test_function(
    self,
    op_name: str,
    op_def: typing.Mapping[str, JsonValue | RuntimeTensorOrValue],
    existing_tests: set[str],
  ) -> cst.FunctionDef | None:
    """Build a ``test_gen_<op_name>`` function definition if >= 2 valid variants exist.

    Args:
        op_name: Canonical operation identifier.
        op_def: Semantic definition mapping for the operation.
        existing_tests: Set of manual/pre-existing test function names to skip.

    Returns:
        Constructed ``cst.FunctionDef`` node, or ``None`` if skipped.
    """
    raw_variants = op_def.get("variants", {})
    variants = raw_variants if isinstance(raw_variants, typing.Mapping) else {}
    valid_variants: dict[str, typing.Mapping[str, JsonValue | RuntimeTensorOrValue]] = {
      str(k): v for k, v in variants.items() if isinstance(v, typing.Mapping) and get_template(self.semantics_mgr, str(k))
    }

    test_func_name = f"test_gen_{op_name}"
    if len(valid_variants) < 2 or test_func_name in existing_tests:
      return None

    std_args_raw = op_def.get("std_args", [])
    args = [parse_arg_def(a) for a in std_args_raw] if isinstance(std_args_raw, (list, tuple)) else []

    return_type = str(op_def.get("return_type", "Tensor"))
    is_void = return_type in ("None", "void")
    differentiable = bool(op_def.get("differentiable", True))
    test_rtol = op_def.get("test_rtol", 0.001)
    test_atol = op_def.get("test_atol", 0.0001)
    ver_mode = str(op_def.get("verification_mode", "fuzzy"))
    exact_kw = "exact=True" if ver_mode == "exact" else "exact=False"

    func_body, input_names, input_is_diff = _build_input_section(args, is_void=is_void)
    check_grads = not is_void and differentiable and input_is_diff

    for fw in sorted(valid_variants.keys()):
      api = str(valid_variants[fw]["api"])
      func_body.append(
        self._build_framework_block(
          fw,
          api,
          args,
          input_names,
          is_void=is_void,
          check_grads=check_grads,
        )
      )

    func_body.extend(
      _build_comparison_section(
        return_type=return_type,
        is_void=is_void,
        test_rtol=test_rtol,
        test_atol=test_atol,
        exact_kw=exact_kw,
        check_grads=check_grads,
      )
    )

    return cst.FunctionDef(
      name=cst.Name(test_func_name),
      params=cst.Parameters(),
      body=cst.IndentedBlock(body=func_body),
      leading_lines=[cst.EmptyLine()],
    )

  def generate(
    self,
    semantics: SemanticsInputMap,
    out_file: pathlib.Path,
  ) -> None:
    """Generate cross-framework physical equivalence pytest functions for all operations in ``semantics``.

    Args:
        semantics: Dictionary mapping operation names to their semantic definition dictionaries.
        out_file: Output file path where the generated pytest module will be written.
    """
    if not semantics:
      return

    self._ensure_runtime_module(out_file.parent, _collect_all_variants(semantics))
    existing_tests = _discover_existing_tests(out_file)

    module_body: list[StmtNode] = [
      cst.SimpleStatementLine(body=[cst.Expr(cst.SimpleString('"""Generated tests."""'))]),
      cst.parse_statement("import pytest"),
      cst.parse_statement("import numpy as np"),
      cst.parse_statement("import numpy"),
      cst.parse_statement("import random"),
      cst.parse_statement("import math"),
      cst.parse_statement("from .runtime import *"),
    ]

    ops_generated = False
    for op_name, op_def in semantics.items():
      if not isinstance(op_def, typing.Mapping):
        continue
      fn_node = self._generate_op_test_function(op_name, op_def, existing_tests)
      if fn_node is not None:
        ops_generated = True
        module_body.append(fn_node)

    if not ops_generated and not existing_tests:
      out_file.write_text(
        f'"""Generated tests."""{_NL}# No tests generated due to insufficient variants.{_NL}',
        encoding="utf-8",
      )
    else:
      footer = [cst.EmptyLine()]
      mod = cst.Module(body=module_body, footer=footer)
      out_file.write_text(mod.code, encoding="utf-8")
