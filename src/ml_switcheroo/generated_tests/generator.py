# mypy: ignore-errors
"""Generator backend for creating physical Python test files.

It orchestrates the generation of PyTest-compatible files that verify
operations across multiple frameworks by using the semantic definitions.
"""

from typing import Any


import ml_switcheroo
import typing

if typing.TYPE_CHECKING:
  import ml_switcheroo.semantics.manager


import ast
import pathlib

from ml_switcheroo.generated_tests.templates import get_template, is_static_arg
from ml_switcheroo.generated_tests.inputs import parse_arg_def, generate_input_value_code
from ml_switcheroo.generated_tests.runtime_builder import ensure_runtime_module


class TestCaseGenerator:
  """Generate PyTest files for ML operators across frameworks (Torch, JAX, etc.).

  Handles argument constraints, type checking, gradient verification, and
  runtime environment setup via helper modules.
  """

  # Prevent pytest from trying to collect this class as a test suite
  __test__ = False

  def __init__(self, semantics_mgr: typing.Optional["ml_switcheroo.semantics.manager.SemanticsManager"] = None) -> None:
    """Initialize the TestCaseGenerator.

    Args:
        semantics_mgr: Manager for semantics and templates.

    Returns:
        None
    """
    self.semantics_mgr = semantics_mgr

  def _ensure_runtime_module(self, out_dir: pathlib.Path, frameworks: typing.Optional[typing.List[str]] = None) -> None:
    """Proxy request to runtime_builder.

    Args:
        out_dir: The directory where the runtime module should be created.
        frameworks: The collection or list of frameworks to configure.

    Returns:
        None
    """
    ensure_runtime_module(out_dir, frameworks, self.semantics_mgr)

  def generate(self, semantics: typing.Dict[str, dict[Any, Any]], out_file: pathlib.Path) -> None:
    """Generate cross-framework physical equivalence pytest functions for all operations in ``semantics``.

    Args:
        semantics: Dictionary mapping operation names to their semantic definition dictionaries.
        out_file: Output file path where the generated pytest module will be written.
    """
    import libcst as cst

    if not semantics:
      return

    # 1. Provide Determinism / Runtime
    all_variants = set()
    for op in semantics.values():
      all_variants.update(op.get("variants", {}).keys())
    self._ensure_runtime_module(out_file.parent, list(all_variants))

    # 2. Check for existing tests (to skip overwriting with generation)
    existing_tests = set()
    if out_file.exists():
      original_content = out_file.read_text(encoding="utf-8")
      try:
        tree = ast.parse(original_content)
        for node in tree.body:
          if isinstance(node, ast.FunctionDef) and node.name.startswith("test_gen_"):
            existing_tests.add(node.name)
      except SyntaxError:
        pass

    # 3. Accumulate lines for file
    module_body = [
      cst.SimpleStatementLine(body=[cst.Expr(cst.SimpleString('"""Generated tests."""'))]),
      cst.parse_statement("import pytest"),
      cst.parse_statement("import numpy as np"),
      cst.parse_statement("import numpy"),
      cst.parse_statement("import random"),
      cst.parse_statement("import math"),
      cst.parse_statement("from .runtime import *"),
      cst.EmptyLine(),
    ]

    ops_generated = False

    for op_name, op_def in semantics.items():
      variants = op_def.get("variants", {})
      valid_variants = {k: v for k, v in variants.items() if get_template(self.semantics_mgr, k)}

      test_func_name = f"test_gen_{op_name}"
      if len(valid_variants) < 2 or test_func_name in existing_tests:
        continue

      ops_generated = True

      std_args_raw = op_def.get("std_args", [])
      args = [parse_arg_def(a) for a in std_args_raw]

      return_type = op_def.get("return_type", "Tensor")
      is_void = return_type in ["None", "void"]
      differentiable = op_def.get("differentiable", True)
      test_rtol = op_def.get("test_rtol", 0.001)
      test_atol = op_def.get("test_atol", 0.0001)
      ver_mode = op_def.get("verification_mode", "fuzzy")
      exact_kw = "exact=True" if ver_mode == "exact" else "exact=False"

      func_body = []
      func_body.append(cst.EmptyLine(comment=cst.Comment("# 1. Inputs")))

      input_names = []
      input_is_diff = False

      for idx, arg in enumerate(args):
        var_name = f"np_{arg['name']}"
        val_code = generate_input_value_code(arg["name"], arg)
        func_body.append(cst.parse_statement(f"{var_name} = {val_code}"))
        input_names.append(var_name)

        if idx == 0 and arg["type"] in ["Array", "Tensor", "float"]:
          input_is_diff = True

      func_body.append(cst.parse_statement("results = {}"))
      if not is_void:
        func_body.append(cst.parse_statement("results_grad = {}"))

      for fw in sorted(valid_variants.keys()):
        fw_cfg = valid_variants[fw]
        api = fw_cfg["api"]
        tmpl = get_template(self.semantics_mgr, fw)

        func_body.append(cst.EmptyLine(comment=cst.Comment(f"# Framework: {fw}")))

        try_body = []
        call_args = []
        for aname in input_names:
          conv_tmpl = tmpl.get("convert_input", "{np_var}")
          call_args.append(conv_tmpl.format(np_var=aname))

        jit_tmpl = tmpl.get("jit_template")

        if jit_tmpl:
          try_body.append(cst.parse_statement(f"fn = {api}"))
          api_to_call = "fn"
        else:
          api_to_call = api

        fn_call_expr = f"{api_to_call}({', '.join(call_args)})"

        if jit_tmpl:
          static_indices = []
          for idx, arg in enumerate(args):
            if is_static_arg(arg):
              static_indices.append(idx)
          static_argnums = f"{tuple(static_indices)}" if static_indices else "None"

          if "{fn}" in jit_tmpl:
            jit_wrapper = jit_tmpl.format(fn=api_to_call, static_argnums=static_argnums)
            fn_call_expr = f"{jit_wrapper}({', '.join(call_args)})"

        try_body.append(cst.parse_statement(f"res = {fn_call_expr}"))

        to_numpy = tmpl.get("to_numpy", "{res_var}")
        res_conv = to_numpy.format(res_var="res")
        res_line = f"results['{fw}'] = {res_conv}" if not is_void else f"results['{fw}'] = res"
        try_body.append(cst.parse_statement(res_line))

        if not is_void and differentiable and input_is_diff:
          try_body.append(cst.EmptyLine(comment=cst.Comment("# Gradient check")))
          lam_args = [f"a{i}" for i in range(len(call_args))]
          lam_sig = ", ".join(lam_args)
          lam_call_body = f"{api_to_call}({', '.join(lam_args)})"

          grad_templates = {
            "jax": ("jax.grad(lambda {lam_sig}: jnp.sum({lam_call_body}))({input_0})", False),
            "torch": ("torch.func.grad(lambda {lam_sig}: torch.sum({lam_call_body}))({input_0})", True),
          }
          if fw in grad_templates:
            input_0 = call_args[0]
            grad_tmpl, needs_to_numpy = grad_templates[fw]
            grad_line = grad_tmpl.format(lam_sig=lam_sig, lam_call_body=lam_call_body, input_0=input_0)
            try_body.append(cst.parse_statement(f"grad_res = {grad_line}"))
            if needs_to_numpy:
              res_conv_grad = to_numpy.format(res_var="grad_res")
              try_body.append(cst.parse_statement(f"results_grad['{fw}'] = {res_conv_grad}"))
            else:
              try_body.append(cst.parse_statement(f"results_grad['{fw}'] = grad_res"))

        handler = cst.ExceptHandler(
          type=cst.Name("Exception"),
          name=cst.AsName(name=cst.Name("e")),
          body=cst.IndentedBlock(body=[cst.parse_statement(f"print(f'Skipping {fw} due to error: {{e}}')")]),
        )
        try_stmt = cst.Try(body=cst.IndentedBlock(body=try_body), handlers=[handler])

        if_stmt = cst.If(test=cst.Name(f"{fw.upper()}_AVAILABLE"), body=cst.IndentedBlock(body=[try_stmt]))
        func_body.append(if_stmt)

      if is_void:
        func_body.append(cst.EmptyLine(comment=cst.Comment("# Comparison")))
        func_body.append(cst.EmptyLine(comment=cst.Comment("# Operation expected to return None / Void")))
        func_body.append(cst.parse_statement("assert len(results) >= 2"))
      else:
        if return_type == "int":
          func_body.append(cst.EmptyLine(comment=cst.Comment("# Type Check")))
          func_body.append(cst.EmptyLine(comment=cst.Comment("# Expected int")))
          func_body.append(
            cst.parse_statement(
              "for fw, val in results.items():\n    assert np.issubdtype(np.array(val).dtype, np.integer) or isinstance(val, int)"
            )
          )
        elif return_type == "bool":
          func_body.append(cst.EmptyLine(comment=cst.Comment("# Type Check")))
          func_body.append(cst.EmptyLine(comment=cst.Comment("# Expected bool")))
          func_body.append(
            cst.parse_statement(
              "for fw, val in results.items():\n    assert np.issubdtype(np.array(val).dtype, bool) or isinstance(val, bool)"
            )
          )
        elif return_type in ["Tensor", "Array"]:
          func_body.append(cst.EmptyLine(comment=cst.Comment("# Type Check")))
          func_body.append(cst.EmptyLine(comment=cst.Comment("# Expected Array/Tensor")))
          func_body.append(
            cst.parse_statement("for fw, val in results.items():\n    assert isinstance(val, (np.ndarray, np.generic))")
          )

        func_body.append(cst.EmptyLine(comment=cst.Comment("# Comparison")))
        cmp_body = [
          cst.parse_statement("vals = list(results.values())"),
          cst.parse_statement("ref = vals[0]"),
          cst.parse_statement(
            f"for val in vals[1:]:\n    assert verify_results(ref, val, rtol={test_rtol}, atol={test_atol}, {exact_kw})"
          ),
        ]
        func_body.append(
          cst.If(
            test=cst.Comparison(
              left=cst.Call(func=cst.Name("len"), args=[cst.Arg(cst.Name("results"))]),
              comparisons=[cst.ComparisonTarget(operator=cst.GreaterThanEqual(), comparator=cst.Integer("2"))],
            ),
            body=cst.IndentedBlock(body=cmp_body),
          )
        )

        if differentiable and input_is_diff:
          func_body.append(cst.EmptyLine(comment=cst.Comment("# Gradient Verification")))
          func_body.append(cst.EmptyLine(comment=cst.Comment("# Grad Check")))
          grad_cmp = [
            cst.parse_statement("g_vals = list(results_grad.values())"),
            cst.parse_statement("g_ref = g_vals[0]"),
            cst.parse_statement(
              "for g_val in g_vals[1:]:\n    np.testing.assert_allclose(g_ref, g_val, rtol=0.01, atol=0.001)"
            ),
          ]
          func_body.append(
            cst.If(
              test=cst.Comparison(
                left=cst.Call(func=cst.Name("len"), args=[cst.Arg(cst.Name("results_grad"))]),
                comparisons=[cst.ComparisonTarget(operator=cst.GreaterThanEqual(), comparator=cst.Integer("2"))],
              ),
              body=cst.IndentedBlock(body=grad_cmp),
            )
          )

      module_body.append(
        cst.FunctionDef(name=cst.Name(test_func_name), params=cst.Parameters(), body=cst.IndentedBlock(body=func_body))
      )
      module_body.append(cst.EmptyLine())

    if not ops_generated and not existing_tests:
      out_file.write_text(
        '"""Generated tests."""\n# No tests generated due to insufficient variants.\n', encoding="utf-8"
      )
    else:
      mod = cst.Module(body=module_body)
      out_file.write_text(mod.code, encoding="utf-8")
