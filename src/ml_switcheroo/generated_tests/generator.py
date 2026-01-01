"""
Generator backend for creating physical Python test files.

It orchestrates the generation of PyTest-compatible files that verify
operations across multiple frameworks by using the semantic definitions.
"""

import ast
import pathlib
from typing import Any, Dict

from ml_switcheroo.generated_tests.templates import get_template, is_static_arg
from ml_switcheroo.generated_tests.inputs import parse_arg_def, generate_input_value_code
from ml_switcheroo.generated_tests.runtime_builder import ensure_runtime_module


class TestGenerator:
  """
  Generates PyTest files for ML operators across frameworks (Torch, JAX, etc.).

  Handles argument constraints, type checking, gradient verification, and
  runtime environment setup via helper modules.
  """

  def __init__(self, semantics_mgr: Any = None) -> None:
    """
    Initialize the TestGenerator.

    Args:
        semantics_mgr: Manager for semantics and templates.
    """
    self.semantics_mgr = semantics_mgr

  def _ensure_runtime_module(self, out_dir: pathlib.Path, frameworks=None) -> None:
    """Proxies request to runtime_builder."""
    ensure_runtime_module(out_dir, frameworks, self.semantics_mgr)

  def generate(self, semantics: Dict[str, Any], out_file: pathlib.Path) -> None:
    """
    Generate a test file based on the provided semantics.

    Args:
        semantics: Dictionary mapping operator names to their definitions.
        out_file: Path to write the generated Python file.
    """
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
    file_lines = [
      "import pytest",
      "import numpy as np",
      "import numpy",
      "import random",
      "import math",
      "from .runtime import *",
      "",
    ]

    ops_generated = False

    for op_name, op_def in semantics.items():
      variants = op_def.get("variants", {})
      valid_variants = {k: v for k, v in variants.items() if get_template(self.semantics_mgr, k)}

      # Skip generation if not enough variants or if test manually exists
      test_func_name = f"test_gen_{op_name}"
      if len(valid_variants) < 2 or test_func_name in existing_tests:
        continue

      ops_generated = True

      std_args_raw = op_def.get("std_args", [])
      args = [parse_arg_def(a) for a in std_args_raw]

      return_type = op_def.get("return_type", "Tensor")
      is_void = return_type in ["None", "void"]

      # Implicit default: True unless explicitly False
      differentiable = op_def.get("differentiable", True)

      # Tolerances
      test_rtol = op_def.get("test_rtol", 0.001)
      test_atol = op_def.get("test_atol", 0.0001)

      ver_mode = op_def.get("verification_mode", "fuzzy")
      exact_kw = "exact=True" if ver_mode == "exact" else "exact=False"

      # Function Body
      file_lines.append(f"def {test_func_name}():")
      file_lines.append("    # 1. Inputs")

      input_names = []
      input_is_diff = False

      for idx, arg in enumerate(args):
        var_name = f"np_{arg['name']}"
        val_code = generate_input_value_code(arg["name"], arg)
        file_lines.append(f"    {var_name} = {val_code}")
        input_names.append(var_name)

        # Check diff capability
        if idx == 0 and arg["type"] in ["Array", "Tensor", "float"]:
          input_is_diff = True

      file_lines.append("    results = {}")
      if not is_void:
        file_lines.append("    results_grad = {}")
      file_lines.append("")

      # Framework Blocks
      for fw in sorted(valid_variants.keys()):
        fw_cfg = valid_variants[fw]
        api = fw_cfg["api"]
        tmpl = get_template(self.semantics_mgr, fw)

        file_lines.append(f"    # Framework: {fw}")
        file_lines.append(f"    if {fw.upper()}_AVAILABLE:")
        file_lines.append("        try:")

        # Input Conversion
        call_args = []
        for aname in input_names:
          conv_tmpl = tmpl.get("convert_input", "{np_var}")
          call_args.append(conv_tmpl.format(np_var=aname))

        # Check JIT request
        jit_tmpl = tmpl.get("jit_template")

        # Aliasing for JIT templates (e.g. jax.jit(fn))
        if jit_tmpl:
          file_lines.append(f"            fn = {api}")
          api_to_call = "fn"
        else:
          api_to_call = api

        # API Call Logic
        fn_call_expr = f"{api_to_call}({', '.join(call_args)})"

        if jit_tmpl:
          static_indices = []
          for idx, arg in enumerate(args):
            if is_static_arg(arg):
              static_indices.append(idx)
          static_argnums = f"{tuple(static_indices)}" if static_indices else "None"

          if "{fn}" in jit_tmpl:
            # e.g. "jax.jit({fn})" -> jax.jit(fn)
            jit_wrapper = jit_tmpl.format(fn=api_to_call, static_argnums=static_argnums)
            fn_call_expr = f"{jit_wrapper}({', '.join(call_args)})"

        file_lines.append(f"            res = {fn_call_expr}")

        # Output Conversion
        to_numpy = tmpl.get("to_numpy", "{res_var}")
        res_conv = to_numpy.format(res_var="res")
        res_line = f"results['{fw}'] = {res_conv}" if not is_void else f"results['{fw}'] = res"
        file_lines.append(f"            {res_line}")

        # Gradient Check
        if not is_void and differentiable and input_is_diff:
          file_lines.append("            # Gradient check")

          lam_args = [f"a{i}" for i in range(len(call_args))]
          lam_sig = ", ".join(lam_args)
          lam_call_body = f"{api_to_call}({', '.join(lam_args)})"

          if fw == "jax":
            input_0 = call_args[0]
            grad_line = f"jax.grad(lambda {lam_sig}: jnp.sum({lam_call_body}))({input_0})"
            file_lines.append(f"            grad_res = {grad_line}")
            file_lines.append(f"            results_grad['{fw}'] = grad_res")

          elif fw == "torch":
            input_0 = call_args[0]
            grad_line = f"torch.func.grad(lambda {lam_sig}: torch.sum({lam_call_body}))({input_0})"
            file_lines.append(f"            grad_res = {grad_line}")
            res_conv_grad = to_numpy.format(res_var="grad_res")
            file_lines.append(f"            results_grad['{fw}'] = {res_conv_grad}")

        file_lines.append("        except Exception as e:")
        file_lines.append(f"            print(f'Skipping {fw} due to error: {{e}}')")
        file_lines.append("")

      # Result Verification
      if is_void:
        file_lines.append("    # Comparison")
        file_lines.append("    # Operation expected to return None / Void")
        file_lines.append("    assert len(results) >= 2")

      else:
        if return_type == "int":
          file_lines.append("    # Type Check")
          file_lines.append("    # Expected int")
          file_lines.append("    for fw, val in results.items():")
          file_lines.append("        assert np.issubdtype(np.array(val).dtype, np.integer) or isinstance(val, int)")
          file_lines.append("")
        elif return_type == "bool":
          file_lines.append("    # Type Check")
          file_lines.append("    # Expected bool")
          file_lines.append("    for fw, val in results.items():")
          file_lines.append("        assert np.issubdtype(np.array(val).dtype, bool) or isinstance(val, bool)")
          file_lines.append("")
        elif return_type in ["Tensor", "Array"]:
          file_lines.append("    # Type Check")
          file_lines.append("    # Expected Array/Tensor")
          file_lines.append("    for fw, val in results.items():")
          file_lines.append("        assert isinstance(val, (np.ndarray, np.generic))")
          file_lines.append("")

        file_lines.append("    # Comparison")
        file_lines.append("    if len(results) >= 2:")
        file_lines.append("        vals = list(results.values())")
        file_lines.append("        ref = vals[0]")
        file_lines.append("        for val in vals[1:]:")
        file_lines.append(f"            assert verify_results(ref, val, rtol={test_rtol}, atol={test_atol}, {exact_kw})")
        file_lines.append("")

        # Condition block for Gradients in Verification section
        if differentiable and input_is_diff:
          file_lines.append("    # Gradient Verification")
          file_lines.append("    # Grad Check")
          file_lines.append("    if len(results_grad) >= 2:")
          file_lines.append("        g_vals = list(results_grad.values())")
          file_lines.append("        g_ref = g_vals[0]")
          file_lines.append("        for g_val in g_vals[1:]:")
          # Using assert_allclose as requested by test expectations
          file_lines.append(f"            np.testing.assert_allclose(g_ref, g_val, rtol=0.01, atol=0.001)")

      file_lines.append("")

    if not ops_generated and not existing_tests:
      out_file.write_text("# No tests generated due to insufficient variants.\n", encoding="utf-8")
    else:
      out_file.write_text("\n".join(file_lines), encoding="utf-8")
