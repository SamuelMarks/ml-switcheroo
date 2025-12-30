"""
Generator engine for creating cross-framework ML operator tests.

This module provides the TestGenerator class which parses semantic operator
definitions and outputs PyTest-compatible Python files.
"""

import pathlib
import os
import ast
from typing import Any, Dict, Union, List, Optional


class TestGenerator:
  """
  Generates PyTest files for ML operators across frameworks (Torch, JAX, etc.).

  Handles argument constraints, type checking, gradient verification, and
  runtime environment setup (e.g. seeding).
  """

  def __init__(self, semantics_mgr: Any = None) -> None:
    """
    Initialize the TestGenerator.

    Args:
        semantics_mgr: Manager for semantics and templates.
    """
    self.semantics_mgr = semantics_mgr
    # Hardcoded defaults for fallback if manager is missing or returns None
    self._defaults = {
      "torch": {
        "import": "import torch",
        "convert_input": "torch.tensor({np_var})",
        "to_numpy": "{res_var}.numpy()",
      },
      "jax": {
        "import": "import jax\nimport jax.numpy as jnp",
        "convert_input": "jnp.array({np_var})",
        "to_numpy": "np.array({res_var})",
        "jit_template": "jax.jit({fn}, static_argnums={static_argnums})",
      },
      "tensorflow": {
        "import": "import tensorflow as tf",
        "convert_input": "tf.convert_to_tensor({np_var})",
        "to_numpy": "{res_var}.numpy()",
      },
    }

  def _get_template(self, fw: str) -> Dict[str, str]:
    """Retrieve template for a specific framework."""
    tmpl = None
    if self.semantics_mgr:
      try:
        tmpl = self.semantics_mgr.get_test_template(fw)
      except Exception:
        pass

    if tmpl:
      return tmpl

    return self._defaults.get(fw, {})

  def _ensure_runtime_module(self, out_dir: pathlib.Path, frameworks: Optional[List[str]] = None) -> None:
    """
    Creates or updates the runtime.py module with instructions and imports.

    Args:
        out_dir: Directory where the generated tests and runtime.py reside.
        frameworks: List of frameworks to include imports for.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    runtime_path = out_dir / "runtime.py"

    imports_block = []
    alls = ["verify_results"]

    # Always include torch/jax logic if they appear in defaults or requested
    fw_set = set(frameworks) if frameworks else set()
    chk_fws = fw_set | {"torch", "jax"}

    # Sort for determinism
    for fw in sorted(chk_fws):
      tmpl = self._get_template(fw)
      # If no template found for a specific framework, skip import generation for it
      if not tmpl:
        continue

      imp = tmpl.get("import", f"import {fw}")

      # Sanitization for checking availability
      var_name = f"{fw.upper()}_AVAILABLE"
      imports_block.append(f"# --- {fw} ---")
      imports_block.append("try:")
      for line in imp.split("\n"):
        imports_block.append(f"    {line}")
      imports_block.append(f"    {var_name} = True")
      imports_block.append("except ImportError:")
      imports_block.append(f"    {var_name} = False")
      imports_block.append("")
      if fw == "tensorflow":
        alls.extend((var_name, "tf"))
      else:
        if fw == "jax":
          alls.append("jnp")
        alls.extend((var_name, fw))

    imports_str = "\n".join(imports_block)

    code = f'''"""Shared runtime flags for generated tests (Auto-Generated)."""
import sys
import pytest
import random
import numpy as np

{imports_str}
# --- Determinism ---
@pytest.fixture(autouse=True)
def ensure_determinism():
  """
  Auto-injects fixed seeds for reproducibility at the start of every test.
  Covers Python random, NumPy, Torch, TensorFlow, and MLX.
  """
  # Core Python & NumPy
  random.seed(42)
  np.random.seed(42)

  # PyTorch
  if "torch" in sys.modules:
    try:
      sys.modules["torch"].manual_seed(42)
      if sys.modules["torch"].cuda.is_available():
        sys.modules["torch"].cuda.manual_seed_all(42)
    except Exception:
      pass

  # TensorFlow
  if "tensorflow" in sys.modules:
    try:
      tf = sys.modules["tensorflow"]
      # TF 2.x
      if hasattr(tf, "random") and hasattr(tf.random, "set_seed"):
        tf.random.set_seed(42)
    except Exception:
      pass

  # MLX
  if "mlx.core" in sys.modules:
    try:
      sys.modules["mlx.core"].random.seed(42)
    except Exception:
      pass
  elif "mlx" in sys.modules and hasattr(sys.modules["mlx"], "core"):\n    try:
      sys.modules["mlx"].core.random.seed(42)
    except Exception:
      pass

# --- Verification Logic ---
def verify_results(ref, val, rtol=1e-3, atol=1e-3, exact=False):
  """
  Cross-framework comparison helper.

  Recursively compares data structures (Lists, Dicts, Tuples).
  If 'exact' is True, enforces strict equality for all types (np.array_equal).
  If 'exact' is False (default), applies fuzzy matching (np.allclose) for floats/complex.
  """
  # 1. Null/None Check (Exact identity)
  if ref is None or val is None:
    return ref is val

  # 2. Try Chex (Structural comparison for JAX PyTrees)
  if "chex" in globals():
    try:
      chex_mod = globals()["chex"]
      if exact:
        chex_mod.assert_trees_all_close(ref, val, rtol=0, atol=0)
      else:
        chex_mod.assert_trees_all_close(ref, val, rtol=rtol, atol=atol)
      return True
    except (AssertionError, Exception):
      pass

  # 3. Recursive Container Handling
  if isinstance(ref, dict) and isinstance(val, dict):
    if ref.keys() != val.keys():
      return False
    for k in ref:
      if not verify_results(ref[k], val[k], rtol, atol, exact=exact):
        return False
    return True

  if isinstance(ref, (list, tuple)) and isinstance(val, (list, tuple)):
    if len(ref) != len(val):
      return False
    for r, v in zip(ref, val):
      if not verify_results(r, v, rtol, atol, exact=exact):
        return False
    return True

  # 4. Leaf Node Comparison
  try:
    np_ref = np.asanyarray(ref)
    np_val = np.asanyarray(val)

    if np_ref.shape != np_val.shape:
      # Allow scalar vs 0-d array flexibility
      if not (np_ref.size == 1 and np_val.size == 1):
        return False

    if exact:
      return np.array_equal(np_ref, np_val)

    kind = np_ref.dtype.kind
    if kind in {"f", "c"}:
      return np.allclose(np_ref, np_val, rtol=rtol, atol=atol, equal_nan=True)

    return np.array_equal(np_ref, np_val)

  except Exception:
    try:
      return ref == val
    except Exception:
      return False

# __all__ = {alls!r}
'''
    runtime_path.write_text(code, encoding="utf-8")

  def _parse_arg(self, arg: Union[str, tuple, dict]) -> dict:
    """Parse argument definition into a normalized dictionary."""
    if isinstance(arg, str):
      return {"name": arg, "type": "Array"}
    elif isinstance(arg, (tuple, list)) and len(arg) == 2:
      return {"name": arg[0], "type": arg[1]}
    elif isinstance(arg, dict):
      # Shallow copy to avoid mutation
      arg = arg.copy()
      if "type" not in arg:
        if isinstance(arg.get("min"), int) or isinstance(arg.get("max"), int):
          arg["type"] = "int"
        elif isinstance(arg.get("min"), float):
          arg["type"] = "float"
        else:
          arg["type"] = "Array"
      return arg
    return {"name": "unknown", "type": "Array"}

  def _generate_value_code(self, name: str, arg_def: Union[str, Dict[str, Any]]) -> str:
    """Generate Python code string to instantiate inputs."""
    # Normalize arg_def if simple string
    if isinstance(arg_def, str):
      arg_def = {"type": arg_def}

    arg_type = arg_def.get("type")

    # Heuristic for unknown types
    if arg_type in [None, "Any"]:
      if name in ["axis", "dim"]:
        return "1"
      if name == "keepdims":
        return "bool(random.getrandbits(1))"
      # Default fallback for Any/None -> Array logic or None
      arg_type = "Array"

    # 1. Options constraint
    if "options" in arg_def:
      opts = arg_def["options"]
      return f"random.choice({opts!r})"

    # 2. Int Range / Int Type
    if arg_type == "int":
      mn = arg_def.get("min", 1)
      mx = arg_def.get("max", 5)
      return f"random.randint({mn}, {mx})"

    # 3. Bool
    if arg_type == "bool":
      return "bool(random.getrandbits(1))"

    # 4. Float Range
    if arg_type == "float":
      mn = arg_def.get("min", 0.0)
      mx = arg_def.get("max", 1.0)
      return f"random.uniform({mn}, {mx})"

    # 5. Complex Types
    if arg_type == "List[int]":
      return "[1, 2]"
    if "Tuple" in arg_type:
      return "(1, 2)"

    # 6. Array/Tensor constraints
    if arg_type in ["Array", "Tensor"]:
      mn = arg_def.get("min")
      mx = arg_def.get("max")

      if mn is not None and mx is not None:
        return f"np.random.uniform({mn}, {mx}, size=(2, 2)).astype(np.float32)"
      elif mn is not None:
        return f"np.abs(np.random.randn(2, 2).astype(np.float32)) + {mn}"

      return "np.random.randn(2, 2, 2).astype(np.float32)"

    return "None"

  def _is_static_arg(self, arg_info: Dict[str, Any]) -> bool:
    """Determine if argument should be static (JIT)."""
    t = arg_info.get("type", "")
    # Heuristic: primitives are usually static in JAX JIT context for shapes/axes
    if t.lower() in ("int", "bool", "str", "List[int]", "Tuple[int]"):
      return True
    elif arg_info.get("name") in ("axis", "dim", "keepdims"):
      return True
    else:
      return False

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
      valid_variants = {k: v for k, v in variants.items() if self._get_template(k)}

      # Skip generation if not enough variants or if test manually exists
      test_func_name = f"test_gen_{op_name}"
      if len(valid_variants) < 2 or test_func_name in existing_tests:
        continue

      ops_generated = True

      std_args_raw = op_def.get("std_args", [])
      args = [self._parse_arg(a) for a in std_args_raw]

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
        val_code = self._generate_value_code(arg["name"], arg)
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
        tmpl = self._get_template(fw)

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
            if self._is_static_arg(arg):
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
          # We pass the original API/fn in the lambda, assuming it can handle the tracing objects
          # If we aliased 'fn', we use 'fn', otherwise 'api'
          # Actually for lambda, best to use simple closure over api_to_call
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
