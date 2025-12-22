"""
Dynamic Test Code Generator (Physical Synthesizer).

This module is responsible for creating physical Python test files based on the
Semantic Knowledge Base.

It operations in two stages:
1.  **Template Resolution**: It aggregates execution templates (`import`, `convert_input`, `to_numpy`)
    from the SemanticsManager, prioritizing overlays (snapshots) over registry defaults.
2.  **Code Emission**: It iterates over Operations in the Spec, generating `def test_gen_<opname>()`
    functions that:
    - Generate random inputs using type hints.
    - Execute the operation across all available backends (Torch, JAX, etc.).
    - Compare outputs using `np.allclose`.
    - **Verify Output Shape** if `output_shape_calc` is defined in the ODL.

JIT Logic:
    If a template specifies `jit_wrap: True` (e.g., JAX), the generator inspects the
    framework traits to identify `static_argnums` (arguments like `axis` or `keepdims`)
    and generates valid `jax.jit(fn, static_argnums=(...))` wrappers.
"""

from pathlib import Path
from typing import Dict, Any, List, Set, Tuple, Optional, Union

from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.semantics.schema import StructuralTraits


class TestGenerator:
  """
  Generates Pytest-compatible test files from Semantic Operations.

  Attributes:
      semantics_mgr (SemanticsManager): Source for op definitions and backend templates.
      _processed_fws (Set[str]): Tracks which frameworks were seen to optimize imports.
  """

  def __init__(self, semantics_mgr: Optional[SemanticsManager] = None):
    """
    Initializes the generator.

    Args:
        semantics_mgr: Valid SemanticsManager. Should be pre-loaded.
    """
    self.semantics_mgr = semantics_mgr or SemanticsManager()
    self._processed_fws: Set[str] = set()
    self._jit_static_cache: Dict[str, Set[str]] = {}

  def _get_static_args_for_framework(self, fw_name: str) -> Set[str]:
    """
    Lazily loads static argument keywords (like 'axis') from the framework config.
    """
    if fw_name in self._jit_static_cache:
      return self._jit_static_cache[fw_name]

    config_dict = self.semantics_mgr.get_framework_config(fw_name)
    static_set = set()

    if config_dict and "traits" in config_dict:
      try:
        traits = StructuralTraits.model_validate(config_dict["traits"])
        static_set = set(traits.jit_static_args)
      except Exception:
        pass

    self._jit_static_cache[fw_name] = static_set
    return static_set

  def generate(self, semantics: Dict[str, Any], output_path: Path) -> None:
    """
    Generates a test file containing functions for each operation in semantics.

    Args:
        semantics: Dictionary of operations (result of mgr.get_known_apis()).
        output_path: Destination path for the .py file.
    """
    existing_tests = self._parse_existing_tests(output_path)
    new_test_funcs: List[str] = []

    # Reset tracker for this generation run
    self._processed_fws = set()

    # 1. Generate Test Functions
    # Sort keys to ensure deterministic file generation
    for op_name in sorted(semantics.keys()):
      details = semantics[op_name]
      func_name = f"test_gen_{op_name}"

      # Skip if user manually defined this test (Human-in-the-Loop Override)
      if func_name in existing_tests:
        continue

      variants_map = details.get("variants", {})

      # Only generate blocks for frameworks we have templates for
      valid_variants = {}
      for fw, info in variants_map.items():
        if info and ("api" in info or "requires_plugin" in info):
          template = self.semantics_mgr.get_test_template(fw)
          if template:
            valid_variants[fw] = info

      # Heuristic: Minimum 2 backends required to perform a comparison
      if len(valid_variants) < 2:
        continue

      # Extract argument list with types (Feature 027)
      # format: [("x", "Array"), ("axis", "int")] OR just ["x", "axis"]
      std_args_raw = details.get("std_args", ["x"])
      args_info: List[Tuple[str, str]] = []

      for item in std_args_raw:
        if isinstance(item, (list, tuple)) and len(item) == 2:
          args_info.append((item[0], item[1]))
        elif isinstance(item, str):
          args_info.append((item, "Any"))
        elif isinstance(item, dict):
          # ODL Dictionary format
          n = item.get("name", f"arg_{len(args_info)}")
          t = item.get("type", "Any")
          args_info.append((n, t))
        else:
          # Fallback for weird structures
          args_info.append((f"arg_{len(args_info)}", "Any"))

      test_code = self._build_function(op_name, valid_variants, args_info, details)
      new_test_funcs.append(test_code)

    if not new_test_funcs:
      print("✨ No new tests to generate.")
      return

    # 2. Build File Content
    # We assume standard numpy import is always needed for inputs
    imports = ["import pytest", "import numpy as np", "import random"]

    # Add framework imports based on what we processed
    for fw in sorted(list(self._processed_fws)):
      template = self.semantics_mgr.get_test_template(fw)
      if template and "import" in template:
        # Split multiline imports
        lines = template["import"].split("\n")
        for line in lines:
          if line.strip() and line not in imports:
            imports.append(line.strip())

    header = "\n".join(imports)
    body = "\n\n".join(new_test_funcs)

    # 3. Write
    mode = "w"  # Always overwrite for reproducibility
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, mode, encoding="utf-8") as f:
      f.write(header + "\n\n")
      f.write(body + "\n")

    print(f"📝 Generated {len(new_test_funcs)} new tests in {output_path}")

  def _build_function(
    self,
    op_name: str,
    variants: Dict[str, Any],
    args_info: List[Tuple[str, str]],
    details: Dict[str, Any],
  ) -> str:
    """
    Constructs a single test function string.
    Includes logic for JIT wrapping and Output Shape Verification.

    Args:
        op_name: Name of the operation (e.g. 'sum').
        variants: Dict of implementations.
        args_info: List of (name, type) tuples used for input generation strategies.
        details: Full operation definition dictionary (for shape calc).

    Returns:
        String containing the python code for the test function.
    """
    lines = [f"def test_gen_{op_name}():"]
    lines.append(f"    # Generated Test for {op_name}")

    # 1. Inputs generation
    lines.append("    # 1. Inputs")
    input_vars = []
    arg_names = []

    for name, type_hint in args_info:
      var_name = f"np_{name}"
      input_vars.append(var_name)
      arg_names.append(name)

      val_code = self._generate_value_code(name, type_hint)
      lines.append(f"    {var_name} = {val_code}")

    lines.append("    results = {}")

    # 2. Execution Blocks
    for fw in sorted(variants.keys()):
      info = variants[fw]
      self._processed_fws.add(fw)

      api_call = info.get("api", f"plugin_required_{op_name}")
      template = self.semantics_mgr.get_test_template(fw)
      if not template:
        continue

      lines.append(f"\n    # Framework: {fw}")
      lines.append("    try:")

      # Convert Inputs
      call_args = []
      for i, _ in enumerate(arg_names):
        input_var = input_vars[i]
        convert_tmpl = template.get("convert_input", "{np_var}")
        arg_code = convert_tmpl.format(np_var=input_var)
        call_args.append(arg_code)

      call_str = ", ".join(call_args)

      # JIT Wrapping Check (Boolean logic handling string "True" from JSON)
      should_jit = str(template.get("jit_wrap", "False")).lower() == "true"

      if should_jit:
        lines.append("        # JIT Compilation Check")

        # Load Static Keywords specifically for this framework
        static_keywords = self._get_static_args_for_framework(fw)

        # Creating a lambda to wrap the call `lambda a0, a1: api(a0, a1)`
        lambda_args = ", ".join([f"a{k}" for k in range(len(arg_names))])
        lambda_call = ", ".join([f"a{k}" for k in range(len(arg_names))])

        # Detect Static Args indices based on dynamic keywords
        static_indices = []
        for idx, name in enumerate(arg_names):
          if name in static_keywords:
            static_indices.append(str(idx))

        jit_kwargs = ""
        if static_indices:
          # e.g. static_argnums=(1,)
          idx_str = ", ".join(static_indices)
          if len(static_indices) == 1:
            idx_str += ","
          jit_kwargs = f", static_argnums=({idx_str})"

        lines.append(f"        fn = lambda {lambda_args}: {api_call}({lambda_call})")
        lines.append(f"        jitted_fn = jax.jit(fn{jit_kwargs})")
        lines.append(f"        res = jitted_fn({call_str})")

      else:
        # Standard Execution
        lines.append(f"        res = {api_call}({call_str})")

      # Normalize Output
      normalize_tmpl = template.get("to_numpy", "{res_var}")
      normalize_code = normalize_tmpl.format(res_var="res")
      lines.append(f"        results['{fw}'] = {normalize_code}")

      lines.append("    except Exception as e:")
      # Use 'print' to debug in pytest -s, but otherwise just skip recording result
      lines.append(f"        print(f'Skipping {fw} due to error: {{e}}')")

    # 3. Shape Verification (Feature 20)
    # If the semantic definition provides a calculation lambda string, we insert checks.
    shape_calc = details.get("output_shape_calc")
    if shape_calc:
      lines.append("\n    # 3. Shape Verification")
      lines.append(f"    shape_fn = {shape_calc}")
      # Pass the numpy input variables to the shape function
      args_str = ", ".join(input_vars)
      lines.append(f"    expected_shape = shape_fn({args_str})")
      lines.append("    for fw, val in results.items():")
      lines.append("        if hasattr(val, 'shape'):")
      lines.append(
        "            assert val.shape == expected_shape, f'{fw} shape mismatch: {val.shape} != {expected_shape}'"
      )

    # 4. Comparison Logic
    lines.append("\n    # 4. Comparison")
    lines.append("    if len(results) < 2:")
    lines.append("        pytest.skip('Not enough successful backends to compare')")
    lines.append("")
    lines.append("    vals = list(results.values())")
    lines.append("    ref = vals[0]")
    lines.append("    for val in vals[1:]:")

    # Use simple tolerance comparison.
    # Note: If output types are non-array (ints/bools), allclose handles it if coerced.
    lines.append("        if hasattr(ref, 'shape') and hasattr(val, 'shape'):")
    lines.append("            np.testing.assert_allclose(ref, val, rtol=1e-3, atol=1e-3)")
    lines.append("        else:")
    lines.append("            assert ref == val")

    return "\n".join(lines)

  def _generate_value_code(self, name: str, type_hint: str) -> str:
    """
    Produces the Python code string to generate a value for the given argument.
    """
    t = type_hint.lower()

    # --- Strategy 1: Explicit Type Hints ---
    if "array" in t or "tensor" in t:
      return "np.random.randn(2, 2, 2).astype(np.float32)"

    if "list" in t and "int" in t:
      return "[1, 2]"
    if "tuple" in t and "int" in t:
      return "(1, 2)"

    if "bool" in t:
      return "bool(random.getrandbits(1))"
    if "int" in t:
      return "random.randint(1, 3)"
    if "float" in t:
      return "random.uniform(0.1, 1.0)"
    if "str" in t or "string" in t:
      return "'test_string'"

    # --- Strategy 2: Heuristics ---
    name_lower = name.lower()
    if name_lower in ["axis", "dim"]:
      return "1"
    if name_lower in ["keepdims", "keepdim"]:
      return "True"
    if "shape" in name_lower or "size" in name_lower:
      return "(2, 2)"

    # --- Fallback ---
    return "np.random.randn(2, 2, 2).astype(np.float32)"

  def _parse_existing_tests(self, path: Path) -> Set[str]:
    """Reads file to find existing test naming collisions."""
    if not path.exists():
      return set()

    try:
      import ast

      tree = ast.parse(path.read_text(encoding="utf-8"))
      return {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    except Exception:
      return set()
