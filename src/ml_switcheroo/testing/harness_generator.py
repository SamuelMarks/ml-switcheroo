"""
Integration Test Harness Generator.

This module is responsible for creating temporary verification scripts (harnesses)
that dynamically import the original Source code and the converted Target code,
feed them identical inputs, and verify that their outputs match.

Crucially, the generated harness is **Standalone**: it does not import from
the `ml_switcheroo` package, allowing verification in isolated environments.

**Update (Split-Brain Fix)**:
Uses `CodeExtractor` to inject the *actual* code of `InputFuzzer` into the harness script,
eliminating the duplicated `StandaloneFuzzer` maintained in string templates.
This ensures that type hint improvements in the main codebase are automatically
reflected in the verification harness.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional

from ml_switcheroo.testing.harness_generator_template import HARNESS_TEMPLATE
from ml_switcheroo.testing.fuzzer import InputFuzzer
from ml_switcheroo.utils.code_extractor import CodeExtractor


class HarnessGenerator:
  """
  Generates a standalone Python script to verify transpilation correctness.
  Dynamically extracts the `InputFuzzer` logic to embed it within the script.

  Attributes:
      extractor (CodeExtractor): Utility to source code from runtime objects.
  """

  def __init__(self) -> None:
    self.extractor = CodeExtractor()

  def generate(
    self,
    source_file: Path,
    target_file: Path,
    output_harness: Path,
    source_fw: str = "torch",
    target_fw: str = "jax",
    semantics: Optional[Dict[str, Any]] = None,
  ) -> None:
    """
    Writes the standalone verification script to disk.

    It reads the source code of `InputFuzzer` and injects it into the template,
    renaming it to `StandaloneFuzzer` to match the template's expectations if needed,
    or simply using the extracted class.

    Args:
        source_file: Path to the original source code.
        target_file: Path to the converted source code.
        output_harness: Where to save the `verify_xyz.py` script.
        source_fw: Name of source framework (for fuzzer adaptation).
        target_fw: Name of target framework (for fuzzer adaptation).
        semantics: Optional semantics dictionary to extract type hints.
    """
    # 1. Extract Fuzzer Logic
    # We extract InputFuzzer directly from the codebase.
    fuzzer_code = self.extractor.extract_class(InputFuzzer)

    # 2. Serialize Hints
    hints_map = {}
    if semantics:
      for op_name, details in semantics.items():
        args_data = details.get("std_args", [])
        func_hints = {}
        for arg in args_data:
          if isinstance(arg, (list, tuple)) and len(arg) == 2:
            name, type_str = arg
            func_hints[name] = type_str
        if func_hints:
          hints_map[op_name] = func_hints

    hints_json = json.dumps(hints_map).replace("'", '"')

    # 3. Construct Script
    # We need to ensure the harness template uses the extracted class name.
    # InputFuzzer is extracted as 'class InputFuzzer...'.
    # The template previously expected 'StandaloneFuzzer'.
    # We alias it at the end of the injection block.
    fuzzer_block = (
      f"{fuzzer_code}\n\n"
      "# Alias to match template expectation\n"
      "class StandaloneFuzzer(InputFuzzer):\n"
      "    # Override adapter logic to be self-contained if needed\n"
      "    # But InputFuzzer uses 'get_adapter' from ml_switcheroo.testing.adapters\n"
      "    # which is NOT available here. We must shim the adapter logic.\n"
      "    pass\n"
    )

    # 4. Injection of Missing Dependencies (Shim)
    # Since InputFuzzer imports `get_adapter` which won't exist in the standalone script,
    # we must inject a shim for `adapt_to_framework` or `get_adapter`.
    # Actually, `InputFuzzer.adapt_to_framework` calls `get_adapter`.
    # In the standalone script, we can't import from ml_switcheroo.
    # Solution: We inject a 'ShimmedInputFuzzer' mixin or overwrite methods in the alias.

    adapter_shim = self._generate_adapter_shim()

    script_content = HARNESS_TEMPLATE.format(
      source_path=source_file.resolve(),
      target_path=target_file.resolve(),
      source_fw=source_fw,
      target_fw=target_fw,
      hints_json=hints_json,
      fuzzer_implementation=f"{adapter_shim}\n{fuzzer_block}",
    )

    with open(output_harness, "wt", encoding="utf-8") as f:
      f.write(script_content)

  def _generate_adapter_shim(self) -> str:
    """
    Generates code to shim the `get_adapter` functionality required by InputFuzzer.
    In a standalone script, we don't have the registry, so we hardcode a robust switch.
    """
    return r"""
# Shim for missing ml_switcheroo.testing.adapters
def get_adapter(framework):
    class GenericAdapter:
        def convert(self, data):
            try:
                import numpy as np
            except ImportError:
                return data
            
            if not isinstance(data, (np.ndarray, np.generic)):
                return data

            if framework == "torch":
                import torch
                try:
                    return torch.from_numpy(data)
                except:
                    return torch.tensor(data)
            elif framework == "jax":
                import jax.numpy as jnp
                return jnp.array(data)
            elif framework == "tensorflow":
                import tensorflow as tf
                return tf.convert_to_tensor(data)
            elif framework == "mlx":
                import mlx.core as mx
                return mx.array(data)
            return data
            
    return GenericAdapter()
"""
