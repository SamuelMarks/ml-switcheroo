"""
Integration Test Harness Generator.

This module creates self-contained Python scripts for verifying transpilation accuracy.
It dynamically extracts source code from the live system to ensure that generated
harnesses support ALL registered frameworks (Torch, JAX, Keras, etc.) without
hardcoding switch statements.

Capabilities:
1. **Dynamic Shim Generation**: Reads `convert()` methods from all registered
   Framework Adapters and synthesizes a standalone `get_adapter` function.
2. **Fuzzer Inlining**: Extracts the `InputFuzzer` class source code.
3. **Isolation**: Generated scripts run without `ml_switcheroo` installed.
"""

import json
import inspect
import textwrap
from pathlib import Path
from typing import Dict, Any, Optional, List

from ml_switcheroo.testing.harness_generator_template import HARNESS_TEMPLATE
from ml_switcheroo.testing.fuzzer import InputFuzzer
from ml_switcheroo.utils.code_extractor import CodeExtractor
from ml_switcheroo.frameworks.base import _ADAPTER_REGISTRY, FrameworkAdapter


class HarnessGenerator:
  """
  Generates standalone verification scripts.

  Attributes:
      extractor (CodeExtractor): Tool to read source from live objects.
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
    """
    # 1. Extract Fuzzer Logic
    fuzzer_code = self.extractor.extract_class(InputFuzzer)

    # 2. Serialize Hints
    hints_map = {}
    if semantics:
      for op_name, details in semantics.items():
        args_data = details.get("std_args", [])
        func_hints = {}
        for arg in args_data:
          if isinstance(arg, (list, tuple)) and len(arg) == 2:
            func_hints[arg[0]] = arg[1]
        if func_hints:
          hints_map[op_name] = func_hints

    hints_json = json.dumps(hints_map).replace("'", '"')

    # 3. Generate Dynamic Adapter Shim
    # This introspects all registered adapters (including Keras)
    # and builds a comprehensive switch statement.
    adapter_shim = self._generate_adapter_shim()

    # 4. Assemble Code Block
    fuzzer_block = (
      f"{adapter_shim}\n\n"
      f"{fuzzer_code}\n\n"
      "# Alias matching template expectation\n"
      "class StandaloneFuzzer(InputFuzzer):\n"
      "    pass\n"
    )

    # 5. Populate Template
    script_content = HARNESS_TEMPLATE.format(
      source_path=source_file.resolve().as_posix(),
      target_path=target_file.resolve().as_posix(),
      source_fw=source_fw,
      target_fw=target_fw,
      hints_json=hints_json,
      fuzzer_implementation=fuzzer_block,
    )

    # 6. Write to Disk
    output_harness.parent.mkdir(parents=True, exist_ok=True)
    with open(output_harness, "wt", encoding="utf-8") as f:
      f.write(script_content)

  def _generate_adapter_shim(self) -> str:
    """
    Introspects registered frameworks to build the `get_adapter` function.

    It extracts the body of the `convert(self, data)` method from each
    registered adapter class and inlines it into an `if/elif` block.
    """
    # Base template for the shim
    shim_lines = [
      "# Shim for missing ml_switcheroo.frameworks.get_adapter",
      "def get_adapter(framework):",
      "    class GenericAdapter:",
      "        def convert(self, data):",
      "            # Passthrough non-arrays",
      "            try:",
      "                import numpy as np",
      "                if not isinstance(data, (np.ndarray, np.generic)) and not isinstance(data, (list, tuple)):",
      "                    return data",
      "            except ImportError:",
      "                pass",
      "",
    ]

    # Iterate all registered frameworks (e.g. torch, jax, keras, tensorflow)
    # Sorting ensures deterministic output
    frameworks = sorted(_ADAPTER_REGISTRY.keys())

    first = True
    for fw_name in frameworks:
      adapter_cls = _ADAPTER_REGISTRY[fw_name]

      # Skip if abstract or missing convert (shouldn't happen for valid adapters)
      if not hasattr(adapter_cls, "convert"):
        continue

      # Extract the raw source of the convert method
      try:
        method_source = inspect.getsource(adapter_cls.convert)
      except OSError:
        continue  # Cannot extract (e.g. dynamic class)

      # Clean indentation: The extracted source includes 'def convert(self, data):'
      # We want just the body.

      # Dedent the block
      clean_block = textwrap.dedent(method_source)
      lines = clean_block.splitlines()

      # Find body start (skip decorator if any, and def line)
      body_start = 0
      for i, line in enumerate(lines):
        if line.strip().startswith("def convert"):
          body_start = i + 1
          break

      body_lines = lines[body_start:]
      if not body_lines:
        continue

      # Check logic for 'if' block
      condition_kw = "if" if first else "elif"
      shim_lines.append(f"            {condition_kw} framework == '{fw_name}':")

      # Indent the body lines by 16 spaces (3 levels deep + if block)
      for line in body_lines:
        # We need to re-indent relative to the if block
        # The extracted body lines have indentation relative to the class method
        # We strip common indentation then add our own
        stripped = line.strip()
        if stripped:
          # Calculate original indent depth relative to 'def'
          # But simple strip/reindent is safer for generated code
          # NOTE: This assumes convert methods are relatively flat.
          # Complex logic might break with naive strip.
          # Better approach: textwrap.indent
          pass

      # Robust Re-indentation using textwrap
      base_indent = " " * 16  # Align with the if block
      body_block = textwrap.dedent("\n".join(body_lines))
      indented_body = textwrap.indent(body_block, base_indent)
      shim_lines.append(indented_body)

      first = False

    # Fallback
    shim_lines.append("")
    shim_lines.append("            # Default/NumPy")
    shim_lines.append("            return data")
    shim_lines.append("")
    shim_lines.append("    return GenericAdapter()")

    return "\n".join(shim_lines)
