"""
Integration Test Harness Generator.

This module creates self-contained Python scripts for verifying transpilation accuracy.
It dynamically extracts source code from the live system to ensure that generated
harnesses support ALL registered frameworks (Torch, JAX, Keras, etc.) without
hardcoding switch statements.
"""

import json
import inspect
import textwrap
import re
from pathlib import Path
from typing import Dict, Any, Optional, List

from ml_switcheroo.testing.harness_generator_template import HARNESS_TEMPLATE
from ml_switcheroo.testing.fuzzer import InputFuzzer
from ml_switcheroo.utils.code_extractor import CodeExtractor
from ml_switcheroo.frameworks.base import _ADAPTER_REGISTRY, get_adapter


class HarnessGenerator:
  """
  Generates standalone verification scripts tailored to the target framework.
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

    Args:
        source_file: Path to original source.
        target_file: Path to transpiled result.
        output_harness: Destination path.
        source_fw: Key of source framework.
        target_fw: Key of target framework.
        semantics: Optional dictionary of semantic definitions (for type hints).
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
    adapter_shim = self._generate_adapter_shim()

    # 4. Assemble Fuzzer Block
    fuzzer_block = (
      f"{adapter_shim}\n\n"
      f"{fuzzer_code}\n\n"
      "# Alias matching template expectation\n"
      "class StandaloneFuzzer(InputFuzzer):\n"
      "    pass\n"
    )

    # 5. Build Dynamic Init Logic (Decoupling Step)
    imports_block, init_helpers_block, injection_logic_block = self._build_dynamic_init(target_fw)

    # 6. Populate Template
    script_content = HARNESS_TEMPLATE.format(
      source_path=source_file.resolve().as_posix(),
      target_path=target_file.resolve().as_posix(),
      source_fw=source_fw,
      target_fw=target_fw,
      hints_json=hints_json,
      fuzzer_implementation=fuzzer_block,
      # New Injection Points
      imports=imports_block,
      init_helpers=init_helpers_block,
      param_injection_logic=injection_logic_block,
    )

    # 7. Write to Disk
    output_harness.parent.mkdir(parents=True, exist_ok=True)
    with open(output_harness, "wt", encoding="utf-8") as f:
      f.write(script_content)

  def _build_dynamic_init(self, target_fw: str) -> tuple[str, str, str]:
    """
    Queries the target adapter to construct framework-specific initialization logic.

    Returns:
        (imports_str, helper_functions_str, injection_logic_str)
    """
    adapter = get_adapter(target_fw)

    # Defaults (Empty)
    if not adapter:
      return "", "", "pass"

    # A. Imports
    imports = getattr(adapter, "harness_imports", [])
    imports_str = "\n".join(imports)

    # B. Helper Code
    init_code = getattr(adapter, "get_harness_init_code", lambda: "")()

    # Extract function name from the code string (e.g. def _make_jax_key)
    match = re.search(r"def\s+([a-zA-Z0-9_]+)\s*\(", init_code)
    helper_name = match.group(1) if match else None

    # C. Injection Logic (The loop body)
    magic_args = getattr(adapter, "declared_magic_args", [])

    injection_lines = []

    if magic_args and helper_name:
      # Construct dispatch: if parameter matches any magic arg, call helper
      quoted_args = [f'"{a}"' for a in magic_args]
      list_str = "[" + ", ".join(quoted_args) + "]"

      # Indentation matches context in template logic
      injection_lines.append(f"val = None")
      injection_lines.append(f"if tp in {list_str}:")
      injection_lines.append(f"    val = {helper_name}(seed=42)")
      injection_lines.append(f"if val is not None:")
      injection_lines.append(f"    tgt_inputs[tp] = val")
    else:
      injection_lines.append("pass")

    final_logic = injection_lines[0]
    for line in injection_lines[1:]:
      final_logic += "\n                    " + line

    return imports_str, init_code, final_logic

  def _generate_adapter_shim(self) -> str:
    """
    Introspects registered frameworks to build the `get_adapter` function.
    """
    shim_lines = [
      "# Shim for missing ml_switcheroo.frameworks.get_adapter",
      "def get_adapter(framework):",
      "    class GenericAdapter:",
      "        def convert(self, data):",
      "            try:",
      "                import numpy as np",
      "                if not isinstance(data, (np.ndarray, np.generic)) and not isinstance(data, (list, tuple)):",
      "                    return data",
      "            except ImportError:",
      "                pass",
      "",
    ]

    frameworks = sorted(_ADAPTER_REGISTRY.keys())
    first = True
    for fw_name in frameworks:
      adapter_cls = _ADAPTER_REGISTRY[fw_name]

      # Skip if abstract/incomplete
      if not hasattr(adapter_cls, "convert"):
        continue

      try:
        method_source = inspect.getsource(adapter_cls.convert)
      except OSError:
        continue

      clean_block = textwrap.dedent(method_source)
      lines = clean_block.splitlines()
      body_start = 0
      for i, line in enumerate(lines):
        if line.strip().startswith("def convert"):
          body_start = i + 1
          break

      body_lines = lines[body_start:]
      if not body_lines:
        continue

      condition_kw = "if" if first else "elif"
      shim_lines.append(f"            {condition_kw} framework == '{fw_name}':")

      base_indent = " " * 16
      body_str = textwrap.dedent("\n".join(body_lines))
      indented_body = textwrap.indent(body_str, base_indent)
      shim_lines.append(indented_body)

      first = False

    shim_lines.append("")
    shim_lines.append("            return data")
    shim_lines.append("")
    shim_lines.append("    return GenericAdapter()")
    return "\n".join(shim_lines)
