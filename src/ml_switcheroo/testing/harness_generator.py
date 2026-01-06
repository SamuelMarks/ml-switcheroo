"""
Integration Test Harness Generator.

Generates standalone verification scripts.
Bundles fuzzer logic (including Hypothesis strategies).
"""

import json
import inspect
import textwrap
import re
from pathlib import Path
from typing import Dict, Any, Optional, List

from ml_switcheroo.testing.harness_generator_template import HARNESS_TEMPLATE
from ml_switcheroo.testing.fuzzer.core import InputFuzzer
from ml_switcheroo.utils.code_extractor import CodeExtractor
from ml_switcheroo.frameworks.base import _ADAPTER_REGISTRY, get_adapter

# Imports needed for bundling the fuzzer logic
import ml_switcheroo.testing.fuzzer.generators
import ml_switcheroo.testing.fuzzer.parser
import ml_switcheroo.testing.fuzzer.heuristics
import ml_switcheroo.testing.fuzzer.utils
import ml_switcheroo.testing.fuzzer.strategies


class HarnessGenerator:
  """
  Generates standalone verification scripts tailored to the target framework.
  """

  def __init__(self) -> None:
    """
    Initializes the generator instance and its code extractor utility.
    """
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
    """Creates the verification harness file."""
    fuzzer_code = self._bundle_fuzzer_dependencies()

    hints_map = {}
    if semantics:
      for op_name, details in semantics.items():
        args_data = details.get("std_args", [])
        func_hints = {}
        for arg in args_data:
          if isinstance(arg, (list, tuple)) and len(arg) == 2:
            func_hints[arg[0]] = arg[1]
          elif isinstance(arg, dict):
            # Support rich ODL parameter definitions
            name = arg.get("name")
            typ = arg.get("type")
            if name and typ:
              func_hints[name] = typ
        if func_hints:
          hints_map[op_name] = func_hints

    hints_json = json.dumps(hints_map).replace("'", '"')
    adapter_shim = self._generate_adapter_shim()

    fuzzer_block = f"{adapter_shim}\n\n{fuzzer_code}\n\nclass StandaloneFuzzer(InputFuzzer):\n    pass\n"

    imports_block, init_helpers_block, injection_logic_block = self._build_dynamic_init(target_fw)
    to_numpy_block = self._build_result_normalization(source_fw, target_fw)

    script_content = HARNESS_TEMPLATE.format(
      source_path=source_file.resolve().as_posix(),
      target_path=target_file.resolve().as_posix(),
      source_fw=source_fw,
      target_fw=target_fw,
      hints_json=hints_json,
      fuzzer_implementation=fuzzer_block,
      imports=imports_block,
      init_helpers=init_helpers_block,
      param_injection_logic=injection_logic_block,
      to_numpy_logic=to_numpy_block,
    )

    output_harness.parent.mkdir(parents=True, exist_ok=True)
    with open(output_harness, "wt", encoding="utf-8") as f:
      f.write(script_content)

  def _bundle_fuzzer_dependencies(self) -> str:
    """
    Extracts all helper functions required by InputFuzzer.
    Injects Hypothesis and typing imports globally for the bundle.
    """
    deps = []

    # Global imports required by the extracted code
    # We must ensure all imports used by strategies.py and core.py types/logic are present
    deps.append("import hypothesis.strategies as st")
    deps.append("import hypothesis.extra.numpy as npst")
    deps.append("import re")
    deps.append("import numpy as np")
    deps.append("from typing import Any, Dict, List, Optional, Tuple, Callable")

    def extract_module_functions(module):
      funcs = inspect.getmembers(module, inspect.isfunction)
      for name, func in funcs:
        if func.__module__ == module.__name__:
          try:
            source = inspect.getsource(func)
            deps.append(textwrap.dedent(source))
          except OSError:
            pass

    # Order matters slightly for resolution order of helpers
    extract_module_functions(ml_switcheroo.testing.fuzzer.utils)
    extract_module_functions(ml_switcheroo.testing.fuzzer.strategies)
    extract_module_functions(ml_switcheroo.testing.fuzzer.generators)
    extract_module_functions(ml_switcheroo.testing.fuzzer.heuristics)
    extract_module_functions(ml_switcheroo.testing.fuzzer.parser)

    fuzzer_class = self.extractor.extract_class(InputFuzzer)

    return "\n\n".join(deps + [fuzzer_class])

  def _build_dynamic_init(self, target_fw: str) -> tuple[str, str, str]:
    adapter = get_adapter(target_fw)
    if not adapter:
      return "", "", "pass"

    imports = getattr(adapter, "harness_imports", [])
    imports_str = "\n".join(imports)
    init_code = getattr(adapter, "get_harness_init_code", lambda: "")()
    match = re.search(r"def\s+([a-zA-Z0-9_]+)\s*\(", init_code)
    helper_name = match.group(1) if match else None
    magic_args = getattr(adapter, "declared_magic_args", [])

    injection_lines = []
    if magic_args and helper_name:
      quoted_args = [f'"{a}"' for a in magic_args]
      list_str = "[" + ", ".join(quoted_args) + "]"
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

  def _build_result_normalization(self, source_fw: str, target_fw: str) -> str:
    blocks = []
    unique_fws = set([source_fw, target_fw])
    if "flax_nnx" in unique_fws:
      unique_fws.add("jax")

    for fw in unique_fws:
      adapter = get_adapter(fw)
      code = None
      if adapter and hasattr(adapter, "get_to_numpy_code"):
        try:
          code = adapter.get_to_numpy_code()
        except Exception:
          pass
      if code:
        indented = textwrap.indent(code, "    ")
        blocks.append(f"# Framework: {fw}\n{indented}")

    return "\n    ".join(blocks)

  def _generate_adapter_shim(self) -> str:
    shim_lines = [
      "# Shim for missing get_adapter",
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
