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

# Imports needed for bundling the fuzzer logic
import ml_switcheroo.testing.fuzzer.generators
import ml_switcheroo.testing.fuzzer.parser
import ml_switcheroo.testing.fuzzer.heuristics
import ml_switcheroo.testing.fuzzer.utils


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
    """
    Writes the standalone verification script to disk.

    Args:
        source_file: Path to original source.
        target_file: Path to transpiled result.
        output_harness: Destination path for the generated script.
        source_fw: Key of source framework (e.g. 'torch').
        target_fw: Key of target framework (e.g. 'jax').
        semantics: Optional dictionary of semantic definitions (for type hints).
    """
    # 1. Extract Fuzzer Logic & Dependencies
    fuzzer_code = self._bundle_fuzzer_dependencies()

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

    # 6. Build Dynamic to_numpy() Logic (Result Normalization)
    # We only need normalization logic for the source and target frameworks involved.
    to_numpy_block = self._build_result_normalization(source_fw, target_fw)

    # 7. Populate Template
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

    # 8. Write to Disk
    output_harness.parent.mkdir(parents=True, exist_ok=True)
    with open(output_harness, "wt", encoding="utf-8") as f:
      f.write(script_content)

  def _bundle_fuzzer_dependencies(self) -> str:
    """
    Extracts all helper functions required by InputFuzzer to allow it to run
    standalone without importing ml_switcheroo modules.

    It iterates through various fuzzer submodules (generators, parser, heuristics),
    extracts their source code via inspection, and bundles them into a single string
    to be injected into the harness template.

    Returns:
        str: A large block of Python source code containing all dependencies.
    """
    deps = []

    # Helper to extract all functions from a module
    def extract_module_functions(module):
      funcs = inspect.getmembers(module, inspect.isfunction)
      for name, func in funcs:
        # Only extract functions actually defined in this module
        if func.__module__ == module.__name__:
          try:
            source = inspect.getsource(func)
            deps.append(textwrap.dedent(source))
          except OSError:
            pass

    # Extract Utils first (deps)
    extract_module_functions(ml_switcheroo.testing.fuzzer.utils)
    # Extract Generators
    extract_module_functions(ml_switcheroo.testing.fuzzer.generators)
    # Extract Heuristics
    extract_module_functions(ml_switcheroo.testing.fuzzer.heuristics)
    # Extract Parser
    extract_module_functions(ml_switcheroo.testing.fuzzer.parser)

    # Extract Class
    fuzzer_class = self.extractor.extract_class(InputFuzzer)

    return "\n\n".join(deps + [fuzzer_class])

  def _build_dynamic_init(self, target_fw: str) -> tuple[str, str, str]:
    """
    Queries the target adapter to construct framework-specific initialization logic.

    This includes imports (e.g. `import jax`), init helpers (e.g. PRNG key creation),
    and parameter injection logic (e.g. checking for 'rngs' argument).

    Args:
        target_fw: The framework key to generate initialization code for.

    Returns:
        tuple[str, str, str]: A tuple containing:
            - imports_str: Framework imports.
            - init_code: Helper function definitions.
            - final_logic: The injection logic block for the loop body.
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

  def _build_result_normalization(self, source_fw: str, target_fw: str) -> str:
    """
    Constructs the `to_numpy` logic by aggregating snippets from adapters.

    This ensures that the result verification can convert tensors from any registered
    framework back to NumPy for numeric comparison.

    Args:
        source_fw: The source framework key.
        target_fw: The target framework key.

    Returns:
        str: A string body of code to be injected into `to_numpy`.
    """
    blocks = []

    # Legacy Fallbacks (ensure we don't break existing FWs before they implement new protocol)
    legacy_defaults = {
      "torch": "if hasattr(obj, 'detach'): return obj.detach().cpu().numpy()",
      "jax": "if hasattr(obj, '__array__'): return np.array(obj)",
      "flax": "if hasattr(obj, '__array__'): return np.array(obj)",
      "flax_nnx": "if hasattr(obj, '__array__'): return np.array(obj)",
      "tensorflow": "if hasattr(obj, 'numpy'): return obj.numpy()",
      "keras": "if hasattr(obj, 'numpy'): return obj.numpy()",
      "mlx": "if hasattr(obj, 'tolist'): return np.array(obj.tolist())",
      # MLX arrays have tolist() but not always __array__ depending on version
    }

    # We collect configs for both Source and Target to ensure full coverage
    # (e.g. if we verify Torch -> JAX, we need to convert both Torch Output and JAX output)
    unique_fws = set([source_fw, target_fw])

    # Expand flavours
    if "flax_nnx" in unique_fws:
      unique_fws.add("jax")

    for fw in unique_fws:
      adapter = get_adapter(fw)
      code = None

      # 1. Try new protocol method
      if adapter and hasattr(adapter, "get_to_numpy_code"):
        try:
          code = adapter.get_to_numpy_code()
        except Exception:
          pass

      # 2. Fallback to legacy map
      if not code and fw in legacy_defaults:
        code = legacy_defaults[fw]

      if code:
        # Indent consistency for injection
        indented = textwrap.indent(code, "    ")
        blocks.append(f"# Framework: {fw}\n{indented}")

    return "\n    ".join(blocks)

  def _generate_adapter_shim(self) -> str:
    """
    Introspects registered frameworks to build the `get_adapter` function shim.

    This shim allows the generated script to perform basic type conversion (e.g.
    Torch Tensor -> Numpy) without importing the actual `ml_switcheroo` package
    or the heavy framework libraries unless they are actually present.

    Returns:
        str: Source code for the `get_adapter` function to be embedded in the harness.
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
