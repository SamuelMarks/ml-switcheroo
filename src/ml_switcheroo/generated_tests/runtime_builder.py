"""Runtime Module Logic Generator.

This module contains the logic to creating the `runtime.py` file required by
generated tests. It injects shared helpers, cross-framework comparison logic,
and determinism fixtures.
"""

import ast
import inspect
import pathlib
from typing import Any, List, Optional

from ml_switcheroo.generated_tests import runtime as _runtime_mod
from ml_switcheroo.generated_tests.templates import get_template


def get_required_packages(imp_str: str) -> List[str]:
  """Parse an import string and return a list of required top-level packages.

  Args:
      imp_str: The import string to parse.

  Returns:
      A list of top-level package names.

  """
  try:
    tree = ast.parse(imp_str)
  except SyntaxError:
    return []

  packages = []
  for node in tree.body:
    if isinstance(node, ast.Import):
      for alias in node.names:
        packages.append(alias.name.split(".")[0])
    elif isinstance(node, ast.ImportFrom):
      if node.module:
        packages.append(node.module.split(".")[0])

  return list(dict.fromkeys(packages))


# The static part of the runtime helper that does not depend on available frameworks.
_SHARED_RUNTIME_LOGIC = (
  "\n# --- Determinism ---\n"
  + inspect.getsource(_runtime_mod.ensure_determinism)
  + "\n# --- Verification Logic ---\n"
  + inspect.getsource(_runtime_mod.verify_results).replace("def verify_results(\n  ref:", "def verify_results(ref:")
)


def ensure_runtime_module(
  out_dir: pathlib.Path,
  frameworks: Optional[List[str]] = None,
  mgr: Any = None,
) -> None:
  """Create or updates the `runtime.py` module in the output directory.

  Injects:
  1. Safe import blocks for all used frameworks (try/except ImportError).
  2. Determinism fixtures.
  3. Recursive result comparison logic (`verify_results`).

  Args:
      out_dir: Directory where the generated tests and runtime.py reside.
      frameworks: List of frameworks to include imports for.
      mgr: SemanticsManager for retrieving custom import templates.

  """
  out_dir.mkdir(parents=True, exist_ok=True)
  runtime_path = out_dir / "runtime.py"

  imports_block = []
  alls = ["verify_results"]

  # Always include torch/jax logic if they appear in defaults or requested
  fw_set = set(frameworks) if frameworks else set()
  chk_fws = fw_set | {"torch", "jax"}

  # Sort for determinism in file output
  for fw in sorted(chk_fws):
    # We need the template dict to know what to import
    tmpl = get_template(mgr, fw)
    if not tmpl:
      continue

    # Get imports string from template, defaulting to simple import
    imp = tmpl.get("import", f"import {fw}")

    # Sanitization for checking availability
    # e.g., TORCH_AVAILABLE = True
    var_name = f"{fw.upper()}_AVAILABLE"
    imports_block.append(f"# --- {fw} ---")
    req_pkgs = get_required_packages(imp)
    if not req_pkgs:
      imports_block.append(f"{var_name} = True")
    else:
      checks = " and ".join(f'importlib.util.find_spec("{p}") is not None' for p in req_pkgs)
      imports_block.append(f"{var_name} = {checks}")
    imports_block.append("")

    export_symbols_map = {
      "tensorflow": (var_name, "tf"),
      "jax": ("jnp", var_name, fw),
    }
    alls.extend(export_symbols_map.get(fw, (var_name, fw)))

  imports_str = "\n".join(imports_block)

  # Combine parts
  code = '"""Shared runtime flags for generated tests (Auto-Generated)."""\n# ruff: noqa\n# fmt: off\n\n'
  code += "import sys\nimport pytest\nimport random\nimport numpy as np\nimport importlib.util\nimport typing\nfrom typing import Any\n\n"
  code += imports_str + "\n"
  code += _SHARED_RUNTIME_LOGIC

  runtime_path.write_text(code, encoding="utf-8")
