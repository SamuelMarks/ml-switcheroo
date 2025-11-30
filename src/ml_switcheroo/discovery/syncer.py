"""
Discovery Tool for linking Framework implementations to Standards.

This module scans installed libraries (e.g., Torch, JAX, TensorFlow, MLX) for
functions that match the names defined in the Semantic Knowledge Base.
It performs signature verification to ensure the found function is compatible
with the abstract specification.
"""

import importlib
import inspect
from typing import Dict, Any, List, Tuple, Union
from ml_switcheroo.utils.console import console, log_info, log_success

# Expanded priority search paths for Array API compliance across supported engines.
SEARCH_PATHS = {
  "torch": ["torch", "torch.linalg", "torch.special", "torch.fft"],
  "jax": ["jax.numpy", "jax.numpy.linalg", "jax.numpy.fft"],
  "tensorflow": [
    "tensorflow",
    "tensorflow.math",
    "tensorflow.linalg",
    "tensorflow.signal",
  ],
  "mlx": ["mlx.core", "mlx.nn", "mlx.core.fft", "mlx.core.linalg"],
  "numpy": ["numpy", "numpy.linalg", "numpy.fft"],
}


class FrameworkSyncer:
  """
  Links abstract operations to concrete framework implementations.

  This class iterates through known modules of a target framework, looking for
  callables that match the operation names defined in the Semantic Knowledge Base.
  It verifies that potential matches have compatible signatures before linking them.

  Attributes:
      console: A configured Rich console for logging output.
  """

  def __init__(self):
    """Initializes the FrameworkSyncer."""
    self.console = console

  def sync(self, tier_data: Dict[str, Any], framework: str) -> None:
    """
    Updates the 'variants' dict in tier_data by hunting for ops in the target framework.

    It attempts to import modules defined in `SEARCH_PATHS` for the given framework.
    If a function with a matching name and compatible signature is found, it is
    recorded in the `tier_data` dictionary under the `variants` key.

    Args:
        tier_data: The Semantic Knowledge Base (dictionary of operations).
            This dictionary is modified in-place.
        framework: The target framework name (e.g., 'torch', 'tensorflow', 'mlx').
    """
    log_info(f"Syncing [code]{framework}[/code] against Array API Standard...")

    # Pre-load modules based on the lookup table
    libs = []
    # Fallback to just the framework name if no specific paths defined
    paths_to_search = SEARCH_PATHS.get(framework, [framework])

    for mod_name in paths_to_search:
      try:
        libs.append(importlib.import_module(mod_name))
      except ImportError:
        # This is expected if the user doesn't have the full stack installed,
        # or if the submodule doesn't exist in that specific version.
        pass

    if not libs:
      self.console.print(f"⚠️  Could not import any modules for {framework}. Is it installed?")
      return

    count = 0
    skipped = 0

    for op_name, details in tier_data.items():
      # If we already have a manual mapping, skip to preserve overrides
      if framework in details.get("variants", {}):
        continue

      # std_args is List[Tuple[str, str]] or List[str] (legacy compat)
      std_args_raw = details.get("std_args", [])
      std_arg_names = self._extract_names(std_args_raw)

      found_path = None

      # 1. Search modules for matching name
      # We iterate in order of SEARCH_PATHS, so earlier modules take precedence.
      for lib in libs:
        if hasattr(lib, op_name):
          obj = getattr(lib, op_name)

          if callable(obj):
            # 2. Verify Signature Compatibility
            if self._is_compatible(obj, std_arg_names):
              found_path = f"{lib.__name__}.{op_name}"
              break
            else:
              skipped += 1

      if found_path:
        if "variants" not in details:
          details["variants"] = {}

        details["variants"][framework] = {"api": found_path}
        count += 1

    log_success(f"Linked {count} operations for {framework} (Skipped {skipped} mismatches).")

  def _extract_names(self, args: List[Union[str, Tuple[str, str]]]) -> List[str]:
    """
    Unpacks argument definitions into a flat list of names.

    Args:
        args: A list of arguments. Elements can be strings (legacy) or
              tuples of (name, type_annotation).

    Returns:
        List[str]: A list of argument names.
    """
    out = []
    for item in args:
      if isinstance(item, (list, tuple)):
        out.append(item[0])  # Name is index 0
      else:
        out.append(item)
    return out

  def _is_compatible(self, func: Any, std_args: List[str]) -> bool:
    """
    Verifies if the candidate function signature can accept the standard arguments.

    It checks:
    1. If the function accepts variable positional arguments (*args) -> Pass.
    2. If the function has too many mandatory arguments -> Fail.
    3. If the function cannot accept enough positional arguments -> Fail.

    Args:
        func: The Python function object to inspect.
        std_args: List of argument names from the spec (e.g. ['x', 'axis']).

    Returns:
        bool: True if compatible, False otherwise.
    """
    try:
      sig = inspect.signature(func)
    except (ValueError, TypeError):
      # Built-ins (C-extensions) or some ufuncs might not support signature inspection.
      # We default to True (Fail-Open) to avoid finding valid C-funcs incompatible.
      return True

    params = list(sig.parameters.values())

    # 1. Check for var_positional (*args)
    has_var_args = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)
    if has_var_args:
      return True

    # Filter for parameters that can accept positional input
    pos_params = [
      p for p in params if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]

    # 2. Check Mandatory count
    # A parameter is mandatory if it has no default value
    mandatory_params = [p for p in pos_params if p.default == inspect.Parameter.empty]

    if len(mandatory_params) > len(std_args):
      # Candidate requires more args than Spec provides
      return False

    # 3. Check Capacity
    # Candidate function must be able to accept at least as many positional args as Spec
    if len(pos_params) < len(std_args):
      return False

    return True
