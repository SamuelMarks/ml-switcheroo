"""
Discovery Tool for linking Framework implementations to Standards.

This module scans installed libraries (e.g., Torch, JAX, TensorFlow, MLX) for
functions that match the names defined in the Semantic Knowledge Base.

Links:
- Hub (Spec): "Abs(x)"
- Spoke (Lib): "torch.abs"

Logic:
1. Module Discovery via FrameworkAdapters.
2. Name Matching (`torch.abs` matches `Abs`).
3. Signature Compatibility Verification. (Robust against C-Exts).
"""

import importlib
import inspect
from typing import Dict, Any, List, Tuple, Union, Optional

from ml_switcheroo.utils.console import console, log_info, log_success, log_warning
from ml_switcheroo.frameworks import get_adapter


class FrameworkSyncer:
  """
  Links abstract operations to concrete framework implementations.

  Attributes:
      console: A configured Rich console for logging output.
  """

  def __init__(self):
    """Initializes the FrameworkSyncer."""
    self.console = console

  def sync(self, tier_data: Dict[str, Any], framework: str) -> None:
    """
    Updates the 'variants' dict in tier_data by hunting for ops in the target framework.
    """
    log_info(f"Syncing [code]{framework}[/code] against Standard...")

    # 1. Resolve Search Paths via Registry
    adapter = get_adapter(framework)
    paths_to_search = []

    if adapter and hasattr(adapter, "search_modules"):
      paths_to_search = adapter.search_modules
    else:
      paths_to_search = [framework]

    # 2. Pre-load modules
    libs = []
    for mod_name in paths_to_search:
      try:
        libs.append(importlib.import_module(mod_name))
      except ImportError:
        pass

    if not libs:
      self.console.print(f"⚠️  Could not import any modules for {framework}. Is it installed?")
      return

    # 3. Discovery Loop
    count = 0
    skipped = 0

    for op_name, details in tier_data.items():
      # If we already have a manual mapping, skip to preserve overrides
      if framework in details.get("variants", {}):
        continue

      std_args_raw = details.get("std_args", [])
      std_arg_names = self._extract_names(std_args_raw)

      found_path = None

      # Iterate modules looking for match
      for lib in libs:
        # Case-insensitive name match loop
        candidate_name = None
        if hasattr(lib, op_name):
          candidate_name = op_name
        else:
          # Fallback: Scan dir() for case-insensitive match (slower but thorough)
          # We optimize by only doing this if direct access fails
          for member_name in dir(lib):
            if member_name.lower() == op_name.lower():
              candidate_name = member_name
              break

        if candidate_name:
          obj = getattr(lib, candidate_name)

          # Must be callable (Function or Class)
          if callable(obj):
            # Verify Signature Compatibility
            if self._is_compatible(obj, std_arg_names):
              found_path = f"{lib.__name__}.{candidate_name}"
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
    """Unpacks argument definitions into a flat list of names."""
    out = []
    for item in args:
      if isinstance(item, (list, tuple)):
        out.append(item[0])  # Name is index 0
      else:
        out.append(item)
    return out

  def _is_compatible(self, obj: Any, std_args: List[str]) -> bool:
    """
    Verifies if the candidate signature can accept the standard arguments.
    Robustly handles C-Extensions by assuming compatibility if inspection fails.
    """
    target_func = obj
    is_class_obj = inspect.isclass(obj)
    found_method = False

    # Handle Classes: Inspect the inference method, not the constructor
    if is_class_obj:
      # Priority: __call__ (JAX/Pax), forward (Torch), call (Keras)
      for method_name in ["__call__", "forward", "call"]:
        if hasattr(obj, method_name):
          target_func = getattr(obj, method_name)
          found_method = True
          break

    try:
      sig = inspect.signature(target_func)
    except (ValueError, TypeError):
      # Built-ins (C-extensions) or some ufuncs might not support signature inspection.
      # FAIL OPEN: Assume compatibility rather than rejecting valid ops like torch.abs
      return True

    params = list(sig.parameters.values())

    # Aggressive self stripping: If we found a method on a class, assume arg 0 is self
    if is_class_obj and found_method and params:
      # Check if bound/unbound. getattr(Class, 'func') is unbound in Py3.
      # usually first param is self.
      if params[0].name in ["self", "cls"]:
        params = params[1:]

    # 1. Check for var_positional (*args)
    # If it accepts *args, it can accept anything. Compatible.
    has_var_args = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)
    if has_var_args:
      return True

    # Filter for parameters that can accept positional input
    pos_params = [
      p for p in params if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]

    # 2. Check Mandatory count
    mandatory_params = [p for p in pos_params if p.default == inspect.Parameter.empty]

    # If function requires MORE mandatory arguments than Spec provides, we can't call it.
    # e.g. Spec: Abs(x) -> Func: Foo(a, b) [needs 2] -> Fail
    if len(mandatory_params) > len(std_args):
      return False

    # 3. Check Capacity
    # If class, we assume extras handled in init, so we skip exact capacity check
    # If function, we check if it can accept the args.
    if not is_class_obj:
      # e.g. Spec: Add(a, b) -> Func: Neg(x) [accepts 1] -> Fail
      if len(pos_params) < len(std_args):
        return False

    return True
