"""
Execution Engine for Semantics Verification.

This module provides the `EquivalenceRunner`, which executes operations across
multiple frameworks (PyTorch, JAX, etc.) to verify behavioral equivalence.
It handles input generation, type adaptation, and crucially, **argument mapping**
to ensure that standard inputs are passed to frameworks using their specific
parameter names (e.g., mapping `axis` to `dim`).

Updates:

- Added support for `shape_calc` lambda verification (Feature 20).
- Uses central registry for output normalization.
"""

import importlib
import traceback
from typing import Dict, Any, Tuple, Optional, List

import numpy as np

from ml_switcheroo.testing.fuzzer import InputFuzzer
from ml_switcheroo.frameworks import get_adapter


class EquivalenceRunner:
  """
  Executes and compares operations across different Deep Learning frameworks.
  """

  def __init__(self) -> None:
    """Initializes the runner with a default fuzzer."""
    self.fuzzer = InputFuzzer()

  def verify(
    self,
    variants: Dict[str, Any],
    params: List[str],
    hints: Optional[Dict[str, str]] = None,
    constraints: Optional[Dict[str, Dict]] = None,
    shape_calc: Optional[str] = None,
  ) -> Tuple[bool, str]:
    """
    Runs the operation across all defined variants and compares results.

    This process includes:

    1. Fuzzing: Generates standard NumPy inputs based on `params`, `hints` and `constraints`.
    2. Renaming: Maps standard argument names to framework-specific names
       defined in the Semantics (e.g., `axis` -> `dim`).
    3. Adaptation: Converts NumPy arrays to framework Tensors.
    4. Execution: Runs the functions via dynamic import.
    5. Normalization: Converts results back to NumPy.
    6. **Shape Check**: Verifies result shape using `shape_calc` lambda if provided.
    7. Comparison: Asserts numeric closeness between frameworks.

    Args:
        variants: Dictionary of framework implementations from the Semantic
                  Knowledge Base.
        params: List of standard argument names (e.g. `['x', 'axis']`).
        hints: Dictionary of type hints (e.g. `{'axis': 'int'}`) to guide
               the fuzzer. Uses explicit types over heuristics.
        constraints: Dictionary of constraint maps (min, max, options) to bound fuzzer.
        shape_calc: Optional python lambda string (e.g. `lambda x: x.shape`) to verify output shape.

    Returns:
        Tuple[bool, str]: A pair containing:
            - bool: True if the verification passed (or skipped).
            - str: A human-readable status message or error log.
    """
    # 1. Generate Base Inputs (Standard Names, Typed via Hints)
    base_inputs = self.fuzzer.generate_inputs(params, hints=hints, constraints=constraints)

    results = {}

    # 2. Run for each framework
    for fw, details in variants.items():
      # Skip if specific API undefined (e.g. complex plugin logic without API string)
      # or if details is None (explicit disable)
      if not isinstance(details, dict) or "api" not in details:
        continue

      api_path = details["api"]
      arg_map = details.get("args", {})

      try:
        # 2a. Remap Argument Names (Standard -> Framework Specific)
        prepped_inputs = self._remap_args(base_inputs, arg_map)

        # 2b. Adapt Types (Numpy -> Tensor)
        fw_inputs = self.fuzzer.adapt_to_framework(prepped_inputs, fw)

        # 2c. Execute
        res = self._execute_api(api_path, fw_inputs)

        # 2d. Normalize Result (Tensor -> Numpy) via central Adapter Registry
        numpy_adapter = get_adapter("numpy")
        if numpy_adapter:
          results[fw] = numpy_adapter.convert(res)
        else:
          # Fallback unlikely if adapters loaded, but safe
          results[fw] = res

      except Exception:
        # Capture traceback for detailed error reporting
        err_msg = traceback.format_exc()
        # We return False immediately on crash to highlight the error
        return False, f"Crash in {fw} ({api_path}):\n{err_msg}"

    if not results:
      return True, "Skipped (No executable variants found)"

    # 3. Shape Verification (Feature 20)
    if shape_calc:
      try:
        # order args for positional lambda arguments if names match input list ordering
        # Assuming lambda signature accepts args in same order as 'params' keys
        # We pass arguments as *args to the lambda
        input_args_ordered = [base_inputs[p] for p in params]

        calc_fn = eval(shape_calc)
        expected_shape = calc_fn(*input_args_ordered)

        for fw, val in results.items():
          if hasattr(val, "shape"):
            if val.shape != expected_shape:
              return False, f"Shape Mismatch in {fw}: {val.shape} != {expected_shape}"
      except Exception as e:
        return False, f"Shape Calculation Error: {e}"

    # 4. Compare Results
    # We need at least 2 successful runs to perform a comparison
    if len(results) < 2:
      return True, "Skipped (Need 2+ frameworks to compare)"

    keys = list(results.keys())
    ref_fw = keys[0]
    ref_val = results[ref_fw]

    for other_fw in keys[1:]:
      val = results[other_fw]
      # Use loose tolerance for float32 differences across backends
      if not self._deep_compare(ref_val, val):
        # Try to provide helpful debug info
        msg = f"Mismatch between {ref_fw} and {other_fw}.\nRef ({ref_fw}): {ref_val}\nCand ({other_fw}): {val}"
        return False, msg

    return True, "✅ Output Matched"

  def _remap_args(self, inputs: Dict[str, Any], mapping: Dict[str, str]) -> Dict[str, Any]:
    """
    Translates input keys from Standard Spec names to Framework Specific names.

    Args:
        inputs: Dictionary of {std_name: value}.
        mapping: Dictionary of {std_name: fw_name} from Semantics.

    Returns:
        New dictionary with renamed keys. Unmapped keys are preserved.
    """
    new_inputs = {}
    for key, val in inputs.items():
      # Look up replacement name, default to original if not found
      # mapping key is Standard Name, value is Framework Name
      new_key = mapping.get(key, key)
      new_inputs[new_key] = val
    return new_inputs

  def _execute_api(self, api_path: str, kwargs: Dict[str, Any]) -> Any:
    """
    Dynamically imports and calls the function with provided keyword arguments.

    Args:
        api_path: Fully qualified path (e.g., 'torch.nn.functional.relu').
        kwargs: Dictionary of arguments ready for this specific function.

    Returns:
        The return value of the function call.

    Raises:
        ImportError: If module cannot be loaded.
        AttributeError: If function not found in module.
        TypeError: If arguments do not match signature.
    """
    if "." not in api_path:
      raise ImportError(f"Invalid API path format: {api_path}")

    module_name, func_name = api_path.rsplit(".", 1)

    # Import module dynamically (e.g. import torch.nn.functional)
    mod = importlib.import_module(module_name)

    # Support nested attributes if needed (though usually flat)
    # but standard split is robust for most cases.
    func = getattr(mod, func_name)

    return func(**kwargs)

  def _deep_compare(self, a: Any, b: Any) -> bool:
    """
    Recursively compares two objects (Arrays, Lists, Scalars).

    Handles:
    - Nested Sequences (List/Tuple)
    - Dictionaries
    - NumPy Arrays (with tolerance)
    - Scalars (int/float/bool)

    Args:
        a: Reference object.
        b: Candidate object.

    Returns:
        bool: True if they are effectively equal.
    """
    # 1. Type Mismatch Check (Loose)
    # We allow list vs tuple comparison if contents match
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
      if len(a) != len(b):
        return False
      return all(self._deep_compare(x, y) for x, y in zip(a, b))

    # 2. Dictionary Check
    if isinstance(a, dict) and isinstance(b, dict):
      if set(a.keys()) != set(b.keys()):
        return False
      return all(self._deep_compare(a[k], b[k]) for k in a)

    # 3. Handle scalars and arrays uniformly via numpy
    # Using loose instance check to catch numpy scalars vs python scalars
    if isinstance(a, (float, int, np.ndarray, np.generic)) and isinstance(b, (float, int, np.ndarray, np.generic)):
      try:
        # asanyarray handles python scalars by wrapping them (0-rank array)
        arr_a = np.asanyarray(a)
        arr_b = np.asanyarray(b)

        # Shape check
        if arr_a.shape != arr_b.shape:
          return False

        # Handle strings or object arrays which allclose doesn't like
        if arr_a.dtype.kind in ["U", "S", "O"]:
          return np.array_equal(arr_a, arr_b)

        # Check for NaNs - if locations match, we consider it equal
        # (isnan throws error on strings/objects hence the check above)
        nan_mask_a = np.isnan(arr_a)
        nan_mask_b = np.isnan(arr_b)
        if np.any(nan_mask_a != nan_mask_b):
          return False

        # Compare finite values
        # rtol=1e-3, atol=1e-4 is standard for ML cross-framework testing (float32 precision)
        return np.allclose(arr_a, arr_b, rtol=1e-3, atol=1e-4, equal_nan=True)
      except Exception:
        # Fallback for shape mismatch or uncomparable types caught late
        return False

    # 4. Fallback for strings, booleans, None, or other objects
    return a == b
