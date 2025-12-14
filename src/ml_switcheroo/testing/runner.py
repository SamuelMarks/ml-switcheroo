"""
Execution Engine for Semantics Verification.

This module provides the `EquivalenceRunner`, which executes operations across
multiple frameworks (PyTorch, JAX, etc.) to verify behavioral equivalence.
It handles input generation, type adaptation, and crucially, **argument mapping**
to ensure that standard inputs are passed to frameworks using their specific
parameter names (e.g., mapping `axis` to `dim`).

Refactor Note:
    This module utilizes the central registry in `ml_switcheroo.frameworks` to handle
    output normalization (Framework -> NumPy) by requesting a "numpy" adapter.
    This provides a unified conversion interface and eliminates redundant normalization logic.
"""

import importlib
from typing import Dict, Any, Tuple, Optional, List

import numpy as np

from ml_switcheroo.testing.fuzzer import InputFuzzer
from ml_switcheroo.frameworks import get_adapter


class EquivalenceRunner:
  """
  Executes and compares operations across different Deep Learning frameworks.

  Attributes:
      fuzzer (InputFuzzer): The engine used to generate randomized inputs.
  """

  def __init__(self) -> None:
    """Initializes the runner with a default fuzzer."""
    self.fuzzer = InputFuzzer()

  def verify(
    self,
    variants: Dict[str, Any],
    params: List[str],
    hints: Optional[Dict[str, str]] = None,
  ) -> Tuple[bool, str]:
    """
    Runs the operation across all defined variants and compares results.

    This process includes:
    1. Fuzzing: Generates standard NumPy inputs based on `params` and `hints`.
    2. Renaming: Maps standard argument names to framework-specific names
       defined in the Semantics (e.g., `axis` -> `dim`).
    3. Adaptation: Converts NumPy arrays to framework Tensors.
    4. Execution: Runs the functions via dynamic import.
    5. Comparison: Converts results back to NumPy and asserts closeness.

    Args:
        variants: Dictionary of framework implementations from the Semantic
                  Knowledge Base.
                  Structure: `{"torch": {"api": "...", "args": {...}}, ...}`
        params: List of standard argument names (e.g. `['x', 'axis']`).
        hints: Dictionary of type hints (e.g. `{'axis': 'int'}`) to guide
               the fuzzer. Uses explicit types over heuristics.

    Returns:
        Tuple[bool, str]: A pair containing:
            - bool: True if the verification passed (or skipped).
            - str: A human-readable status message or error log.
    """
    # 1. Generate Base Inputs (Standard Names, Typed via Hints)
    base_inputs = self.fuzzer.generate_inputs(params, hints=hints)

    results = {}

    # 2. Run for each framework
    for fw, details in variants.items():
      # Skip if specific API undefined (e.g. complex plugin logic without API)
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

      except Exception as e:
        # We return False immediately on crash to highlight the error
        return False, f"Crash in {fw} ({api_path}): {e}"

    # 3. Compare Results
    # We need at least 2 successful runs to compare logic
    if len(results) < 2:
      return True, "Skipped (Need 2+ frameworks to compare)"

    keys = list(results.keys())
    ref_fw = keys[0]
    ref_val = results[ref_fw]

    for other_fw in keys[1:]:
      val = results[other_fw]
      # Use loose tolerance for float32 differences across backends
      if not self._deep_compare(ref_val, val):
        return False, f"Mismatch between {ref_fw} and {other_fw}"

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
    func = getattr(mod, func_name)

    return func(**kwargs)

  def _deep_compare(self, a: Any, b: Any) -> bool:
    """
    Recursively compares two objects (Arrays, Lists, Scalars).

    Args:
        a: Reference object.
        b: Candidate object.

    Returns:
        bool: True if they are effectively equal.
    """
    # Check types match roughly first (sequences)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
      if len(a) != len(b):
        return False
      return all(self._deep_compare(x, y) for x, y in zip(a, b))

    # Handle scalars and arrays uniformly via numpy
    # Using loose instance check to catch numpy scalars vs python scalars
    if isinstance(a, (float, int, np.ndarray, np.generic)) and isinstance(b, (float, int, np.ndarray, np.generic)):
      try:
        # asanyarray handles python scalars by wrapping them
        arr_a = np.asanyarray(a)
        arr_b = np.asanyarray(b)

        # Handle strings or object arrays which allclose doesn't like
        if arr_a.dtype.kind in ["U", "S", "O"]:
          return np.array_equal(arr_a, arr_b)

        return np.allclose(arr_a, arr_b, rtol=1e-3, atol=1e-4)
      except Exception:
        # Fallback for shape mismatch or uncomparable types
        return False

    # Fallback for strings, booleans, or other objects
    return a == b
