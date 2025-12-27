"""
Core Engine of the Input Fuzzer.

Defines the `InputFuzzer` class which acts as the high-level API
for generating test inputs and adapting them to target frameworks.
"""

from typing import Any, Dict, List, Optional, Tuple

from ml_switcheroo.frameworks import get_adapter
from ml_switcheroo.testing.fuzzer.generators import get_random_shape
from ml_switcheroo.testing.fuzzer.heuristics import generate_by_heuristic
from ml_switcheroo.testing.fuzzer.parser import generate_from_hint


class InputFuzzer:
  """
  Generates dummy inputs (Arrays, Scalars, Containers) for equivalence testing.

  Orchestrates:
  1.  Resolution of Explicit Type Hints via Parser.
  2.  Resolution of Implicit Heuristics via Heuristics engine.
  3.  Framework adaptation via Adapters.
  """

  MAX_RECURSION_DEPTH = 3

  def __init__(self, seed_shape: Optional[Tuple[int, ...]] = None):
    """
    Initializes the Fuzzer.

    Args:
        seed_shape: If provided, heuristic array generation defaults to this shape.
    """
    self._seed_shape = seed_shape
    self.max_depth = self.MAX_RECURSION_DEPTH

  def generate_inputs(
    self,
    params: List[str],
    hints: Optional[Dict[str, str]] = None,
    constraints: Optional[Dict[str, Dict]] = None,
  ) -> Dict[str, Any]:
    """
    Creates a dictionary of `{arg_name: value}`.

    Resolves symbolic dimensions across the entire parameter set. For example,
    if hints are `{'x': "Array['N']", 'y': "Array['N']"}`, both arrays will
    have matching lengths.

    Args:
        params: List of argument names to generate (e.g. `['x', 'axis']`).
        hints: Dictionary of `{arg_name: type_string}` derived from Spec.
        constraints: Dictionary of `{arg_name: {min, max, options, dtype}}`.

    Returns:
        Dict[str, Any]: Randomized inputs ready for Framework adaptation.
    """
    kwargs: Dict[str, Any] = {}
    # Context to resolve symbolic dimensions like 'B', 'N' across arguments
    symbol_map: Dict[str, int] = {}

    # Decide on a consistent base shape for heuristics fallback
    base_shape = get_random_shape(self._seed_shape)
    hints = hints or {}
    constraints_map = constraints or {}

    for p in params:
      hint = hints.get(p)
      cons = constraints_map.get(p, {})

      # Strategy 1: Explicit Type Hint
      if hint and hint != "Any":
        try:
          val = generate_from_hint(
            type_str=hint,
            base_shape=base_shape,
            depth=0,
            max_depth=self.max_depth,
            symbol_map=symbol_map,
            constraints=cons,
          )
          kwargs[p] = val
          continue
        except Exception:
          # If parsing fails, fall back to heuristics...
          pass

      # Strategy 2: Heuristic Matching based on Name
      kwargs[p] = generate_by_heuristic(p, base_shape, constraints=cons)

    return kwargs

  def adapt_to_framework(self, kwargs: Dict[str, Any], framework: str) -> Dict[str, Any]:
    """
    Converts Numpy inputs to framework-specific tensor types.

    Delegates to registered adapters (e.g., `TorchAdapter`, `JaxAdapter`).

    Args:
        kwargs: Input dictionary with Numpy values.
        framework: Key of the framework (e.g., "torch", "jax").

    Returns:
        Dict with framework-specific tensors.
    """
    adapter = get_adapter(framework)

    # If no adapter found, return pure numpy/python objects (Pass-through)
    if not adapter:
      return kwargs

    converted = {}
    for k, v in kwargs.items():
      try:
        converted[k] = adapter.convert(v)
      except Exception:
        # If conversion logic fails, keep original
        converted[k] = v

    return converted
