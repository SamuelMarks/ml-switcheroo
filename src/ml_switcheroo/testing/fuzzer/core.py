"""
Core Engine of the Input Fuzzer (Hypothesis Integration).

This module provides the `InputFuzzer` facade which now delegates generation logic
to Hypothesis Strategies. It maintains backward compatibility for casual usage via `generate_inputs`.
"""

from typing import Any, Dict, List, Optional
import hypothesis.strategies as st
from ml_switcheroo.frameworks import get_adapter
from ml_switcheroo.testing.fuzzer.strategies import strategies_from_spec
from ml_switcheroo.testing.fuzzer.heuristics import guess_dtype_by_name


class InputFuzzer:
  """
  Facade for creating Hypothesis strategies based on Semantic Spec.
  """

  # Legacy constant for compatibility with extraction tests
  MAX_RECURSION_DEPTH = 3

  def build_strategies(
    self,
    params: List[str],
    hints: Optional[Dict[str, str]] = None,
    constraints: Optional[Dict[str, Dict]] = None,
  ) -> Dict[str, st.SearchStrategy]:
    """
    Constructs a dictionary of Hypothesis strategies for the given parameters.
    Automatically handles shared symbolic dimensions (e.g. Array['N']).

    Args:
        params: List of argument names.
        hints: Mapping of name -> type string.
        constraints: Mapping of name -> dict constraints (min, max, etc).

    Returns:
        Dict[str, Strategy]: Strategies ready to be fed into @given.
    """
    hints = hints or {}
    constraints = constraints or {}
    strategies = {}

    # Shared Context for dimension symbols
    shared_dims = {}

    for p in params:
      hint = hints.get(p)
      cons = constraints.get(p, {})

      # If hint maps to "Any" or is missing, try to infer from heuristic name
      if not hint or hint == "Any":
        # Heuristic Logic Ported to Strategy
        inferred_type = guess_dtype_by_name(p)

        # Handling name-based heuristics via explicit types to reuse strategies logic
        if p in ["axis", "dim"]:
          # Dimension usually integer within small range
          hint = "int"
          cons.setdefault("min", 0)
          cons.setdefault("max", 3)
        elif p in ["shape", "size"]:
          # Shape is Tuple[int, ...], but we can use list strategy here
          hint = "Tuple[int, ...]"
        elif inferred_type == "bool":
          # Default to boolean Array if name implies mask, or scalar logic?
          # Legacy behavior was Array('bool').
          hint = "Array"
          cons.setdefault("dtype", "bool")
        elif inferred_type == "int":
          # Indices are usually arrays
          hint = "Array"
          cons.setdefault("dtype", "int")
        elif inferred_type == "float":
          # Alpha/eps scalars
          if any(prefix in p.lower() for prefix in ["alpha", "eps", "scalar", "val"]):
            hint = "float"
          else:
            hint = "Array"
            cons.setdefault("dtype", "float")

      strategies[p] = strategies_from_spec(hint, cons, shared_dims)

    return strategies

  def generate_inputs(
    self, params: List[str], hints: Optional[Dict[str, str]] = None, constraints: Optional[Dict[str, Dict]] = None
  ) -> Dict[str, Any]:
    """
    Generates a SINGLE set of inputs. Valid for simple legacy tests and harnesses.

    Under the hood, this builds a Hypothesis strategy and draws one example.
    This replaces the old ad-hoc random generation logic.

    Args:
        params: List of argument names.
        hints: Type hints.
        constraints: Constraints.

    Returns:
        Dict[str, Any]: A dictionary of generated input values.
    """
    strat_dict = self.build_strategies(params, hints, constraints)
    # Create a fixed dictionary strategy
    composite = st.fixed_dictionaries(strat_dict)
    return composite.example()

  def adapt_to_framework(self, kwargs: Dict[str, Any], framework: str) -> Dict[str, Any]:
    """
    Delegates to Framework Adapter to convert Numpy/Native inputs to Tensors.

    Args:
        kwargs: Dictionary of input values.
        framework: Target framework key (e.g. 'torch').

    Returns:
        Dict with converted values.
    """
    adapter = get_adapter(framework)
    if not adapter:
      return kwargs

    converted = {}
    for k, v in kwargs.items():
      try:
        converted[k] = adapter.convert(v)
      except Exception:
        converted[k] = v
    return converted
