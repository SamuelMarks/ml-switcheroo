"""Core Engine of the Input Fuzzer (Hypothesis Integration).

This module provides the `InputFuzzer` facade which now delegates generation logic
to Hypothesis Strategies. It maintains backward compatibility for casual usage via `generate_inputs`.
"""

import typing

import hypothesis.strategies as st

from ml_switcheroo.enums import RuntimeTensorOrValue
from ml_switcheroo.frameworks import get_adapter
from ml_switcheroo.testing.fuzzer.heuristics import guess_dtype_by_name
from ml_switcheroo.testing.fuzzer.strategies import strategies_from_spec


class InputFuzzer:
  """Facade for creating Hypothesis strategies based on Semantic Spec."""

  def build_strategies(
    self,
    params: list[str],
    hints: typing.Mapping[str, str] | None = None,
    constraints: typing.Mapping[str, typing.Mapping[str, RuntimeTensorOrValue]] | None = None,
  ) -> dict[str, st.SearchStrategy[RuntimeTensorOrValue]]:
    """Construct a dictionary of Hypothesis strategies for the given parameters.

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
    strategies: dict[str, st.SearchStrategy[RuntimeTensorOrValue]] = {}

    # Shared Context for dimension symbols
    shared_dims: dict[str, st.SearchStrategy[int]] = {}

    for p in params:
      hint = hints.get(p)
      cons: dict[str, RuntimeTensorOrValue] = dict(constraints.get(p, {}))

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

      strategies[p] = strategies_from_spec(str(hint or "Any"), cons, shared_dims)

    return strategies

  @staticmethod
  def _convert_value(
    convert_fn: typing.Callable[[RuntimeTensorOrValue], RuntimeTensorOrValue],
    val: RuntimeTensorOrValue,
  ) -> RuntimeTensorOrValue:
    """Convert a single input value via ``convert_fn``, falling back to ``val`` on failure.

    Args:
        convert_fn: Adapter conversion callable accepting a runtime value.
        val: Raw input value to convert.

    Returns:
        Converted framework tensor/value, or the original ``val`` if conversion fails.
    """
    try:
      return convert_fn(val)
    except (ImportError, AttributeError, ValueError, TypeError, RuntimeError):
      return val

  def adapt_to_framework(
    self, kwargs: typing.Mapping[str, RuntimeTensorOrValue], framework: str
  ) -> dict[str, RuntimeTensorOrValue]:
    """Delegate to Framework Adapter to convert NumPy/Native inputs to Tensors.

    Args:
        kwargs: Dictionary of input values.
        framework: Target framework key (e.g. 'torch').

    Returns:
        Dict with converted values.

    """
    adapter = get_adapter(framework)
    if not adapter:
      return dict(kwargs)

    return {k: self._convert_value(adapter.convert, v) for k, v in kwargs.items()}
