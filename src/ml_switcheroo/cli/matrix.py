"""Compatibility Matrix rendering logic.

This module generates the visual comparison table between supported frameworks.
It retrieves operations from the `SemanticsManager` and intersects them with
the dynamically discovered frameworks from the registry.

Sorting Order:
    Columns are sorted based on `ui_priority` defined in Framework Adapters.
    Typically: PyTorch (0) -> JAX (10) -> NumPy (20) -> TensorFlow (30) -> ...
"""

import typing

from pydantic import JsonValue
from rich.console import Console
from rich.table import Table

from ml_switcheroo.config import get_framework_priority_order
from ml_switcheroo.core.dsl import FrameworkVariant
from ml_switcheroo.semantics.manager import SemanticsManager


class CompatibilityMatrix:
  """Logic for generating the Framework Compatibility Matrix.

  It dynamically queries the Knowledge Base and Framework Registry to create
  a grid of [Operations x Frameworks].
  """

  def __init__(self, semantics: SemanticsManager):
    """Initialize the Matrix generator.

    Args:
        semantics (SemanticsManager): The loaded semantics manager.

    """
    self.semantics = semantics
    self.console = Console()

  def _get_sorted_engines(self) -> list[str]:
    """Return list of frameworks sorted for UI consistency.

    Delegates to ``ml_switcheroo.config.get_framework_priority_order`` which
    inspects ``ui_priority`` on registered adapters.

    Returns:
        List[str]: Identifiers like ['torch', 'jax', 'numpy'].

    """
    return get_framework_priority_order()

  def get_json(self) -> list[dict[str, str]]:
    """Return the compatibility matrix as properly structured data.

    Useful for downstream tools, web frontends, or CI parsers.

    Returns:
        List[Dict]: Rows of the matrix. Structure:

        .. code-block:: json

            [
                {
                    "operation": "Conv2d",
                    "tier": "Neural",
                    "torch": "✅",
                    "jax": "🧩"
                }
            ]

    """
    rows = []
    engines = self._get_sorted_engines()
    import types

    data = self.semantics.get_known_apis()
    gka_fn = self.semantics.get_known_apis
    is_default_gka = isinstance(gka_fn, types.MethodType) and gka_fn.__func__ is SemanticsManager.get_known_apis
    typed_ops = self.semantics.operations if is_default_gka else {}
    use_typed = bool(typed_ops) and set(typed_ops.keys()) == set(data.keys())
    sorted_keys = sorted(typed_ops.keys() if use_typed else data.keys())

    # Determine Origin/Tier for operations (heuristic or metadata)
    raw_origins = self.semantics._key_origins
    origins = raw_origins or {}

    for op_name in sorted_keys:
      tier_raw = origins.get(op_name, "Standard")
      tier_label = tier_raw.replace("_", " ").title()
      row_dict = {
        "operation": op_name,
        "tier": tier_label,
      }

      op_model = typed_ops.get(op_name) if use_typed else None
      if op_model is not None:
        for engine in engines:
          row_dict[engine] = self._get_status_icon(op_model.variants.get(engine))
      else:
        details = data[op_name]
        raw_vars = details.get("variants", {})
        variants = raw_vars if isinstance(raw_vars, dict) else {}
        for engine in engines:
          v_entry = variants.get(engine)
          row_dict[engine] = self._get_status_icon(v_entry if isinstance(v_entry, dict) else None)

      rows.append(row_dict)

    return rows

  def render(self) -> None:
    """Generate and prints the compatibility table to the standard output.

    Uses ``rich.Table`` for formatting.
    """
    table = Table(title="ml-switcheroo Compatibility Matrix")

    # Fixed Leading Columns
    table.add_column("Operation", style="cyan", no_wrap=True)
    table.add_column("Tier", style="magenta")

    # Dynamic Framework Columns
    engines = self._get_sorted_engines()
    for engine in engines:
      # Header is Uppercase Framework Name
      table.add_column(engine.upper(), justify="center")

    # Data Rows
    matrix_data = self.get_json()

    for row_data in matrix_data:
      # Build list of values matching the column order defined above
      row_values = [row_data["operation"], row_data["tier"]]
      row_values.extend(row_data[engine] for engine in engines)

      table.add_row(*row_values)

    self.console.print(table)

  def _get_status_icon(self, variant_info: FrameworkVariant | typing.Mapping[str, JsonValue] | None) -> str:
    """Determine the visual status icon for a mapping entry.

    Args:
        variant_info: Validated ``FrameworkVariant`` or raw variant mapping dictionary.

    Returns:
        str: Status emoji/character.
             ✅ = Direct API mapping available.
             🧩 = Plugin or complex logic required.
             ❌ = Not mapped / Missing.

    """
    if not variant_info:
      return "❌"

    if isinstance(variant_info, FrameworkVariant):
      return "🧩" if variant_info.requires_plugin else "✅"

    if "requires_plugin" in variant_info:
      return "🧩"

    return "✅"
