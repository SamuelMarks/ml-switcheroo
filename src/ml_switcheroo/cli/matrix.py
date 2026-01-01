"""
Compatibility Matrix rendering logic.

This module generates the visual comparison table between supported frameworks.
It retrieves operations from the `SemanticsManager` and intersects them with
the dynamically discovered frameworks from the registry.

Sorting Order:
    Columns are sorted based on `ui_priority` defined in Framework Adapters.
    Typically: PyTorch (0) -> JAX (10) -> NumPy (20) -> TensorFlow (30) -> ...
"""

from typing import Any, Dict, List, Optional, Union

from rich.console import Console
from rich.table import Table

from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.config import get_framework_priority_order


class CompatibilityMatrix:
  """
  Logic for generating the Framework Compatibility Matrix.

  It dynamically queries the Knowledge Base and Framework Registry to create
  a grid of [Operations x Frameworks].
  """

  def __init__(self, semantics: SemanticsManager):
    """
    Initializes the Matrix generator.

    Args:
        semantics (SemanticsManager): The loaded semantics manager.
    """
    self.semantics = semantics
    self.console = Console()

  def _get_sorted_engines(self) -> List[str]:
    """
    Returns list of frameworks sorted for UI consistency.

    Delegates to `ml_switcheroo.config.get_framework_priority_order` which
    inspects `ui_priority` on registered adapters.

    Returns:
        List[str]: Identifiers like ['torch', 'jax', 'numpy'].
    """
    return get_framework_priority_order()

  def get_json(self) -> List[Dict[str, str]]:
    """
    Returns the compatibility matrix as properly structured data.

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
    data = self.semantics.get_known_apis()
    sorted_keys = sorted(data.keys())

    # Determine Origin/Tier for operations (heuristic or metadata)
    origins = getattr(self.semantics, "_key_origins", {})

    for op_name in sorted_keys:
      details = data[op_name]
      variants = details.get("variants", {})

      # Resolve Tier Name
      tier_raw = origins.get(op_name, "Standard")
      # Clean up tier string (e.g. 'array' -> 'Array')
      tier_label = tier_raw.replace("_", " ").title()

      # Base Row
      row_dict = {
        "operation": op_name,
        "tier": tier_label,
      }

      # Populate status for each registered engine
      for engine in engines:
        status = self._get_status_icon(variants.get(engine))
        row_dict[engine] = status

      rows.append(row_dict)

    return rows

  def render(self) -> None:
    """
    Generates and prints the compatibility table to the standard output.
    Uses `rich.Table` for formatting.
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
      for engine in engines:
        row_values.append(row_data[engine])

      table.add_row(*row_values)

    self.console.print(table)

  def _get_status_icon(self, variant_info: Optional[Dict[str, Any]]) -> str:
    """
    Determines the visual status icon for a mapping entry.

    Args:
        variant_info (Optional[Dict]): The dictionary describing the variant's implementation.

    Returns:
        str: Status emoji/character.
             ✅ = Direct API mapping available.
             🧩 = Plugin or complex logic required.
             ❌ = Not mapped / Missing.
    """
    # If None, explicit failure. If missing, missing.
    if not variant_info:
      return "❌"

    # If a plugin is required, it's supported but complex.
    if "requires_plugin" in variant_info:
      return "🧩"

    # Otherwise assumed direct API mapping
    return "✅"
