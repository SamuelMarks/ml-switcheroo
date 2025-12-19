"""
Compatibility Matrix rendering logic.

This module is responsible for generating the visual comparison table between
supported frameworks. It extracts operation mapping statuses from the
`SemanticsManager` and presents them either as a formatted Rich table (CLI)
or structured JSON data (Web Interface).

Refactor: Uses `get_framework_priority_order()` for fully dynamic sorting.
"""

from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.table import Table

from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.config import get_framework_priority_order


class CompatibilityMatrix:
  """
  Logic for generating the Framework Compatibility Matrix.

  It queries the Knowledge Base to determine which operations have defined
  variants for specific backends.
  """

  def __init__(self, semantics: SemanticsManager):
    """
    Initializes the Matrix generator.

    Args:
        semantics (SemanticsManager): The loaded semantics manager containing API data.
    """
    self.semantics = semantics
    self.console = Console()

  def _get_sorted_engines(self) -> List[str]:
    """
    Returns list of frameworks sorted for UI consistency.
    Delegates to central config logic which queries adapters.
    """
    return get_framework_priority_order()

  def get_json(self) -> List[Dict[str, str]]:
    """
    Returns the compatibility matrix as structured data.

    Useful for web interfaces or downstream tools that need to render
    the table dynamically (e.g. in React/Angular).

    Returns:
        List[Dict[str, str]]: A list of row objects.
        Each dictionary contains:
            - 'operation': The abstract operation name.
            - 'tier': The semantic category.
            - '{engine}': The status icon for that engine (e.g. 'torch': '✅').
    """
    rows = []
    engines = self._get_sorted_engines()
    data = self.semantics.get_known_apis()
    sorted_keys = sorted(data.keys())

    for op_name in sorted_keys:
      details = data[op_name]
      variants = details.get("variants", {})

      # Base Row Data
      row_dict = {
        "operation": op_name,
        "tier": "Standard",  # Placeholder for future Tier enrichment
      }

      # Engine Statuses
      for engine in engines:
        status = self._get_status_icon(variants.get(engine))
        row_dict[engine] = status

      rows.append(row_dict)

    return rows

  def render(self) -> None:
    """
    Generates and prints the compatibility table to the console.

    Uses the same underlying logic as `get_json` but formats it into
    a Rich Table with colored columns.
    """
    table = Table(title="ml-switcheroo Compatibility Matrix")

    # Columns
    table.add_column("Operation", style="cyan", no_wrap=True)
    table.add_column("Tier", style="magenta")

    engines = self._get_sorted_engines()
    for engine in engines:
      table.add_column(engine.upper(), justify="center")

    # Reuse data generation logic
    matrix_data = self.get_json()

    for row_data in matrix_data:
      # Extract list of values in specific order for Table.add_row
      row_values = [row_data["operation"], row_data["tier"]]
      for engine in engines:
        row_values.append(row_data[engine])

      table.add_row(*row_values)

    self.console.print(table)

  def _get_status_icon(self, variant_info: Optional[Dict[str, Any]]) -> str:
    """
    Determines the visual status icon for a mapping entry.

    Args:
        variant_info (Optional[Dict]): The variant definition dictionary
                                       from the semantics JSON.

    Returns:
        str: An emoji/char representing status.
             '❌' : Not mapped / Missing.
             '🧩' : Mapped via Plugin (Complex).
             '✅' : Mapped directly via API.
    """
    if not variant_info:
      return "❌"

    if "requires_plugin" in variant_info:
      return "🧩"  # Puzzle piece means plugin logic

    return "✅"
