"""Utilities for updating the project README.

This module provides logic to inject automated verification reports (the Compatibility Matrix)
directly into the `README.md` file, ensuring documentation stays valid with the code.
"""

from pathlib import Path
from typing import Dict, Optional, Any

from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.utils.console import log_error, log_success


class ReadmeEditor:
  """Utility to programmatically update the README.md with verification results.

  It regenerates the "Compatibility Matrix" table based on the current state of
  the Knowledge Base and the results of the latest CI run, then splices it
  into the README content using structural parsing.
  """

  def __init__(self, semantics: SemanticsManager, readme_path: Path) -> None:
    """Initializes the editor.

    Args:
        semantics: The loaded semantics manager used to fetch API details.
        readme_path: File system path to the target markdown file.

    """
    self.semantics = semantics
    self.readme_path = readme_path

  def update_matrix(self, validation_results: Dict[str, bool]) -> bool:
    """Regenerates the Markdown table and injects it into the README structurally.

    It finds the Markdown heading `✅ Compatibility Matrix` and replaces the paragraph/table
    following it up until the next heading.

    Args:
        validation_results: Dictionary mapping op_name -> boolean pass status.

    Returns:
        bool: True if the update was successful, False otherwise.

    """
    if not self.readme_path.exists():
      log_error(f"README not found at {self.readme_path}")
      return False

    try:
      content = self.readme_path.read_text(encoding="utf-8")
    except OSError as e:
      log_error(f"Could not read README: {e}")
      return False

    # 1. Generate New Table
    new_table = self._generate_markdown_table(validation_results)

    # 2. Inject structurally using markdown-it
    from markdown_it import MarkdownIt

    md = MarkdownIt()
    tokens = md.parse(content)

    header_marker = "✅ Compatibility Matrix"
    target_idx = -1
    next_heading_idx = -1

    for i, token in enumerate(tokens):
      if token.type == "heading_open":
        if i + 1 < len(tokens) and tokens[i + 1].type == "inline":  # pragma: no branch
          if header_marker in tokens[i + 1].content:
            target_idx = i
            break

    if target_idx == -1:
      log_error(f"Could not find '{header_marker}' section in README.")
      return False

    for i in range(target_idx + 3, len(tokens)):
      if tokens[i].type == "heading_open":
        next_heading_idx = i
        break

    # We need to map tokens back to lines.
    # tokens[target_idx].map contains the line numbers [start, end]
    start_line = tokens[target_idx].map[1] if tokens[target_idx].map else -1  # type: ignore

    if start_line == -1:
      log_error("Could not determine line mapping for header.")
      return False

    end_line = (
      tokens[next_heading_idx].map[0]  # type: ignore
      if next_heading_idx != -1 and tokens[next_heading_idx].map
      else len(content.splitlines())
    )

    lines = content.splitlines()
    pre_lines = lines[:start_line]
    post_lines = lines[end_line:] if end_line != len(lines) else []

    new_lines = pre_lines + [""] + new_table.splitlines() + [""] + post_lines

    try:
      self.readme_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
      log_success(f"Updated README.md with {len(validation_results)} status entries.")
      return True
    except OSError as e:
      log_error(f"Failed to write to README: {e}")
      return False

  def _generate_markdown_table(self, results: Dict[str, bool]) -> str:
    """Constructs the ASCII Markdown table from semantics data.

    Generates a table row for every operation known in the SemanticsManager.

    Columns:
        - Category: Derived from Semantic Tier heuristics.
        - PyTorch: Source API path (from variants).
        - JAX: Target API path (from variants).
        - Verification: Status icon based on results dict.

    Args:
        results: Validation outcomes (True/False).

    Returns:
        str: The fully formatted markdown table string.

    """
    known_apis = self.semantics.get_known_apis()
    # Sort for deterministic output
    sorted_ops = sorted(known_apis.keys())

    # Header
    lines = [
      "View the live matrix by running `ml_switcheroo matrix`",
      "",
      "| Category | PyTorch | JAX | Verification |",
      "| :--- | :--- | :--- | :--- |",
    ]

    for op in sorted_ops:
      details = known_apis[op]
      variants = details.get("variants", {})

      # Determine API Text (using code backticks)
      torch_api = variants.get("torch", {}).get("api", "—")
      jax_variant = variants.get("jax", {})

      # Handle explicit None (null) used in Tier C
      if jax_variant is None:
        jax_api = "—"
        plugin_info = False
      else:
        jax_api = jax_variant.get("api", "—")
        plugin_info = "requires_plugin" in jax_variant

      t_cell = f"`{torch_api}`" if torch_api != "—" else "—"
      j_cell = f"`{jax_api}`" if jax_api != "—" else "—"

      # Determine Status Icon
      is_valid = results.get(op, False)

      if is_valid:
        status = "✅ Passing"
      elif plugin_info:
        # If verification failed but it uses a plugin, mark separately implies complexity
        status = "🧩 Plugin (Complex)"
      else:
        status = "⚠️ Untested/Fail"

      # Determine Category
      category = _guess_category(torch_api, jax_variant)

      row = f"| **{category}** | {t_cell} | {j_cell} | {status} |"
      lines.append(row)

    return "\n".join(lines)


def _guess_category(api_name: str, target_var: Optional[Dict[Any, Any]]) -> str:
  """Heuristic helper to categorize op based on API string contents.

  Args:
      api_name: The Torch/Source API path.
      target_var: The target dictionary (to check for plugins).

  Returns:
      str: "Neural", "Special", or "Math".

  """
  if "nn" in api_name or "Linear" in api_name or "Conv" in api_name:
    return "Neural"
  if target_var and "requires_plugin" in target_var:
    return "Special"
  return "Math"
