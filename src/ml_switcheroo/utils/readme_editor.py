import re
from pathlib import Path
from typing import Dict, Optional

from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.utils.console import log_error, log_success


class ReadmeEditor:
  """
  Utility to programmatically update the README.md with verification results.

  It regenerates the "Compatibility Matrix" table based on the current state of
  the Knowledge Base and the results of the latest CI run, then splices it
  into the README content using regex markers.
  """

  def __init__(self, semantics: SemanticsManager, readme_path: Path) -> None:
    """
    Initializes the editor.

    Args:
        semantics: The loaded semantics manager.
        readme_path: File system path to the target markdown file.
    """
    self.semantics = semantics
    self.readme_path = readme_path

  def update_matrix(self, validation_results: Dict[str, bool]) -> bool:
    """
    Regenerates the Markdown table and injects it into the README.

    It looks for the section starting with `## ✅ Compatibility Matrix` and
    replaces the content up to the next `## Header` or End of File.

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

    # 2. Inject via Regex
    # Pattern: Matches the Header, newlines, then captures content until next ## header
    header_marker = "## ✅ Compatibility Matrix"

    if header_marker not in content:
      log_error(f"Could not find '{header_marker}' section in README.")
      return False

    # Splitting approach is more robust than regex grouping for large multiline blocks
    parts = content.split(header_marker)

    # Pre-header content
    pre_table = parts[0]
    # Content after header
    remainder = parts[1]

    # Find the start of the next section (## ...) inside remainder
    # We look for a newline followed by ##
    next_section_match = re.search(r"\n## ", remainder)

    if next_section_match:
      # Found next section
      post_table = remainder[next_section_match.start() :]
    else:
      # No next section, table goes to EOF
      post_table = ""

    # Add buffering newlines
    new_content = f"{pre_table}{header_marker}\n\n{new_table}\n{post_table}"

    try:
      self.readme_path.write_text(new_content, encoding="utf-8")
      log_success(f"Updated README.md with {len(validation_results)} status entries.")
      return True
    except OSError as e:
      log_error(f"Failed to write to README: {e}")
      return False

  def _generate_markdown_table(self, results: Dict[str, bool]) -> str:
    """
    Constructs the ASCII Markdown table from semantics data.

    Columns:
        - Category: Derived from Semantic Tier heuristics.
        - PyTorch: Source API path.
        - JAX: Target API path.
        - Verification: Status icon based on results dict.

    Args:
        results: Validation outcomes.

    Returns:
        str: The formatted markdown table.
    """
    known_apis = self.semantics.get_known_apis()
    # Sort for deterministic output
    sorted_ops = sorted(known_apis.keys())

    # Header
    lines = [
      "View the live matrix by running `ml_switcheroo matrix`\n",
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
        # If verification failed but it uses a plugin, mark separately
        status = "🧩 Plugin (Complex)"
      else:
        status = "⚠️ Untested/Fail"

      # Simple Category Heuristic (Ideally this data is stored in SemanticsManager)
      category = _guess_category(torch_api, jax_variant)

      row = f"| **{category}** | {t_cell} | {j_cell} | {status} |"
      lines.append(row)

    return "\n".join(lines)


def _guess_category(api_name: str, target_var: Optional[Dict]) -> str:
  """Helper to categorize op based on API string."""
  if "nn" in api_name or "Linear" in api_name or "Conv" in api_name:
    return "Neural"
  if target_var and "requires_plugin" in target_var:
    return "Special"
  return "Math"
