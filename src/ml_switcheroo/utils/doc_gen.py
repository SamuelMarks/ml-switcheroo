"""
Migration Guide Generator.

This module provides the logic to generate human-readable Markdown documentation
comparing two frameworks (Source vs Target). It iterates through the
Semantic Knowledge Base, calculating differences in API names and argument
conventions (e.g., `dim` vs `axis`), and producing a structured migration guide.
"""

from typing import Dict, Any, List, Optional
from collections import defaultdict

from ml_switcheroo.semantics.manager import SemanticsManager


class MigrationGuideGenerator:
  """
  Generates a Markdown Migration Guide by diffing semantic specifications.

  Attributes:
      semantics (SemanticsManager): The loaded knowledge base containing
          operations, variants, and argument maps.
  """

  def __init__(self, semantics: SemanticsManager):
    """
    Initializes the generator.

    Args:
        semantics: The knowledge base manager.
    """
    self.semantics = semantics

  def generate(self, source_fw: str, target_fw: str) -> str:
    """
    Produces the Markdown content for the migration guide.

    It groups operations by their Semantic Tier (Math, Neural, Extras),
    sorts them alphabetically, and generates a comparison table for each tier.

    Args:
        source_fw: The framework converting FROM (e.g., 'torch').
        target_fw: The framework converting TO (e.g., 'jax').

    Returns:
        A string containing the rendered Markdown.
    """
    # 1. Group Operations by Tier
    # Structure: {"array": [op_name, ...], "neural": [...]}
    tiered_ops: Dict[str, List[str]] = defaultdict(list)
    known_apis = self.semantics.get_known_apis()
    origins = getattr(self.semantics, "_key_origins", {})

    for op_name in known_apis.keys():
      tier = origins.get(op_name, "Uncategorized")
      tiered_ops[tier].append(op_name)

    # 2. Build Markdown
    md_lines = [
      f"# Migration Guide: {source_fw.capitalize()} to {target_fw.capitalize()}",
      "",
      "This guide is automatically generated from the ml-switcheroo Semantic Knowledge Base.",
      "It highlights API name mappings and argument renaming rules.",
      "",
    ]

    # Sort tiers for consistent output order
    # We manually prioritize: array, neural, extras, others
    tier_priority = ["array", "neural", "extras"]
    sorted_tiers = sorted(
      tiered_ops.keys(),
      key=lambda t: tier_priority.index(t) if t in tier_priority else 99,
    )

    for tier in sorted_tiers:
      ops_list = sorted(tiered_ops[tier])
      # Filter ops to identifying those that actually have source/target definitions
      valid_ops = [op for op in ops_list if self._has_variants(op, source_fw)]

      if not valid_ops:
        continue

      md_lines.append(f"## {tier.capitalize().replace('_', ' ')}")
      md_lines.append("")
      md_lines.append(self._generate_table_header(source_fw, target_fw))

      for op in valid_ops:
        row = self._generate_op_row(op, source_fw, target_fw)
        md_lines.append(row)

      md_lines.append("")

    return "\n".join(md_lines)

  def _has_variants(self, op_name: str, source: str) -> bool:
    """Checks if an operation has definitions for at least the source framework."""
    details = self.semantics.get_definition_by_id(op_name)
    if not details:
      return False
    variants = details.get("variants", {})
    # We list it if Source exists. Target might be missing (which is useful info).
    return source in variants

  def _generate_table_header(self, source: str, target: str) -> str:
    """Creates the Markdown table header."""
    s_title = source.capitalize()
    t_title = target.capitalize()
    return f"| {s_title} API | {t_title} API | Argument Changes |\n| :--- | :--- | :--- |"

  def _generate_op_row(self, op_name: str, source: str, target: str) -> str:
    """
    Calculates the diff row for a single operation.

    Args:
        op_name: Abstract operation ID.
        source: Source framework name.
        target: Target framework name.

    Returns:
        A Markdown table row string.
    """
    details = self.semantics.get_definition_by_id(op_name) or {}
    variants = details.get("variants", {})
    std_args = details.get("std_args", [])

    # Clean std_args (remove types if present)
    # ["x"] or [("x", "int")] -> ["x"]
    clean_std_args = []
    for item in std_args:
      if isinstance(item, (list, tuple)):
        clean_std_args.append(item[0])
      else:
        clean_std_args.append(item)

    # 1. Get API Paths
    src_var = variants.get(source)
    tgt_var = variants.get(target)

    src_api = self._fmt_api(src_var)
    tgt_api = self._fmt_api(tgt_var)

    # 2. Calculate Argument Diff
    # We want to show: "dim -> axis"
    # Logic:
    #   Src Arg Name = src_map[std_name] OR std_name
    #   Tgt Arg Name = tgt_map[std_name] OR std_name
    #   If Src != Tgt: Record diff
    diffs = []

    src_args_map = src_var.get("args", {}) if src_var else {}
    tgt_args_map = tgt_var.get("args", {}) if tgt_var else {}

    # Invert the maps for lookup: {fw_name: std_name}
    # Actually no, semantics stores {std_name: fw_name}.
    # Wait, let's verify semantics structure from previous files.
    # k_array_api.json: "std_args": ["x", "axis"], "variants": { "torch": { "args": {"axis": "dim"} } }
    # So key is std_name, value is fw_name. Correct.

    for std_arg in clean_std_args:
      src_arg = src_args_map.get(std_arg, std_arg)
      tgt_arg = tgt_args_map.get(std_arg, std_arg)

      if src_arg != tgt_arg:
        diffs.append(f"`{src_arg}`&#8594;`{tgt_arg}`")

    diff_str = ", ".join(diffs) if diffs else "-"

    # 3. Check for Plugins
    if tgt_var and "requires_plugin" in tgt_var:
      plugin = tgt_var["requires_plugin"]
      tgt_api += f" <br/>*(Plugin: {plugin})*"

    return f"| `{src_api}` | `{tgt_api}` | {diff_str} |"

  def _fmt_api(self, variant: Optional[Dict[str, Any]]) -> str:
    """Safe formatter for API variant."""
    if not variant:
      return "—"
    return variant.get("api", "—")
