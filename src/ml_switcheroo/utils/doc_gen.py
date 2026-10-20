"""Migration Guide Generator.

This module provides the logic to generate human-readable Markdown documentation
comparing two frameworks (Source vs Target). It iterates through the
Semantic Knowledge Base, calculating differences in API names and argument
conventions (e.g., `dim` vs `axis`), and producing a structured migration guide.
"""

import typing
from collections import defaultdict

from pydantic import JsonValue

if typing.TYPE_CHECKING:
  from ml_switcheroo.semantics.manager import SemanticsManager


class MigrationGuideGenerator:
  """Generate a Markdown Migration Guide by diffing semantic specifications."""

  def __init__(self, semantics: "SemanticsManager") -> None:
    """Initialize the generator.

    Args:
        semantics: The knowledge base manager.

    """
    self.semantics = semantics

  def generate(self, source_fw: str, target_fw: str) -> str:
    """Produce the Markdown content for the migration guide.

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
    tiered_ops: typing.Dict[str, typing.List[str]] = defaultdict(list)
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
    """Check whether an abstract operation defines a variant for the given source framework.

    Args:
        op_name: Abstract operation identifier string.
        source: Source framework key to check.

    Returns:
        True if the operation has a variant entry for ``source``, False otherwise.
    """
    details = self.semantics.get_definition_by_id(op_name)
    if not details:
      return False
    variants = details.get("variants", {})
    return isinstance(variants, dict) and source in variants

  def _generate_table_header(self, source: str, target: str) -> str:
    """Create the Markdown table header.

    Args:
        source: The name of the source framework.
        target: The name of the target framework.

    Returns:
        The Markdown table header string.
    """
    s_title = source.capitalize()
    t_title = target.capitalize()
    return f"| {s_title} API | {t_title} API | Argument Changes |\n| :--- | :--- | :--- |"

  def _generate_op_row(self, op_name: str, source: str, target: str) -> str:
    """Calculate the diff row for a single operation.

    Args:
        op_name: Abstract operation ID.
        source: Source framework name.
        target: Target framework name.

    Returns:
        A Markdown table row string.

    """
    details = self.semantics.get_definition_by_id(op_name) or {}
    raw_variants = details.get("variants", {})
    variants = raw_variants if isinstance(raw_variants, dict) else {}
    raw_std_args = details.get("std_args", [])
    std_args = raw_std_args if isinstance(raw_std_args, list) else []

    # Clean std_args (remove types if present)
    # ["x"] or [("x", "int")] -> ["x"]
    clean_std_args: typing.List[str] = []
    for item in std_args:
      if isinstance(item, (list, tuple)) and item:
        clean_std_args.append(str(item[0]))
      elif isinstance(item, dict):
        clean_std_args.append(str(item.get("name", "")))
      else:
        clean_std_args.append(str(item))

    # 1. Get API Paths
    raw_src_var = variants.get(source)
    src_var = raw_src_var if isinstance(raw_src_var, dict) else None
    raw_tgt_var = variants.get(target)
    tgt_var = raw_tgt_var if isinstance(raw_tgt_var, dict) else None

    src_api = self._fmt_api(src_var)
    tgt_api = self._fmt_api(tgt_var)

    # 2. Calculate Argument Diff
    diffs = []

    raw_src_args = src_var.get("args", {}) if src_var else {}
    src_args_map = raw_src_args if isinstance(raw_src_args, dict) else {}
    raw_tgt_args = tgt_var.get("args", {}) if tgt_var else {}
    tgt_args_map = raw_tgt_args if isinstance(raw_tgt_args, dict) else {}

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

  def _fmt_api(
    self, variant: typing.Optional[typing.Mapping[str, typing.Union[JsonValue, typing.Hashable, typing.Sized]]]
  ) -> str:
    """Safe formatter for API variant.

    Args:
        variant: The API variant definition dictionary, or None.

    Returns:
        The formatted API path string.
    """
    if not variant:
      return "—"
    res = variant.get("api", "—")
    assert isinstance(res, str)
    return res
