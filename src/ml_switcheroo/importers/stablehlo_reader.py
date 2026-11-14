r"""Importer for StableHLO Specification (OpenXLA).

This module parses the official StableHLO Markdown specifications (spec.md),
extracting operator names, descriptions, and argument signatures into the
Semantic Knowledge Base format.

It parses the specific structure of OpenXLA docs:
- Headers: `### \`op_name\`` (Backticked names).
- Semantics: Text bodies following headers.
- Syntax: `#### Syntax` blocks containing MLIR signatures.
"""

import re
import typing
from pathlib import Path

from pydantic import JsonValue

from ml_switcheroo.core.mlir.cst import OperationNode
from ml_switcheroo.utils.console import log_error, log_info

if typing.TYPE_CHECKING:
  from markdown_it.token import Token

OpDraftVal: typing.TypeAlias = list[str] | str | list[tuple[str, str]] | OperationNode


class StableHloSpecImporter:
  """Parse StableHLO Markdown specification files."""

  def parse_file(self, target_file: Path) -> dict[str, dict[str, JsonValue]]:
    """Parse `spec.md` from the StableHLO repository.

    Args:
        target_file: Path to the markdown file.

    Returns:
        Dictionary mapping Operator IDs (e.g. 'Abs') to ODL definitions.
    """
    if not target_file.exists():
      log_error(f"File not found: {target_file}")
      return {}

    log_info(f"Parsing StableHLO Spec: {target_file.name}...")
    return self._parse_markdown(target_file)

  @staticmethod
  def _extract_h3_raw_op_name(inline_token: "Token") -> str | None:
    """Extract snake_case operator name from a StableHLO ``<h3>`` inline token.

    Args:
        inline_token: ``markdown_it`` inline token following ``<h3>``.

    Returns:
        Raw snake_case operator identifier if matched, otherwise ``None``.
    """
    if inline_token.children and len(inline_token.children) >= 1:
      child = inline_token.children[0]
      if child.type in ("code_inline", "text") and re.match(r"^[a-z0-9_]+$", child.content.strip()):
        return child.content.strip()
    return None

  @staticmethod
  def _parse_syntax_fence(fence_token: "Token", current_def: dict[str, OpDraftVal]) -> None:
    """Populate ``raw_syntax`` and ``parsed_op`` on ``current_def`` from a Markdown code fence.

    Args:
        fence_token: ``markdown_it`` fence token.
        current_def: Mutable draft definition dictionary for the active operator.
    """
    if "mlir" not in fence_token.info.lower() and "stablehlo" not in fence_token.content:
      return
    from ml_switcheroo.core.mlir.parser import MlirParser

    for line in fence_token.content.splitlines():
      if "stablehlo." in line:
        stripped = line.strip()
        try:
          module = MlirParser(stripped).parse()
          if module.body.operations:
            current_def["raw_syntax"] = stripped
            current_def["parsed_op"] = module.body.operations[0]
        except (ValueError, TypeError, KeyError, IndexError, RuntimeError):
          current_def["raw_syntax"] = stripped

  def _parse_markdown(self, fpath: Path) -> dict[str, dict[str, JsonValue]]:
    """Parse markdown structures directly into semantic definitions.

    Args:
        fpath: Path to the markdown file.

    Returns:
        Dictionary mapping Operator IDs to definitions.
    """
    from markdown_it import MarkdownIt

    tokens = MarkdownIt().parse(fpath.read_text(encoding="utf-8"))
    semantics: dict[str, dict[str, JsonValue]] = {}
    current_op: str | None = None
    current_def: dict[str, OpDraftVal] = {}

    for i, token in enumerate(tokens):
      if token.type == "heading_open" and token.tag == "h3":
        if i + 1 < len(tokens) and tokens[i + 1].type == "inline":
          raw_name = self._extract_h3_raw_op_name(tokens[i + 1])
          if raw_name is not None:
            if current_op and current_def:
              self._finalize_op(semantics, current_op, current_def)
            current_op = self._normalize_op_name(raw_name)
            current_def = {"description": [], "raw_syntax": "", "std_args": []}
      elif current_op:
        if token.type == "paragraph_open" and i + 1 < len(tokens) and tokens[i + 1].type == "inline":
          desc_list = current_def.get("description")
          if isinstance(desc_list, list) and not desc_list:
            current_def["description"] = [tokens[i + 1].content]
        elif token.type == "fence":
          self._parse_syntax_fence(token, current_def)

    if current_op and current_def:
      self._finalize_op(semantics, current_op, current_def)

    return semantics

  def _finalize_op(
    self,
    semantics: dict[str, dict[str, JsonValue]],
    name: str,
    details: typing.Mapping[str, OpDraftVal],
  ) -> None:
    """Clean up and register the operation.

    Args:
        semantics: The accumulator dictionary to update.
        name: The operation name (Abstract ID).
        details: The raw extracted details (description list, syntax string).
    """
    # 1. Clean Description
    raw_desc = details.get("description", [])
    desc_list: list[str] = [str(s) for s in raw_desc] if isinstance(raw_desc, list) else []
    desc = " ".join(desc_list)
    if len(desc) > 300:
      desc = desc[:297] + "..."

    # 2. Extract Args from Syntax string
    args: list[JsonValue] = []

    parsed_op = details.get("parsed_op")
    if isinstance(parsed_op, OperationNode):
      # Use parsed operands, filtering out numeric intermediate values if any (though typically operands are named)
      for v in parsed_op.operands:
        v_name = v.name.strip("%")
        if not v_name.isdigit() and v_name not in ["result", "results"]:
          args.append(v_name)

    # Fallback if parsing failed or no arguments found
    if not args:
      args = ["input"]

    stablehlo_api_suffix = name.lower()
    if name == "Add":
      stablehlo_api_suffix = "add"
    elif name == "Sub":
      stablehlo_api_suffix = "subtract"
    elif name == "Mul":
      stablehlo_api_suffix = "multiply"
    elif name == "Div":
      stablehlo_api_suffix = "divide"
    elif name == "Pow":
      stablehlo_api_suffix = "power"

    semantics[name] = {
      "description": desc,
      "std_args": args,
      # We explicitly output the StableHLO variant here since we are reading its spec
      "variants": {"stablehlo": {"api": f"stablehlo.{stablehlo_api_suffix}"}},
    }

  def _normalize_op_name(self, name: str) -> str:
    """Convert 'abs' -> 'Abs', 'log_plus_one' -> 'LogPlusOne'.

    StableHLO uses snake_case. ODL uses PascalCase for Abstract IDs.

    Args:
        name: The raw snake_case name (e.g. 'log_plus_one').

    Returns:
        str: The PascalCase name (e.g. 'LogPlusOne').
    """
    # Manual overrides for consistency with existing Hub standards
    overrides = {
      "abs": "Abs",
      "add": "Add",
      "subtract": "Sub",
      "multiply": "Mul",
      "divide": "Div",
      "power": "Pow",
    }
    if name in overrides:
      return overrides[name]

    return "".join(word.capitalize() for word in name.split("_"))
