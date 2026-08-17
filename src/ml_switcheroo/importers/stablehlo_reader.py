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
from pathlib import Path
from typing import Any, Dict, List, Union

from ml_switcheroo.utils.console import log_error, log_info


class StableHloSpecImporter:
  """Parses StableHLO Markdown specification files."""

  def parse_file(self, target_file: Path) -> Dict[str, Any]:
    """Parses `spec.md` from the StableHLO repository.

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

  def _parse_markdown(self, fpath: Path) -> Dict[str, Any]:
    """Parse markdown structures directly into semantic definitions.

    Args:
        fpath: Path to the markdown file.

    Returns:
        Dictionary mapping Operator IDs to definitions.
    """
    from markdown_it import MarkdownIt

    content = fpath.read_text(encoding="utf-8")
    md = MarkdownIt()
    tokens = md.parse(content)

    semantics: Dict[str, Any] = {}
    current_op: Union[str, None] = None
    current_def: Dict[str, Any] = {}

    for i, token in enumerate(tokens):
      if token.type == "heading_open" and token.tag == "h3":
        if i + 1 < len(tokens) and tokens[i + 1].type == "inline":  # pragma: no branch
          inline = tokens[i + 1]
          if inline.children and len(inline.children) >= 1:
            child = inline.children[0]
            if child.type in ("code_inline", "text") and re.match(r"^[a-z0-9_]+$", child.content.strip()):
              raw_name = child.content.strip()
              if current_op and current_def:
                self._finalize_op(semantics, current_op, current_def)
              current_op = self._normalize_op_name(raw_name)
              current_def = {"description": [], "raw_syntax": "", "std_args": []}
      elif current_op:
        if token.type == "paragraph_open":
          if i + 1 < len(tokens) and tokens[i + 1].type == "inline":  # pragma: no branch
            if not current_def["description"]:
              current_def["description"].append(tokens[i + 1].content)
        elif token.type == "fence":
          # Syntax Block
          if "mlir" in token.info.lower() or "stablehlo" in token.content:
            from ml_switcheroo.core.mlir.parser import MlirParser

            for line in token.content.splitlines():
              if "stablehlo." in line:
                try:
                  parser = MlirParser(line.strip())
                  module = parser.parse()
                  if module.body and module.body.operations:  # pragma: no branch
                    parsed_op = module.body.operations[0]
                    current_def["raw_syntax"] = line.strip()
                    current_def["parsed_op"] = parsed_op
                except Exception:
                  # Fallback to saving raw string if parse fails
                  current_def["raw_syntax"] = line.strip()

    if current_op and current_def:  # pragma: no branch
      self._finalize_op(semantics, current_op, current_def)

    return semantics

  def _finalize_op(self, semantics: Dict[str, Any], name: str, details: Dict[str, Any]) -> None:
    """Clean up and register the operation.

    Args:
        semantics: The accumulator dictionary to update.
        name: The operation name (Abstract ID).
        details: The raw extracted details (description list, syntax string).
    """
    # 1. Clean Description
    desc_list: List[str] = details.get("description", [])
    desc = " ".join(desc_list)
    if len(desc) > 300:
      desc = desc[:297] + "..."

    # 2. Extract Args from Syntax string
    args = []

    if "parsed_op" in details:
      parsed_op = details["parsed_op"]
      # Use parsed operands, filtering out numeric intermediate values if any (though typically operands are named)
      for v in parsed_op.operands:
        v_name = v.name.strip("%")
        if not v_name.isdigit() and v_name not in ["result", "results"]:  # pragma: no branch
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
    """Converts 'abs' -> 'Abs', 'log_plus_one' -> 'LogPlusOne'.

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
