"""
Importer for StableHLO Specification (OpenXLA).

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
from typing import Any, Dict, List

from ml_switcheroo.utils.console import log_error, log_info


class StableHloSpecImporter:
  """
  Parses StableHLO Markdown specification files.
  """

  # Regex to find headers like ### `abs`
  _HEADER_RE = re.compile(r"^### `(?P<name>[a-zA-Z0-9_]+)`$")

  # Regex to extract MLIR syntax: %result = stablehlo.abs %operand : tensor<...>
  # Captures the arguments roughly
  _SYNTAX_RE = re.compile(r"stablehlo\.[a-z_]+\s+(?P<args>.*?)\s+:")

  def parse_file(self, target_file: Path) -> Dict[str, Any]:
    """
    Parses `spec.md` from the StableHLO repository.

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
    """
    Iterates usage of the markdown file line-by-line to build op definitions.

    Args:
        fpath: Path to the markdown file.

    Returns:
        Dictionary of semantic operation definitions.
    """
    content = fpath.read_text(encoding="utf-8")
    lines = content.splitlines()

    semantics: Dict[str, Any] = {}
    current_op: str | None = None
    current_def: Dict[str, Any] = {}

    # Parse Loop
    for line in lines:
      line = line.strip()

      # 1. Detect Header (New Operation)
      match = self._HEADER_RE.match(line)
      if match:
        # Save previous
        if current_op and current_def:
          self._finalize_op(semantics, current_op, current_def)

        # Start new
        # Op names are lower_case (e.g. 'add'). We capitalize for Abstract ID 'Add'.
        raw_name = match.group("name")
        current_op = self._normalize_op_name(raw_name)
        current_def = {"description": [], "raw_syntax": "", "std_args": []}
        continue

      # 2. Capture Description (if inside op)
      if current_op:
        # Heuristic: Capture logic text until the next Header or syntax block
        if line.startswith("#"):
          # Sub-headers like #### Semantics, #### Inputs
          pass
        elif line.startswith("```"):
          # Code block limiters
          pass
        elif not current_def["description"] and line:
          # First non-empty paragraph is usually the summary
          current_def["description"].append(line)

        # 3. Capture Syntax Block (Basic Argument Inference)
        # We look for the MLIR syntax line inside code blocks (naive heuristic)
        if "stablehlo." in line and "%" in line:
          syntax_match = self._SYNTAX_RE.search(line)
          if syntax_match:
            current_def["raw_syntax"] = syntax_match.group("args")

    # Finalize last op
    if current_op and current_def:
      self._finalize_op(semantics, current_op, current_def)

    return semantics

  def _finalize_op(self, semantics: Dict[str, Any], name: str, details: Dict[str, Any]) -> None:
    """
    Clean up and register the operation.

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
    # Syntax usually looks like: "%lhs, %rhs" or "(%lhs, %rhs)"
    args = []
    raw_syntax = details.get("raw_syntax", "")
    if raw_syntax:
      # Find tokens starting with %
      vars_found = re.findall(r"%([a-zA-Z0-9_]+)", raw_syntax)
      for v in vars_found:
        # Filter out 'result' or '0' if they appear in valid position
        # '0', '1' are often results or intermediate SSAs
        if not v.isdigit() and v not in ["result", "results"]:
          args.append(v)

    # Fallback if parsing failed
    if not args:
      args = ["input"]

    # Infer base name for StableHLO API from ODL name if possible,
    # but really we should use the map or derive it.
    # However, the extraction loop normalized the name for the key.
    # We need the original snake_case for the API.
    # Since we lost it in _normalize_op_name, we reconstruct or store it?
    # Reconstructing from PascalCase is hard if acronyms involved.
    # Ideally _normalize_op_name handled it, but here we can just lower().
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
    # For others produced by capitalize loop (e.g. Abs -> abs), lower() works.

    semantics[name] = {
      "description": desc,
      "std_args": args,
      # We explicitly output the StableHLO variant here since we are reading its spec
      "variants": {"stablehlo": {"api": f"stablehlo.{stablehlo_api_suffix}"}},
    }

  def _normalize_op_name(self, name: str) -> str:
    """
    Converts 'abs' -> 'Abs', 'log_plus_one' -> 'LogPlusOne'.
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
