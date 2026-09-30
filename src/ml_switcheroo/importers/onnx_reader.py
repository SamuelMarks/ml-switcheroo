"""Importer for ONNX Markdown Specifications.

This module parses official ONNX Operators documentation (Markdown files),
extracting operator names, summaries, inputs, and attributes. It converts
these definitions into the Operation Definition Language (ODL) format.

Key Features:
- **Markdown Splitting**: Identifies operators via structurally parsing headings and links.
- **Input & Attribute Parsing**: Extracts definitions lists (``<dl>``).
- **Type Extraction**: Parses HTML type signatures (e.g., ``<dt>x : T</dt>``)
  and maps them to standard type hints (e.g., ``Tensor``, ``int``).
- **Sanitization**: Cleans HTML tags like ``<tt>``, ``<b>`` from names.
- **ODL Generation**: Generates validated ODL dictionary mappings with domain and opset tracking.
"""

from pathlib import Path
import re
from typing import Any, Dict, List
import yaml

from ml_switcheroo.utils.console import log_error, log_info


def clean_html_tags(text: str) -> str:
  """Clean HTML tags and formatting markup from text strings.

  Args:
      text: Input string potentially containing HTML tags.

  Returns:
      Cleaned text string.
  """
  clean = re.sub(r"<[^>]+>", "", text)
  return clean.replace("*", "").strip()


class OnnxSpecImporter:
  """Parse ONNX Markdown specification files into semantic ODL structures.

  This class reads Markdown files (like `Operators.md`), identifies operator
  blocks, and parses their Inputs and Attributes sections to build a rich
  function signature including type hints and ODL dictionary mappings.
  """

  def parse_file(self, target_file: Path) -> Dict[str, Dict[str, Any]]:
    """Parse a specific ONNX Markdown file (e.g. Operators.md).

    Args:
        target_file: Path to the .md file to parse.

    Returns:
        Dictionary mapping Operator IDs (e.g., "Conv") to their semantic definition.
        The definition includes 'std_args' as a list of (name, type) tuples.
    """
    if not target_file.exists():
      log_error(f"File not found: {target_file}")
      return {}

    log_info(f"Parsing ONNX Spec: {target_file.name}...")
    return self._parse_markdown(target_file)

  def _parse_markdown(self, fpath: Path) -> Dict[str, Dict[str, Any]]:
    """Parse markdown structurally.

    Args:
        fpath: Path to markdown file.

    Returns:
        Dict mapping operator names to extracted semantic properties.
    """
    from bs4 import BeautifulSoup, Tag
    from markdown_it import MarkdownIt

    content = fpath.read_text(encoding="utf-8")
    md = MarkdownIt()
    tokens = md.parse(content)

    semantics: Dict[str, Dict[str, Any]] = {}
    current_op: str = ""
    current_section: str = ""

    for i, token in enumerate(tokens):
      if token.type == "heading_open" and token.tag == "h3":
        if i + 1 < len(tokens) and tokens[i + 1].type == "inline":
          inline_content = tokens[i + 1].content
          soup = BeautifulSoup(inline_content, "html.parser")
          a_tag = soup.find("a", attrs={"name": True})
          if isinstance(a_tag, Tag) and isinstance(a_tag.get("name"), str):
            name_val = a_tag["name"]
            current_op = str(name_val)
            if current_op not in semantics:
              semantics[current_op] = {
                "from": fpath.name,
                "description": "",
                "std_args": [],
                "_raw_summary": [],
              }
            current_section = "Summary"
      elif token.type == "heading_open" and token.tag == "h4":
        if i + 1 < len(tokens) and tokens[i + 1].type == "inline":
          current_section = tokens[i + 1].content.strip()
      elif current_op:
        if current_section == "Summary":
          if token.type == "inline":
            cast_list: Any = semantics[current_op]["_raw_summary"]
            if not cast_list:
              cast_list.append(token.content)
        elif current_section in ("Inputs", "Attributes"):
          if token.type == "html_block" or (token.type == "inline" and "<dl>" in token.content):
            soup = BeautifulSoup(token.content, "html.parser")
            dts = soup.find_all("dt")
            for dt in dts:
              text = dt.get_text()
              if ":" in text:
                raw_name, raw_type = text.split(":", 1)
              else:
                raw_name = text
                raw_type = "Any"
              arg_name = raw_name.strip().split()[0] if raw_name.strip().split() else ""
              if arg_name:
                type_hint = self._map_onnx_type(raw_type)
                std_args: Any = semantics[current_op]["std_args"]
                std_args.append((arg_name, type_hint))

    for op in semantics.values():
      if "_raw_summary" in op:
        summary = " ".join(op["_raw_summary"]).strip()
        max_len = 300
        op["description"] = (summary[:max_len] + "...") if len(summary) > max_len else summary
        del op["_raw_summary"]

    return semantics

  def _map_onnx_type(self, raw_type: str) -> str:
    """Map ONNX Markdown type strings to Python/Fuzzer compatible hints.

    Examples:
        'T' -> 'Tensor'
        'list of ints' -> 'List[int]'
        'bool' -> 'bool'

    Args:
        raw_type: The string extracted right of the colon (e.g. "list of ints").

    Returns:
        A normalized type string.
    """
    raw = raw_type.lower().strip()

    # Lists
    if "list" in raw and ("int" in raw or "ints" in raw):
      return "List[int]"
    if "list" in raw and ("float" in raw or "floats" in raw):
      return "List[float]"
    if "list" in raw and ("string" in raw or "strings" in raw):
      return "List[str]"
    if "ints" in raw:
      return "List[int]"
    if "floats" in raw:
      return "List[float]"

    # Primitives
    if "string" in raw or "str" in raw:
      return "str"
    if "bool" in raw:
      return "bool"
    if "float" in raw:
      return "float"
    if "int" in raw:
      return "int"

    # Tensors
    if "tensor" in raw or raw == "t":
      return "Tensor"

    # Fallback
    return "Any"

  def convert_to_odl(
    self,
    parsed_ops: Dict[str, Dict[str, Any]],
    domain: str = "ai.onnx",
    opset_version: int = 21,
  ) -> Dict[str, Dict[str, Any]]:
    """Convert parsed ONNX specifications into valid ODL dictionary entries.

    Args:
        parsed_ops: Dictionary of operator name to raw parsed metadata.
        domain: ONNX operator domain (e.g. 'ai.onnx', 'ai.onnx.ml').
        opset_version: Target ONNX opset version.

    Returns:
        Dictionary mapping operation names to full ODL specification dictionaries.
    """
    odl_catalog: Dict[str, Dict[str, Any]] = {}

    for op_name, op_data in parsed_ops.items():
      raw_desc = op_data.get("description", "")
      clean_desc = clean_html_tags(raw_desc) or f"Standardized definition for {op_name}."

      std_args: List[Dict[str, Any]] = []
      for arg in op_data.get("std_args", []):
        if isinstance(arg, (tuple, list)) and len(arg) >= 2:
          std_args.append({"name": str(arg[0]), "type": str(arg[1])})
        elif isinstance(arg, str):
          std_args.append({"name": arg, "type": "Any"})
        elif isinstance(arg, dict) and "name" in arg:
          std_args.append(arg)

      odl_catalog[op_name] = {
        "operation": op_name,
        "description": clean_desc,
        "std_args": std_args,
        "variants": {
          "onnx": {
            "api": f"{domain}.{op_name}",
            "domain": domain,
            "opset_version": opset_version,
          }
        },
      }

    return odl_catalog

  def export_odl_yamls(
    self,
    odl_entries: Dict[str, Dict[str, Any]],
    out_dir: Path,
  ) -> int:
    """Export ODL operation definitions into discrete YAML files in out_dir.

    Args:
        odl_entries: Dictionary of operation definitions.
        out_dir: Destination directory path.

    Returns:
        Number of YAML files successfully written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    for op_name, entry in odl_entries.items():
      safe_name = op_name.replace("/", "_")
      target_file = out_dir / f"{safe_name}.yaml"
      with open(target_file, "w", encoding="utf-8") as f:
        yaml.dump(entry, f, sort_keys=False, indent=2)
      count += 1

    return count
