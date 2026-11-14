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

import re
import typing
from pathlib import Path

import yaml
from pydantic import JsonValue

from ml_switcheroo.enums import RuntimeTensorOrValue
from ml_switcheroo.utils.console import log_error, log_info

OnnxParsedDef = dict[
  str,
  str | list[tuple[str, str]],
]


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

  def parse_file(self, target_file: Path) -> dict[str, OnnxParsedDef]:
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

  @staticmethod
  def _extract_h3_operator_name(inline_content: str) -> str | None:
    """Extract the anchored operator identifier from an ``<h3>`` heading inline token.

    Args:
        inline_content: Raw Markdown/HTML inline string inside ``<h3>``.

    Returns:
        Operator name from ``<a name="...">``, or ``None`` if absent.
    """
    from bs4 import BeautifulSoup, Tag

    soup = BeautifulSoup(inline_content, "html.parser")
    a_tag = soup.find("a", attrs={"name": True})
    if isinstance(a_tag, Tag) and isinstance(a_tag.get("name"), str):
      return str(a_tag["name"])
    return None

  def _extract_dl_arguments(self, html_content: str) -> list[tuple[str, str]]:
    """Parse ``<dt>`` entries inside an Inputs/Attributes HTML block into ``(name, type_hint)`` pairs.

    Args:
        html_content: HTML block string containing ``<dl>/<dt>`` definitions.

    Returns:
        Ordered list of ``(arg_name, mapped_type_hint)`` tuples.
    """
    import bs4.element
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html_content, "html.parser")
    extracted: list[tuple[str, str]] = []
    for dt in soup.find_all("dt"):
      raw_type = "Any"
      first_tag = dt.find(["tt", "b", "code", "span"])
      if first_tag:
        raw_name = first_tag.get_text().strip()
        remaining_text = "".join(str(c) for c in dt.contents if isinstance(c, bs4.element.NavigableString))
        if ":" in remaining_text:
          raw_type = remaining_text.split(":", 1)[1].strip()
      else:
        text = dt.get_text()
        if ":" in text:
          raw_name, raw_type = text.split(":", 1)
        else:
          raw_name = text

      tokens = raw_name.strip().split()
      arg_name = tokens[0] if tokens else ""
      if arg_name:
        extracted.append((arg_name, self._map_onnx_type(raw_type)))
    return extracted

  def _parse_markdown(self, fpath: Path) -> dict[str, OnnxParsedDef]:
    """Parse markdown structurally.

    Args:
        fpath: Path to markdown file.

    Returns:
        Dict mapping operator names to extracted semantic properties.
    """
    from markdown_it import MarkdownIt

    tokens = MarkdownIt().parse(fpath.read_text(encoding="utf-8"))
    semantics: dict[str, OnnxParsedDef] = {}
    op_std_args: dict[str, list[tuple[str, str]]] = {}
    raw_summaries: dict[str, list[str]] = {}
    current_op = ""
    current_section = ""
    is_in_heading = False

    for i, token in enumerate(tokens):
      if token.type == "heading_open":
        is_in_heading = True
        has_next_inline = i + 1 < len(tokens) and tokens[i + 1].type == "inline"
        if token.tag == "h3" and has_next_inline:
          op_id = self._extract_h3_operator_name(tokens[i + 1].content)
          if op_id is not None:
            current_op = op_id
            if current_op not in semantics:
              args_buf: list[tuple[str, str]] = []
              op_std_args[current_op] = args_buf
              semantics[current_op] = {"from": fpath.name, "description": "", "std_args": args_buf}
              raw_summaries[current_op] = []
            current_section = "Summary"
        elif token.tag == "h4" and has_next_inline:
          current_section = tokens[i + 1].content.strip()
      elif token.type == "heading_close":
        is_in_heading = False
      elif current_op and not is_in_heading:
        if current_section == "Summary" and token.type == "inline" and not raw_summaries[current_op]:
          text = token.content.replace("*", "").strip()
          if text and text != current_op:
            raw_summaries[current_op].append(token.content)
        elif current_section in ("Inputs", "Attributes") and (
          token.type == "html_block" or (token.type == "inline" and "<dl>" in token.content)
        ):
          op_std_args[current_op].extend(self._extract_dl_arguments(token.content))

    max_len = 300
    for op_key, op in semantics.items():
      summary = " ".join(raw_summaries[op_key]).strip()
      op["description"] = (summary[:max_len] + "...") if len(summary) > max_len else summary

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
    parsed_ops: typing.Mapping[str, typing.Mapping[str, RuntimeTensorOrValue]],
    domain: str = "ai.onnx",
    opset_version: int = 21,
  ) -> dict[str, dict[str, JsonValue]]:
    """Convert parsed ONNX specifications into valid ODL dictionary entries.

    Args:
        parsed_ops: Dictionary of operator name to raw parsed metadata.
        domain: ONNX operator domain (e.g. 'ai.onnx', 'ai.onnx.ml').
        opset_version: Target ONNX opset version.

    Returns:
        Dictionary mapping operation names to full ODL specification dictionaries.
    """
    odl_catalog: dict[str, dict[str, JsonValue]] = {}

    for op_name, op_data in parsed_ops.items():
      raw_desc = str(op_data.get("description", ""))
      clean_desc = clean_html_tags(raw_desc) or f"Standardized definition for {op_name}."

      std_args: list[JsonValue] = []
      raw_std_args = op_data.get("std_args", [])
      arg_items = raw_std_args if isinstance(raw_std_args, list) else []
      for arg in arg_items:
        if isinstance(arg, (tuple, list)) and len(arg) >= 2:
          std_args.append({"name": str(arg[0]), "type": str(arg[1])})
        elif isinstance(arg, str):
          std_args.append({"name": arg, "type": "Any"})
        elif isinstance(arg, dict) and "name" in arg:
          std_args.append({str(k): str(v) for k, v in arg.items()})

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
    odl_entries: typing.Mapping[str, typing.Mapping[str, RuntimeTensorOrValue]],
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
