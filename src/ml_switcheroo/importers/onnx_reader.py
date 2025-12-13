"""
Importer for ONNX Markdown Specifications.

This module parses the official ONNX Operators documentation (Markdown files),
extracting operator names, summaries, inputs, and attributes. It converts
these definitions into the Semantic Knowledge Base format.

Key Features:
- **Markdown Splitting**: Identifies operators via specific HTML anchors.
- **Input & Attribute Parsing**: Extracts definitions lists (``<dl>``).
- **Type Extraction**: Parses HTML type signatures (e.g., ``<dt>x : T</dt>``)
  and maps them to ml-switcheroo/Fuzzer compatible type hints (e.g., ``Tensor``, ``int``).
- **Sanitization**: Cleans HTML tags like ``<tt>``, ``<b>`` from names.
"""

import re
from pathlib import Path
from typing import Dict, Any, List, Tuple
from ml_switcheroo.utils.console import log_info, log_error


class OnnxSpecImporter:
  """
  Parses ONNX Markdown specification files into semantic JSON structures.

  This class reads Markdown files (like `Operators.md`), identifies operator
  blocks, and parses their Inputs and Attributes sections to build a rich
  function signature including type hints.
  """

  def parse_file(self, target_file: Path) -> Dict[str, Any]:
    """
    Parses a specific ONNX Markdown file (e.g. Operators.md).

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

  def _parse_markdown(self, fpath: Path) -> Dict[str, Any]:
    """
    Splits the markdown content by operator anchors and extracts metadata.

    Args:
        fpath: Path object to read.

    Returns:
        Serialized knowledge graph dictionary.
    """
    content = fpath.read_text(encoding="utf-8")
    semantics = {}

    # Regex explanation:
    # ONNX docs use this anchor format for every operator header:
    # ### <a name="OpName"></a> ... **OpName**
    op_chunks = re.split(r'### <a name="([a-zA-Z0-9_]+)"></a>', content)

    # The split returns: [Preamble, Name1, Body1, Name2, Body2, ...]
    # We start at index 1 and jump by 2
    for i in range(1, len(op_chunks), 2):
      op_name = op_chunks[i]
      body_text = op_chunks[i + 1]

      # If an op appears multiple times (versions), keep the first occurrence
      # (assuming top-down document order prioritizes latest or canonical info).
      if op_name in semantics:
        continue

      summary = self._extract_summary(body_text)

      # Extract both Inputs (tensors) and Attributes (hyperparameters)
      # merging them into a unified argument list for the function signature.
      inputs = self._extract_section_keys(body_text, "Inputs")
      attrs = self._extract_section_keys(body_text, "Attributes")

      semantics[op_name] = {"from": fpath.name, "description": summary, "std_args": inputs + attrs}

    return semantics

  def _extract_summary(self, text: str) -> str:
    """
    Extracts the first paragraph describing the operator.

    Args:
        text: The full markdown body for a specific operator.

    Returns:
        A truncated string summary.
    """
    lines = text.strip().splitlines()

    summary = []
    for line in lines:
      line = line.strip()
      if not line:
        continue

      # Stop if we hit a sub-header (Inputs, Attributes, Constraints, etc)
      if line.startswith("####"):
        break

      # Skip the <a name=...> line or bolded title if it lingers
      if line.startswith("<a") or line.startswith("**"):
        continue

      summary.append(line)

    # Join and truncate if too long
    full_text = " ".join(summary)
    max_len = 300
    return (full_text[:max_len] + "...") if len(full_text) > max_len else full_text

  def _extract_section_keys(self, text: str, header_name: str) -> List[Tuple[str, str]]:
    """
    Parses definition lists under a specific markdown header including types.

    Used for both 'Inputs' and 'Attributes'.
    Structure matches standard HTML definition lists:
        #### Header
        <dl>
        <dt><tt>arg_name</tt> : type</dt>
        ...

    Args:
        text: The full operator markdown body.
        header_name: The section title to search for (e.g., "Inputs", "Attributes").

    Returns:
        List of (name, type) tuples found in that section.
    """
    args = []
    header_marker = f"#### {header_name}"

    # Robust finder for the section
    if header_marker not in text:
      return []

    # Grab text between '#### Header' and the next '####' (or end of string)
    section_content = text.split(header_marker)[1]

    # If there is another header later, cut off there
    if "####" in section_content:
      section_content = section_content.split("####")[0]

    for line in section_content.splitlines():
      line = line.strip()

      # Logic: Match <dt>TAG</dt> or <dt>TAG : type</dt>
      # ONNX often wraps names in <tt>...</tt> inside the <dt>
      if line.startswith("<dt>"):
        # 1. Strip outer Definition Term tags
        clean = re.sub(r"</?dt>", "", line)

        # 2. Split on colon (Name : Type)
        if ":" in clean:
          parts = clean.split(":", 1)
          raw_name = parts[0]
          raw_type = parts[1]
        else:
          raw_name = clean
          raw_type = "Any"

        # 3. Clean common HTML formatting inside the name part
        # Remove <tt>, </tt>, bolding, backticks
        clean_name = (
          raw_name.replace("<tt>", "")
          .replace("</tt>", "")
          .replace("*", "")
          .replace("`", "")
          .replace("<b>", "")
          .replace("</b>", "")
          .strip()
        )

        # 4. Take the first valid word
        # (Some lines might be "<dt>X, Y, Z</dt>" but extraction needs singular args)
        arg_name = clean_name.split(" ")[0]

        if arg_name:
          type_hint = self._map_onnx_type(raw_type)
          args.append((arg_name, type_hint))

    return args

  def _map_onnx_type(self, raw_type: str) -> str:
    """
    Maps ONNX Markdown type strings to Python/Fuzzer compatible hints.

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
    if "ints" in raw:  # Common shorthand
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
    # "T", "tensor", "tensor(T)", "tensor(float)"
    if "tensor" in raw or raw == "t":
      return "Tensor"

    # Fallback
    return "Any"
