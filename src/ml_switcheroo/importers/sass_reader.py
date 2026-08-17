"""Importer for NVIDIA SASS Documentation (HTML).

This module parses the official NVIDIA CUDA Binary Utilities documentation
(HTML format) to extract instruction set definitions. It targets tables
listing Opcodes and Descriptions (e.g., Turing, Ampere sections).

It normalizes assembly mnemonics into the ML-Switcheroo Abstract Standard options
where possible (e.g., "FP32 Add" -> "Add").
"""

from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ml_switcheroo.utils.console import log_info, log_error, log_success


class SassHtmlParser(HTMLParser):
  """State-machine based HTML parser for extracting SASS Instruction tables."""

  def __init__(self) -> None:
    """Initializes the HTML parser state and buffers.

    Sets up boolean flags for tracking parser state within table structures,
    an empty buffer for cell text, and an empty list to accumulate extracted
    SASS operations.
    """
    super().__init__()
    self.in_table = False
    self.in_tbody = False
    self.in_row = False
    self.in_cell = False
    self.current_row_cells: List[str] = []
    self.extracted_ops: List[Tuple[str, str]] = []  # (Opcode, Description)  # type: ignore
    self.cell_buffer = ""

  def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
    """Handles the start of an HTML tag.

    Tracks entry into tables, table bodies, rows, and cells to selectively
    extract data from relevant instruction tables.

    Args:
        tag: The name of the HTML tag being processed (e.g., 'table', 'tr', 'td').
        attrs: A list of (name, value) tuples containing the tag's attributes.
    """
    if tag == "table":
      # Heuristic: Check if table has a summary or class indicating instruction set
      # The provided HTML uses generic classes like 'table-no-stripes', so we grab all
      # and filter later based on content headers.
      self.in_table = True
    elif tag == "tbody" and self.in_table:
      self.in_tbody = True
    elif tag == "tr" and self.in_tbody:
      self.in_row = True
      self.current_row_cells = []
    elif tag == "td" and self.in_row:  # pragma: no branch
      self.in_cell = True
      self.cell_buffer = ""

  def handle_endtag(self, tag: str) -> None:
    """Handles the closing of an HTML tag.

    Processes cell contents on closed 'td' tags, and extracts and filters
    SASS instruction opcodes and descriptions upon closing 'tr' tags.

    Args:
        tag: The name of the HTML tag being closed (e.g., 'td', 'tr').
    """
    if tag == "td" and self.in_cell:
      self.in_cell = False
      # Clean up content (remove newlines, extra spaces)
      clean_text = " ".join(self.cell_buffer.split())
      self.current_row_cells.append(clean_text)
    elif tag == "tr" and self.in_row:
      self.in_row = False
      # We expect typically 2 columns: Opcode, Description
      # Sometimes 3 columns in other doc versions, but the provided spec is 2.
      # We filter for rows that look like instruction definitions.
      if len(self.current_row_cells) >= 2:
        opcode = self.current_row_cells[0]
        desc = self.current_row_cells[1]
        # Filter out headers that might be in tbody or empty rows
        if opcode and desc and not opcode.lower().startswith("opcode"):
          # Check if Opcode looks like a mnemonic (UPPERCASE, no spaces usually)
          if " " not in opcode and opcode.isupper():
            self.extracted_ops.append((opcode, desc))
    elif tag == "tbody":
      self.in_tbody = False
    elif tag == "table":  # pragma: no branch
      self.in_table = False

  def handle_data(self, data: str) -> None:
    """Handles textual data inside HTML tags.

    Accumulates text content within active table cells into the cell buffer.

    Args:
        data: The text data encountered inside an HTML element.
    """
    if self.in_cell:
      self.cell_buffer += data


class SassSpecImporter:
  """Facade for importing, parsing, and normalizing SASS specifications from HTML files."""

  # Mapping logic: Substring pattern in Description -> Abstract Op Name
  _DESCRIPTION_MAP = [
    ("FP32 Add", "Add"),
    ("FP32 Subtract", "Sub"),  # Rare, usually FADD with neg
    ("FP32 Multiply", "Mul"),
    ("Integer Addition", "Add"),
    ("Integer Multiply", "Mul"),
    ("FP32 Minimum", "Min"),  # Handles Min/Max
    ("FP32 Maximum", "Max"),
    ("Convert Integer to FP32", "CastFloat"),
    ("Convert FP32 to Integer", "CastInt"),
    ("Absolute Value", "Abs"),
    ("Logic Operation", "BitwiseOp"),  # LOP/LOP3
    ("Fused Multiply and Add", "FusedMultiplyAdd"),
  ]

  def parse_file(self, html_path: Path) -> Dict[str, Any]:
    """Parses an HTML file containing SASS documentation.

    Reads the specified HTML documentation file, parses the SASS instruction
    tables, normalizes/maps the operations to abstract operation identifiers,
    and returns a structured dictionary of the extracted instructions.

    Args:
        html_path: Path to the .html file containing SASS documentation.

    Returns:
        A dictionary mapping abstract operation names or TitleCase mnemonics to
        their corresponding SASS instruction representation (with 'api' keys).
    """
    if not html_path.exists():
      log_error(f"File not found: {html_path}")
      return {}

    log_info(f"Parsing SASS Spec: {html_path.name}...")

    try:
      content = html_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
      # Fallback for some windows encodings
      content = html_path.read_text(encoding="latin-1")

    parser = SassHtmlParser()
    parser.feed(content)

    mappings = {}  # type: ignore

    for opcode, desc in parser.extracted_ops:
      abstract_id = self._infer_abstract_op(opcode, desc)

      # If we identified a mapping, add it.
      # If the mnemonic exists but no mapping, we assume it's a specific SASS op
      # and verify if it matches an internal standard name.

      key = abstract_id if abstract_id else opcode.title()  # Default to TitleCase mnemonic

      entry = {
        "api": opcode,
        "_description": desc,  # Store raw desc for context
      }

      # Conflict resolution: Prefer FP32 versions for generic math ops if collisions occur
      if key in mappings:
        prev_desc = mappings[key]["_description"]
        if "FP32" in desc and "FP32" not in prev_desc:  # pragma: no branch
          mappings[key] = entry
      else:
        mappings[key] = entry

    log_success(f"Extracted {len(mappings)} SASS instructions.")

    # Clean up internal keys
    final_map = {}
    for k, v in mappings.items():
      final_map[k] = {"api": v["api"]}

    return final_map

  def _infer_abstract_op(self, opcode: str, desc: str) -> Optional[str]:
    """Derives the Abstract Operation ID (e.g., 'Add') from the SASS text.

    Uses mnemonic rules and substring/pattern analysis of the instruction description
    to map raw SASS operations to standard ML-Switcheroo abstract operations.

    Args:
        opcode: The raw SASS opcode string (e.g., 'FADD', 'FMUL').
        desc: The instruction's description text containing functionality details.

    Returns:
        The matched Abstract Operation string (e.g., 'Add', 'Mul', 'CastFloat'),
        or None if no standard mapping could be inferred.
    """
    # 1. Check Mnemonics directly
    if opcode == "FADD":
      return "Add"
    if opcode == "FMUL":
      return "Mul"
    if opcode == "IADD3":
      return "Add3"
    if opcode == "IABS":
      return "Abs"

    # 2. Check Description Patterns
    desc_lower = desc.lower()

    for pattern, abstract_name in self._DESCRIPTION_MAP:
      if pattern.lower() in desc_lower:
        # Ambiguity handling
        if abstract_name in ["Min", "Max"] and "minimum/maximum" in desc_lower:
          # MNMX instructions handle both based on predicates/modifiers.
          # We map to a generic "MinMax" or ignore specific alignment for now.
          return "MinMax"

        # Additional checks for "Convert.*Integer.*FP32" which we changed to exact string
        # Let's keep it robust for the test cases if they use slightly different phrasing
        return abstract_name

    # Handle the specific regex cases that were modified to substrings:
    if "convert" in desc_lower and "integer" in desc_lower and "fp32" in desc_lower:
      if desc_lower.find("integer") < desc_lower.find("fp32"):
        return "CastFloat"
      else:
        return "CastInt"

    return None
