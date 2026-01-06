"""
Importer for NVIDIA SASS Documentation (HTML).

This module parses the official NVIDIA CUDA Binary Utilities documentation
(HTML format) to extract instruction set definitions. It targets tables
listing Opcodes and Descriptions (e.g., Turing, Ampere sections).

It normalizes assembly mnemonics into the ML-Switcheroo Abstract Standard options
where possible (e.g., "FP32 Add" -> "Add").
"""

from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import re

from ml_switcheroo.utils.console import log_info, log_error, log_success


class SassHtmlParser(HTMLParser):
  """
  State-machine based HTML parser for extracting SASS Instruction tables.
  """

  def __init__(self):
    super().__init__()
    self.in_table = False
    self.in_tbody = False
    self.in_row = False
    self.in_cell = False
    self.current_row_cells: List[str] = []
    self.extracted_ops: List[Tuple[str, str]] = []  # (Opcode, Description)
    self.cell_buffer = ""

  def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]):
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
    elif tag == "td" and self.in_row:
      self.in_cell = True
      self.cell_buffer = ""

  def handle_endtag(self, tag: str):
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
    elif tag == "table":
      self.in_table = False

  def handle_data(self, data: str):
    if self.in_cell:
      self.cell_buffer += data


class SassSpecImporter:
  """
  Facade for importing SASS specifications.
  """

  # Mapping logic: Regex pattern in Description -> Abstract Op Name
  _DESCRIPTION_MAP = [
    (r"FP32 Add", "Add"),
    (r"FP32 Subtract", "Sub"),  # Rare, usually FADD with neg
    (r"FP32 Multiply", "Mul"),
    (r"Integer Addition", "Add"),
    (r"Integer Multiply", "Mul"),
    (r"FP32 Minimum", "Min"),  # Handles Min/Max
    (r"FP32 Maximum", "Max"),
    (r"Convert.*Integer.*FP32", "CastFloat"),
    (r"Convert.*FP32.*Integer", "CastInt"),
    (r"Absolute Value", "Abs"),
    (r"Logic Operation", "BitwiseOp"),  # LOP/LOP3
    (r"Fused Multiply and Add", "FusedMultiplyAdd"),
  ]

  def parse_file(self, html_path: Path) -> Dict[str, Any]:
    """
    Parses an HTML file containing SASS documentation.

    Args:
        html_path: Path to the .html file.

    Returns:
        Dictionary mapping Abstract Operations to SASS implementations.
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

    mappings = {}

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
        if "FP32" in desc and "FP32" not in prev_desc:
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
    """
    Derives the Abstract Operation ID (e.g. 'Add') from the SASS text.
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
    for pattern, abstract_name in self._DESCRIPTION_MAP:
      if re.search(pattern, desc, re.IGNORECASE):
        # Ambiguity handling
        if abstract_name in ["Min", "Max"] and "Minimum/Maximum" in desc:
          # MNMX instructions handle both based on predicates/modifiers.
          # We map to a generic "MinMax" or ignore specific alignment for now.
          return "MinMax"
        return abstract_name

    return None
