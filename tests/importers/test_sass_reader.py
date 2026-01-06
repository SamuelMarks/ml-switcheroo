"""
Tests for SASS HTML Spec Importer.

Verifies:
1. Parsing of the provided HTML snippet.
2. Extraction of FADD, FMUL, etc.
3. Filtering of headers.
4. Correct association of Opcode with Abstract Name (FADD -> Add).
"""

import pytest
from pathlib import Path
from ml_switcheroo.importers.sass_reader import SassSpecImporter

# Minimal representation of the structure provided in the prompt
MOCK_HTML_CONTENT = """
<table class="table-no-stripes longtable docutils align-default" id="turing-turing-instruction-set-table"> 
  <caption>Turing Instruction Set</caption> 
  <thead> 
    <tr class="row-odd"> 
      <th class="head"><p>Opcode</p></th> 
      <th class="head"><p>Description</p></th> 
    </tr> 
  </thead> 
  <tbody> 
    <tr class="row-even"> 
      <td colspan="2"><p><strong>Floating Point Instructions</strong></p></td> 
    </tr> 
    <tr class="row-odd"> 
      <td><p>FADD</p></td> 
      <td><p>FP32 Add</p></td> 
    </tr> 
    <tr class="row-even"> 
      <td><p>FMUL</p></td> 
      <td><p>FP32 Multiply</p></td> 
    </tr> 
    <tr class="row-odd"> 
        <td><p>IABS</p></td>
        <td><p>Integer Absolute Value</p></td>
    </tr>
    <tr class="row-even">
        <td><p>MUFU</p></td>
        <td><p>FP32 Multi Function Operation</p></td>
    </tr>
  </tbody> 
</table>
"""


@pytest.fixture
def spec_file(tmp_path):
  f = tmp_path / "sass.html"
  f.write_text(MOCK_HTML_CONTENT, encoding="utf-8")
  return f


def test_sass_reader_extraction(spec_file):
  importer = SassSpecImporter()
  mappings = importer.parse_file(spec_file)

  # Verify keys are Abstract (if inferred) or TitleCase
  assert "Add" in mappings
  assert mappings["Add"]["api"] == "FADD"

  assert "Mul" in mappings
  assert mappings["Mul"]["api"] == "FMUL"

  assert "Abs" in mappings
  assert mappings["Abs"]["api"] == "IABS"

  # Generic fallback
  assert "Mufu" in mappings
  assert mappings["Mufu"]["api"] == "MUFU"


def test_sass_reader_ignores_headers(spec_file):
  importer = SassSpecImporter()
  mappings = importer.parse_file(spec_file)

  # "Floating Point Instructions" row should be filtered
  assert "Floating Point Instructions" not in mappings
  assert "Opcode" not in mappings
