"""Test module."""

import typing
from pathlib import Path
from unittest.mock import MagicMock, patch

from ml_switcheroo.importers.sass_reader import SassSpecImporter


def test_sass_reader_missing_file(tmp_path: Path) -> None:
  """Docstring."""
  importer = SassSpecImporter()
  res = importer.parse_file(tmp_path / "missing.html")
  assert res == {}


def test_sass_reader_parse_html(tmp_path: Path) -> None:
  """Docstring."""
  importer = SassSpecImporter()

  html_content = """
<html>
<body>
    <table>
        <tbody>
            <tr>
                <td>Opcode</td>
                <td>Description</td>
            </tr>
            <tr>
                <td>FADD</td>
                <td>FP32 Add</td>
            </tr>
            <tr>
                <td>FMUL</td>
                <td>FP32 Multiply</td>
            </tr>
            <tr>
                <td>IADD3</td>
                <td>Integer Addition 3-way</td>
            </tr>
            <tr>
                <td>IABS</td>
                <td>Integer Absolute Value</td>
            </tr>
            <tr>
                <td>MNMX</td>
                <td>FP32 Minimum/Maximum</td>
            </tr>
            <tr>
                <td>CONV</td>
                <td>Convert Integer to FP32</td>
            </tr>
            <tr>
                <td>CONV2</td>
                <td>Convert FP32 to Integer</td>
            </tr>
            <tr>
                <td>LOP3</td>
                <td>Logic Operation 3-way</td>
            </tr>
            <tr>
                <td>FFMA</td>
                <td>Fused Multiply and Add</td>
            </tr>
            <tr>
                <td>IADD</td>
                <td>Integer Addition</td>
            </tr>
            <tr>
                <td>IADD</td>
                <td>FP32 Add (should override)</td>
            </tr>
        </tbody>
    </table>
</body>
</html>
    """
  html_file = tmp_path / "sass.html"
  html_file.write_text(html_content)

  res = importer.parse_file(html_file)

  assert "Add" in res
  # FADD -> Add
  assert res["Add"]["api"] == "FADD"
  assert "Mul" in res
  assert "Add3" in res
  assert "Abs" in res
  assert "MinMax" in res
  assert "CastFloat" in res
  assert "CastInt" in res
  assert "BitwiseOp" in res
  assert "FusedMultiplyAdd" in res

  # Check fallback decoding
  latin_file = tmp_path / "sass_latin.html"
  latin_file.write_bytes(html_content.encode("latin-1"))
  res_latin = importer.parse_file(latin_file)
  assert "Add" in res_latin


def test_fallback_opcodes(tmp_path: Path) -> None:
  """Docstring."""
  importer = SassSpecImporter()

  html_content = """
    <table>
        <tbody>
            <tr>
                <td>UNK</td>
                <td>Unknown description</td>
            </tr>
        </tbody>
    </table>
    """
  html_file = tmp_path / "sass2.html"
  html_file.write_text(html_content)

  res = importer.parse_file(html_file)
  assert "Unk" in res
  assert res["Unk"]["api"] == "UNK"


def test_invalid_ops(tmp_path: Path) -> None:
  """Docstring."""
  importer = SassSpecImporter()

  html_content = """
    <table>
        <tbody>
            <tr>
                <td>L O P</td>
                <td>Space not allowed</td>
            </tr>
            <tr>
                <td>lop</td>
                <td>Lowercase not allowed</td>
            </tr>
            <tr>
                <td></td>
                <td>Empty opcode</td>
            </tr>
        </tbody>
    </table>
    """
  html_file = tmp_path / "sass3.html"
  html_file.write_text(html_content)

  res = importer.parse_file(html_file)
  assert res == {}


# --- Merged from test_sass_reader_extra.py ---


def test_sass_reader_unicode_decode_error(tmp_path: Path) -> None:
  """Docstring."""
  html_path: Path = tmp_path / "index.html"
  # Write invalid utf-8 byte
  html_path.write_bytes(b"\xff\xfeh\x00e\x00l\x00l\x00o\x00")  # Some UTF-16 maybe, but forces error in utf-8

  with patch("ml_switcheroo.importers.sass_reader.SassHtmlParser") as mock_parser:
    parser_instance: typing.Any = mock_parser.return_value
    parser_instance.instruction_map = {}

    # We need to make sure read_text(encoding="utf-8") raises UnicodeDecodeError
    with patch("pathlib.Path.read_text") as mock_read:
      mock_read.side_effect = [UnicodeDecodeError("utf-8", b"", 0, 1, "invalid start byte"), "dummy"]
      reader = SassSpecImporter()
      reader.parse_file(html_path)
      assert mock_read.call_count == 2
      mock_read.assert_called_with(encoding="latin-1")


def test_sass_reader_fp32_update() -> None:
  """Docstring."""
  # Test line 173
  reader = SassSpecImporter()
  html_path = MagicMock()
  html_path.read_text.return_value = ""

  with patch("ml_switcheroo.importers.sass_reader.SassHtmlParser") as mock_parser:
    parser_instance: typing.Any = mock_parser.return_value
    parser_instance.extracted_ops = [("MOV", "Move integer"), ("MOV", "Move FP32 value")]

    res: dict[str, typing.Any] = reader.parse_file(html_path)
    assert "Mov" in res


def test_sass_reader_regex_substrings() -> None:
  """Docstring."""
  reader = SassSpecImporter()
  # "integer" before "fp32"
  assert reader._infer_abstract_op("dummy", "convert an integer into a fp32") == "CastFloat"
  # "fp32" before "integer"
  assert reader._infer_abstract_op("dummy", "convert a fp32 into an integer") == "CastInt"


def test_sass_reader_extra_branches(tmp_path: Path) -> None:
  """Docstring."""
  importer = SassSpecImporter()
  html_content = """
<table>
<tbody>
<tr>
<td>OpcodeOnly</td>
</tr>
</tbody>
</table>
    """
  html_file = tmp_path / "Extra.html"
  html_file.write_text(html_content)
  res = importer.parse_file(html_file)
  assert res == {}
