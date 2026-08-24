"""Test module."""

from pathlib import Path
from ml_switcheroo.importers.sass_reader import SassSpecImporter


def test_sass_reader_missing_file(tmp_path: Path) -> None:
  """Test element."""
  importer = SassSpecImporter()
  res = importer.parse_file(tmp_path / "missing.html")
  assert res == {}


def test_sass_reader_parse_html(tmp_path: Path) -> None:
  """Test element."""
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
  """Test element."""
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
  """Test element."""
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
