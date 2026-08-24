"""Test extra sass reader."""

from unittest.mock import MagicMock, patch
from ml_switcheroo.importers.sass_reader import SassSpecImporter


def test_sass_reader_unicode_decode_error(tmp_path):
  """Test element."""
  html_path = tmp_path / "index.html"
  # Write invalid utf-8 byte
  html_path.write_bytes(b"\xff\xfeh\x00e\x00l\x00l\x00o\x00")  # Some UTF-16 maybe, but forces error in utf-8

  with patch("ml_switcheroo.importers.sass_reader.SassHtmlParser") as mock_parser:
    parser_instance = mock_parser.return_value
    parser_instance.instruction_map = {}

    # We need to make sure read_text(encoding="utf-8") raises UnicodeDecodeError
    with patch("pathlib.Path.read_text") as mock_read:
      mock_read.side_effect = [UnicodeDecodeError("utf-8", b"", 0, 1, "invalid start byte"), "dummy"]
      reader = SassSpecImporter()
      reader.parse_file(html_path)
      assert mock_read.call_count == 2
      mock_read.assert_called_with(encoding="latin-1")


def test_sass_reader_fp32_update():
  """Test element."""
  # Test line 173
  reader = SassSpecImporter()
  html_path = MagicMock()
  html_path.read_text.return_value = ""

  with patch("ml_switcheroo.importers.sass_reader.SassHtmlParser") as mock_parser:
    parser_instance = mock_parser.return_value
    parser_instance.extracted_ops = [("MOV", "Move integer"), ("MOV", "Move FP32 value")]

    res = reader.parse_file(html_path)
    assert "Mov" in res


def test_sass_reader_regex_substrings():
  """Test element."""
  reader = SassSpecImporter()
  # "integer" before "fp32"
  assert reader._infer_abstract_op("dummy", "convert an integer into a fp32") == "CastFloat"
  # "fp32" before "integer"
  assert reader._infer_abstract_op("dummy", "convert a fp32 into an integer") == "CastInt"
