"""Tests for scripts/audit_mlir_spec.py."""

from pathlib import Path
import sys
import tempfile
from typing import Any
from unittest.mock import MagicMock, patch
import pytest

import scripts.audit_mlir_spec as auditor


def test_get_pip_cache_dir_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
  """Test get_pip_cache_dir when PIP_CACHE_DIR is set."""
  custom_dir = tmp_path / "custom_pip_cache"
  monkeypatch.setenv("PIP_CACHE_DIR", str(custom_dir))
  assert auditor.get_pip_cache_dir() == custom_dir


def test_get_pip_cache_dir_platforms(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
  """Test get_pip_cache_dir across darwin, win32, and linux platforms."""
  monkeypatch.delenv("PIP_CACHE_DIR", raising=False)
  monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")

  # Darwin
  monkeypatch.setattr(sys, "platform", "darwin")
  assert auditor.get_pip_cache_dir() == tmp_path / "home" / "Library" / "Caches" / "pip"

  # Win32 with LOCALAPPDATA
  monkeypatch.setattr(sys, "platform", "win32")
  monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "appdata"))
  assert auditor.get_pip_cache_dir() == tmp_path / "appdata" / "pip" / "cache"

  # Win32 without LOCALAPPDATA
  monkeypatch.delenv("LOCALAPPDATA", raising=False)
  assert auditor.get_pip_cache_dir() == tmp_path / "home" / "AppData" / "Local" / "pip" / "cache"

  # Linux with XDG_CACHE_HOME
  monkeypatch.setattr(sys, "platform", "linux")
  monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
  assert auditor.get_pip_cache_dir() == tmp_path / "xdg" / "pip"

  # Linux without XDG_CACHE_HOME
  monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
  assert auditor.get_pip_cache_dir() == tmp_path / "home" / ".cache" / "pip"

  # Fallback on Path.home() exception
  def _raising_home() -> Path:
    """Mock Path.home raising RuntimeError."""
    raise RuntimeError("No home dir")

  monkeypatch.setattr(Path, "home", _raising_home)
  assert auditor.get_pip_cache_dir() == Path(tempfile.gettempdir()) / "pip"


def test_get_fallback_raw_url() -> None:
  """Test get_fallback_raw_url for GitHub raw URLs and invalid ones."""
  raw_url = "https://raw.githubusercontent.com/llvm/llvm-project/main/mlir/docs/LangRef.md"
  expected = "https://cdn.jsdelivr.net/gh/llvm/llvm-project@main/mlir/docs/LangRef.md"
  assert auditor.get_fallback_raw_url(raw_url) == expected

  # Non github raw URL
  assert auditor.get_fallback_raw_url("https://example.com/file.md") is None
  # Truncated raw github URL
  assert auditor.get_fallback_raw_url("https://raw.githubusercontent.com/llvm/llvm-project") is None


def test_fetch_raw_https() -> None:
  """Test fetch_raw_https directly."""
  mock_response = MagicMock()
  mock_response.read.return_value = b"# Spec Content"
  mock_response.__enter__.return_value = mock_response

  with patch("urllib.request.urlopen", return_value=mock_response):
    assert auditor.fetch_raw_https("https://fake.url") == "# Spec Content"

  with patch("urllib.request.urlopen", side_effect=Exception("network down")):
    with pytest.raises(RuntimeError, match="Failed to download spec"):
      auditor.fetch_raw_https("https://fake.url")


def test_download_spec(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  """Test download_spec caching, cache hits, fallbacks, and errors."""
  cache_dir = tmp_path / "pip_cache"
  monkeypatch.setenv("PIP_CACHE_DIR", str(cache_dir))

  mock_response = MagicMock()
  mock_response.read.return_value = b"# Downloaded Spec"
  mock_response.__enter__.return_value = mock_response

  cache_file = cache_dir / "ml_switcheroo" / "LangRef.md"

  # 1. Cache miss: downloads and writes to cache
  with patch("urllib.request.urlopen", return_value=mock_response) as mock_urlopen:
    content = auditor.download_spec("https://fake.url/LangRef.md")
    assert content == "# Downloaded Spec"
    assert mock_urlopen.call_count == 1
    assert cache_file.exists()
    assert cache_file.read_text(encoding="utf-8") == "# Downloaded Spec"

  # 2. Cache hit: returns cached content without calling urlopen
  with patch("urllib.request.urlopen", side_effect=AssertionError("Should not hit network")):
    cached_content = auditor.download_spec("https://fake.url/LangRef.md")
    assert cached_content == "# Downloaded Spec"

  # 3. Force download: bypasses cache
  mock_response_v2 = MagicMock()
  mock_response_v2.read.return_value = b"# Updated Spec"
  mock_response_v2.__enter__.return_value = mock_response_v2
  with patch("urllib.request.urlopen", return_value=mock_response_v2):
    updated_content = auditor.download_spec("https://fake.url/LangRef.md", force_download=True)
    assert updated_content == "# Updated Spec"
    assert cache_file.read_text(encoding="utf-8") == "# Updated Spec"

  # 4. Fallback to raw CDN URL when primary URL fails
  cache_file.unlink()
  primary_url = "https://raw.githubusercontent.com/llvm/llvm-project/main/mlir/docs/LangRef.md"

  def _side_effect(req: Any, **kwargs: Any) -> MagicMock:
    """Mock urlopen side effect simulating CDN fallback."""
    if "cdn.jsdelivr.net" in req.full_url:
      resp = MagicMock()
      resp.read.return_value = b"# CDN Content"
      resp.__enter__.return_value = resp
      return resp
    raise Exception("HTTP 429 Too Many Requests")

  with patch("urllib.request.urlopen", side_effect=_side_effect):
    cdn_content = auditor.download_spec(primary_url)
    assert cdn_content == "# CDN Content"

  # 5. Network fails completely but cache exists: fallback to cached version
  with patch("urllib.request.urlopen", side_effect=Exception("Total network failure")):
    fallback_content = auditor.download_spec(primary_url, force_download=True)
    assert fallback_content == "# CDN Content"

  # 6. Both primary and fallback fail with no cache: raises RuntimeError
  cache_file.unlink()
  with patch("urllib.request.urlopen", side_effect=Exception("Total network failure")):
    with pytest.raises(RuntimeError, match="Failed to download spec"):
      auditor.download_spec(primary_url)

  # 7. Unparseable/corrupt cache file: falls back to download
  cache_file.write_text("   " + chr(10), encoding="utf-8")
  with patch("urllib.request.urlopen", return_value=mock_response):
    refreshed = auditor.download_spec(primary_url)
    assert refreshed == "# Downloaded Spec"

  # 8. Cache write failure does not break the function
  cache_file.unlink()
  with patch.object(Path, "write_text", side_effect=PermissionError("read-only")):
    with patch("urllib.request.urlopen", return_value=mock_response):
      content = auditor.download_spec("https://fake.url/LangRef.md")
      assert content == "# Downloaded Spec"

  # 9. Initial cache read exception proceeds to download
  cache_file.touch()
  with patch.object(Path, "read_text", side_effect=OSError("Disk read error")):
    with patch("urllib.request.urlopen", return_value=mock_response):
      res = auditor.download_spec("https://fake.url/LangRef.md")
      assert res == "# Downloaded Spec"

  # 10. Fallback cache read exception proceeds to raise RuntimeError
  with patch.object(Path, "read_text", side_effect=OSError("Disk read error")):
    with patch("urllib.request.urlopen", side_effect=Exception("Network down")):
      with pytest.raises(RuntimeError, match="Failed to download spec"):
        auditor.download_spec("https://fake.url/LangRef.md", force_download=True)

  # 11. Fallback cache exists but is empty/whitespace when network fails: raises RuntimeError
  cache_file.write_text("   \n", encoding="utf-8")
  with patch("urllib.request.urlopen", side_effect=Exception("Network down")):
    with pytest.raises(RuntimeError, match="Failed to download spec"):
      auditor.download_spec("https://fake.url/LangRef.md", force_download=True)


def test_parse_grammar_rules() -> None:
  """Test parsing BNF rules from markdown blocks."""
  content = """# Header
```bnf
operation ::= op-result-list? (generic-operation | custom-operation)
// Comment line
ignored ::= rule // should be ignored if in IGNORED_LHS_TERMS
alternation ::= a | b
val-id ::= "%" (bare-id | decimal-literal) // trailing comment
```

```text
No ::= rules here
```

```bnf
broken_rule // missing rhs
```
"""
  rules = auditor.parse_grammar_rules(content)
  assert "operation" in rules
  assert "val-id" in rules
  assert "alternation" not in rules
  assert rules["val-id"] == '"%" (bare-id | decimal-literal)'


def test_extract_implemented_lark_rules(tmp_path: Path) -> None:
  """Test extracting rules from grammar.lark."""
  missing_lark = tmp_path / "missing.lark"
  assert auditor.extract_implemented_lark_rules(missing_lark) == set()

  lark_file = tmp_path / "grammar.lark"
  lark_file.write_text(
    """// Grammar
operation: op_result
val_id: "%" NAME
string: ESCAPED_STRING
number: INT
""",
    encoding="utf-8",
  )

  rules = auditor.extract_implemented_lark_rules(lark_file)
  assert "operation" in rules
  assert "val_id" in rules
  assert "val-id" in rules
  # Manual mapping check: string -> string-literal
  assert "string-literal" in rules
  # Manual mapping check: number -> integer-literal
  assert "integer-literal" in rules


def test_audit_mlir() -> None:
  """Test audit_mlir logic."""
  grammar_rules = {
    "operation": "spec",
    "val-id": "spec",
    "missing-rule": "spec",
  }
  implemented_rules = {"operation", "val_id"}

  implemented, missing = auditor.audit_mlir(grammar_rules, implemented_rules)
  assert "operation" in implemented
  assert "val-id" in implemented
  assert "missing-rule" in missing


def test_generate_reports(tmp_path: Path) -> None:
  """Test report generation."""
  json_out = tmp_path / "out.json"
  md_out = tmp_path / "out.md"

  grammar_rules = {"op": "syntax", "missing_op": "missing_syntax"}
  missing_rules = ["missing_op"]

  auditor.generate_reports(grammar_rules, missing_rules, json_out, md_out)
  assert json_out.exists()
  assert md_out.exists()

  # Without missing rules
  auditor.generate_reports(grammar_rules, [], json_out, md_out)
  assert json_out.exists()


def test_main_and_entrypoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  """Test main execution with success, missing rules, and error branches."""
  import runpy

  lark_file = tmp_path / "grammar.lark"
  lark_file.write_text(
    """operation: op
""",
    encoding="utf-8",
  )

  json_out = tmp_path / "report.json"
  md_out = tmp_path / "report.md"

  # 1. Success (no missing rules)
  valid_content = """```bnf
operation ::= syntax
```
"""
  res_success = auditor.main(
    lark_path=lark_file,
    json_output=json_out,
    md_output=md_out,
    content=valid_content,
  )
  assert res_success == 0

  # 2. Missing rules (returns 1)
  missing_content = """```bnf
operation ::= syntax
missing ::= syntax
```
"""
  res_missing = auditor.main(
    lark_path=lark_file,
    json_output=json_out,
    md_output=md_out,
    content=missing_content,
  )
  assert res_missing == 1

  # 3. Download failure branch
  with patch("scripts.audit_mlir_spec.download_spec", side_effect=RuntimeError("fail")):
    res_fail = auditor.main(
      lark_path=lark_file,
      json_output=json_out,
      md_output=md_out,
      content=None,
    )
    assert res_fail == 1

  # 4. __main__ entrypoint
  with patch("scripts.audit_mlir_spec.main", return_value=0):
    with pytest.raises(SystemExit) as excinfo:
      runpy.run_module("scripts.audit_mlir_spec", run_name="__main__")
    assert excinfo.value.code == 0
