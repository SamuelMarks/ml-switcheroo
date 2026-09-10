"""Tests for scripts/audit_stablehlo_spec.py."""

import json
from pathlib import Path
import sys
import tempfile
from typing import Any
from unittest.mock import MagicMock, patch
import pytest

import scripts.audit_stablehlo_spec as auditor


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
  raw_url = "https://raw.githubusercontent.com/openxla/stablehlo/main/docs/spec.md"
  expected = "https://cdn.jsdelivr.net/gh/openxla/stablehlo@main/docs/spec.md"
  assert auditor.get_fallback_raw_url(raw_url) == expected

  # Non github raw URL
  assert auditor.get_fallback_raw_url("https://example.com/spec.md") is None
  # Truncated raw github URL
  assert auditor.get_fallback_raw_url("https://raw.githubusercontent.com/openxla/stablehlo") is None


def test_fetch_raw_https() -> None:
  """Test fetch_raw_https directly."""
  mock_response = MagicMock()
  mock_response.read.return_value = b"# StableHLO Spec Content"
  mock_response.__enter__.return_value = mock_response

  with patch("urllib.request.urlopen", return_value=mock_response):
    assert auditor.fetch_raw_https("https://fake.url") == "# StableHLO Spec Content"

  with patch("urllib.request.urlopen", side_effect=Exception("network down")):
    with pytest.raises(RuntimeError, match="Failed to download spec"):
      auditor.fetch_raw_https("https://fake.url")


def test_download_spec(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  """Test download_spec caching, cache hits, fallbacks, and errors."""
  cache_dir = tmp_path / "pip_cache"
  monkeypatch.setenv("PIP_CACHE_DIR", str(cache_dir))

  mock_response = MagicMock()
  mock_response.read.return_value = b"# StableHLO Spec"
  mock_response.__enter__.return_value = mock_response

  cache_file = cache_dir / "ml_switcheroo" / "stablehlo_spec.md"

  # 1. Cache miss: downloads and writes to cache
  with patch("urllib.request.urlopen", return_value=mock_response) as mock_urlopen:
    content = auditor.download_spec("https://fake.url/spec.md")
    assert content == "# StableHLO Spec"
    assert mock_urlopen.call_count == 1
    assert cache_file.exists()
    assert cache_file.read_text(encoding="utf-8") == "# StableHLO Spec"

  # 2. Cache hit: returns cached content without calling urlopen
  with patch("urllib.request.urlopen", side_effect=AssertionError("Should not hit network")):
    cached_content = auditor.download_spec("https://fake.url/spec.md")
    assert cached_content == "# StableHLO Spec"

  # 3. Force download: bypasses cache
  mock_response_v2 = MagicMock()
  mock_response_v2.read.return_value = b"# Updated StableHLO Spec"
  mock_response_v2.__enter__.return_value = mock_response_v2
  with patch("urllib.request.urlopen", return_value=mock_response_v2):
    updated_content = auditor.download_spec("https://fake.url/spec.md", force_download=True)
    assert updated_content == "# Updated StableHLO Spec"
    assert cache_file.read_text(encoding="utf-8") == "# Updated StableHLO Spec"

  # 4. Fallback to raw CDN URL when primary URL fails
  cache_file.unlink()
  primary_url = "https://raw.githubusercontent.com/openxla/stablehlo/main/docs/spec.md"

  def _side_effect(req: Any, **kwargs: Any) -> MagicMock:
    """Mock urlopen side effect simulating CDN fallback."""
    if "cdn.jsdelivr.net" in req.full_url:
      resp = MagicMock()
      resp.read.return_value = b"# CDN StableHLO Spec"
      resp.__enter__.return_value = resp
      return resp
    raise Exception("HTTP 429 Too Many Requests")

  with patch("urllib.request.urlopen", side_effect=_side_effect):
    cdn_content = auditor.download_spec(primary_url)
    assert cdn_content == "# CDN StableHLO Spec"

  # 5. Network fails completely but cache exists: fallback to cached version
  with patch("urllib.request.urlopen", side_effect=Exception("Total network failure")):
    fallback_content = auditor.download_spec(primary_url, force_download=True)
    assert fallback_content == "# CDN StableHLO Spec"

  # 6. Both primary and fallback fail with no cache: raises RuntimeError
  cache_file.unlink()
  with patch("urllib.request.urlopen", side_effect=Exception("Total network failure")):
    with pytest.raises(RuntimeError, match="Failed to download spec"):
      auditor.download_spec(primary_url)

  # 7. Unparseable/corrupt cache file: falls back to download
  cache_file.write_text("   " + chr(10), encoding="utf-8")
  with patch("urllib.request.urlopen", return_value=mock_response):
    refreshed = auditor.download_spec(primary_url)
    assert refreshed == "# StableHLO Spec"

  # 8. Cache write failure does not break the function
  cache_file.unlink()
  with patch.object(Path, "write_text", side_effect=PermissionError("read-only")):
    with patch("urllib.request.urlopen", return_value=mock_response):
      content = auditor.download_spec("https://fake.url/spec.md")
      assert content == "# StableHLO Spec"

  # 9. Initial cache read exception proceeds to download
  cache_file.touch()
  with patch.object(Path, "read_text", side_effect=OSError("Disk read error")):
    with patch("urllib.request.urlopen", return_value=mock_response):
      res = auditor.download_spec("https://fake.url/spec.md")
      assert res == "# StableHLO Spec"

  # 10. Fallback cache read exception proceeds to raise RuntimeError
  with patch.object(Path, "read_text", side_effect=OSError("Disk read error")):
    with patch("urllib.request.urlopen", side_effect=Exception("Network down")):
      with pytest.raises(RuntimeError, match="Failed to download spec"):
        auditor.download_spec("https://fake.url/spec.md", force_download=True)

  # 11. Fallback cache exists but is empty/whitespace when network fails: raises RuntimeError
  cache_file.write_text("   \n", encoding="utf-8")
  with patch("urllib.request.urlopen", side_effect=Exception("Network down")):
    with pytest.raises(RuntimeError, match="Failed to download spec"):
      auditor.download_spec("https://fake.url/spec.md", force_download=True)


def test_parse_stablehlo_ops() -> None:
  """Test parsing operations from ### headings."""
  content = """# Spec
## Section
### abs
Description of abs.
### add
Description of add.
### not an op
"""
  ops = auditor.parse_stablehlo_ops(content)
  assert ops == ["abs", "add"]


def test_load_implemented_stablehlo_ops(tmp_path: Path) -> None:
  """Test loading implemented ops from ODL directory."""
  missing_dir = tmp_path / "missing_odl"
  assert auditor.load_implemented_stablehlo_ops(missing_dir) == set()

  odl_dir = tmp_path / "odl"
  odl_dir.mkdir()

  # 1. Valid StableHLO op
  (odl_dir / "Abs.yaml").write_text(
    """operation: Abs
variants:
  stablehlo:
    api: stablehlo.abs
""",
    encoding="utf-8",
  )

  # 2. StableHLO variant with non-stablehlo prefix
  (odl_dir / "Other.yaml").write_text(
    """operation: Other
variants:
  stablehlo:
    api: chlo.broadcast_add
""",
    encoding="utf-8",
  )

  # 3. No stablehlo variant
  (odl_dir / "TorchOnly.yaml").write_text(
    """operation: TorchOnly
variants:
  torch:
    api: torch.relu
""",
    encoding="utf-8",
  )

  # 4. Non-dict stablehlo variant
  (odl_dir / "NonDictShlo.yaml").write_text(
    """operation: NonDict
variants:
  stablehlo: not_a_dict
""",
    encoding="utf-8",
  )

  # 5. Broken / non-dict YAML
  (odl_dir / "broken.yaml").write_text("invalid: [broken yaml", encoding="utf-8")
  (odl_dir / "list.yaml").write_text(
    """- item1
- item2
""",
    encoding="utf-8",
  )

  implemented = auditor.load_implemented_stablehlo_ops(odl_dir)
  assert implemented == {"abs"}


def test_generate_reports(tmp_path: Path) -> None:
  """Test JSON and Markdown report generation."""
  json_out = tmp_path / "out.json"
  md_out = tmp_path / "out.md"

  official = ["abs", "add"]
  implemented = {"abs"}
  missing = ["add"]

  auditor.generate_reports(official, implemented, missing, json_out, md_out)
  assert json_out.exists()
  assert md_out.exists()

  with open(json_out, "r", encoding="utf-8") as f:
    data = json.load(f)
  assert data["official_ops"] == ["abs", "add"]
  assert data["implemented"] == ["abs"]
  assert data["missing"] == ["add"]

  # Without missing ops
  auditor.generate_reports(official, implemented, [], json_out, md_out)
  assert json_out.exists()


def test_main_and_entrypoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  """Test main execution with success, missing ops, and download failure."""
  import runpy

  odl_dir = tmp_path / "odl"
  odl_dir.mkdir()
  (odl_dir / "Abs.yaml").write_text(
    """operation: Abs
variants:
  stablehlo:
    api: stablehlo.abs
""",
    encoding="utf-8",
  )

  json_out = tmp_path / "report.json"
  md_out = tmp_path / "report.md"

  # 1. Success (all implemented)
  valid_content = """### abs
"""
  res_success = auditor.main(
    odl_dir=odl_dir,
    json_output=json_out,
    md_output=md_out,
    content=valid_content,
  )
  assert res_success == 0

  # 2. Missing ops (returns 1)
  missing_content = """### abs
### add
"""
  res_missing = auditor.main(
    odl_dir=odl_dir,
    json_output=json_out,
    md_output=md_out,
    content=missing_content,
  )
  assert res_missing == 1

  # 3. Download failure branch
  with patch("scripts.audit_stablehlo_spec.download_spec", side_effect=RuntimeError("fail")):
    res_fail = auditor.main(
      odl_dir=odl_dir,
      json_output=json_out,
      md_output=md_out,
      content=None,
    )
    assert res_fail == 1

  # 4. __main__ entrypoint
  with patch("scripts.audit_stablehlo_spec.main", return_value=0):
    with pytest.raises(SystemExit) as excinfo:
      runpy.run_module("scripts.audit_stablehlo_spec", run_name="__main__")
    assert excinfo.value.code == 0
