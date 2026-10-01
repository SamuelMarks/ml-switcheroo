"""Unit tests for scripts/audit_stablehlo_spec.py StableHLO specification auditor."""

from pathlib import Path
import runpy
import tempfile
from unittest.mock import MagicMock, patch
import pytest

from scripts.audit_stablehlo_spec import (
  download_spec,
  fetch_raw_https,
  generate_reports,
  get_fallback_raw_url,
  get_pip_cache_dir,
  load_implemented_stablehlo_ops,
  main,
  parse_stablehlo_ops,
)


def test_get_pip_cache_dir_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
  """Test get_pip_cache_dir when PIP_CACHE_DIR environment variable is set.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  monkeypatch.setenv("PIP_CACHE_DIR", "/custom/pip/cache")
  assert get_pip_cache_dir() == Path("/custom/pip/cache")


def test_get_pip_cache_dir_home_exception(monkeypatch: pytest.MonkeyPatch) -> None:
  """Test get_pip_cache_dir fallback when Path.home() raises an exception.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  monkeypatch.delenv("PIP_CACHE_DIR", raising=False)
  with patch("pathlib.Path.home", side_effect=RuntimeError("No home dir")):
    cache_dir = get_pip_cache_dir()
    assert str(cache_dir).startswith(tempfile.gettempdir())


def test_get_pip_cache_dir_platforms(monkeypatch: pytest.MonkeyPatch) -> None:
  """Test get_pip_cache_dir across darwin, win32, and linux platforms.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  monkeypatch.delenv("PIP_CACHE_DIR", raising=False)
  fake_home = Path("/Users/fakeuser")
  with patch("pathlib.Path.home", return_value=fake_home):
    # 1. Darwin
    with patch("sys.platform", "darwin"):
      assert get_pip_cache_dir() == fake_home / "Library" / "Caches" / "pip"

    # 2. Win32 with LOCALAPPDATA
    with patch("sys.platform", "win32"):
      monkeypatch.setenv("LOCALAPPDATA", r"C:\AppData\Local")
      assert get_pip_cache_dir() == Path(r"C:\AppData\Local") / "pip" / "cache"

      monkeypatch.delenv("LOCALAPPDATA", raising=False)
      assert get_pip_cache_dir() == fake_home / "AppData" / "Local" / "pip" / "cache"

    # 3. Linux / generic with XDG_CACHE_HOME
    with patch("sys.platform", "linux"):
      monkeypatch.setenv("XDG_CACHE_HOME", "/custom/xdg")
      assert get_pip_cache_dir() == Path("/custom/xdg") / "pip"

      monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
      assert get_pip_cache_dir() == fake_home / ".cache" / "pip"


def test_get_fallback_raw_url() -> None:
  """Test URL transformation to jsdelivr CDN fallback."""
  github_url = "https://raw.githubusercontent.com/openxla/stablehlo/main/docs/spec.md"
  expected_fallback = "https://cdn.jsdelivr.net/gh/openxla/stablehlo@main/docs/spec.md"
  assert get_fallback_raw_url(github_url) == expected_fallback

  # Non-matching URL
  assert get_fallback_raw_url("https://example.com/spec.md") is None

  # Malformed raw github URL with insufficient segments
  assert get_fallback_raw_url("https://raw.githubusercontent.com/short") is None


def test_fetch_raw_https() -> None:
  """Test fetching content via HTTPS and handling errors."""
  mock_response = MagicMock()
  mock_response.__enter__.return_value.read.return_value = b"# Sample Spec Content"

  with patch("urllib.request.urlopen", return_value=mock_response):
    content = fetch_raw_https("https://example.com/spec.md")
    assert content == "# Sample Spec Content"

  with patch("urllib.request.urlopen", side_effect=Exception("Connection refused")):
    with pytest.raises(RuntimeError, match="Failed to download spec"):
      fetch_raw_https("https://example.com/spec.md")


def test_download_spec_from_valid_cache(tmp_path: Path) -> None:
  """Test download_spec returns cached content when present.

  Args:
      tmp_path: Temporary directory fixture from pytest.
  """
  cache_file = tmp_path / "stablehlo_spec.md"
  cache_file.write_text("# Cached Spec", encoding="utf-8")

  result = download_spec("https://example.com/spec.md", cache_path=cache_file, force_download=False)
  assert result == "# Cached Spec"


def test_download_spec_cache_read_exception(tmp_path: Path) -> None:
  """Test download_spec when reading existing cache raises an exception.

  Args:
      tmp_path: Temporary directory fixture from pytest.
  """
  cache_file = tmp_path / "stablehlo_spec.md"
  cache_file.touch()

  with patch.object(Path, "read_text", side_effect=OSError("Read error")):
    with patch("scripts.audit_stablehlo_spec.fetch_raw_https", return_value="# Downloaded"):
      res = download_spec("https://example.com/spec.md", cache_path=cache_file)
      assert res == "# Downloaded"


def test_download_spec_cache_read_exception_and_fetch(tmp_path: Path) -> None:
  """Test download_spec when cache file is whitespace and forces re-download.

  Args:
      tmp_path: Temporary directory fixture from pytest.
  """
  cache_file = tmp_path / "stablehlo_spec.md"
  cache_file.write_text("   ", encoding="utf-8")

  with patch("scripts.audit_stablehlo_spec.fetch_raw_https", return_value="# Downloaded Spec"):
    result = download_spec("https://example.com/spec.md", cache_path=cache_file, force_download=False)
    assert result == "# Downloaded Spec"
    assert cache_file.read_text(encoding="utf-8") == "# Downloaded Spec"


def test_download_spec_primary_fails_fallback_succeeds(tmp_path: Path) -> None:
  """Test download_spec falls back to CDN when primary URL fails.

  Args:
      tmp_path: Temporary directory fixture from pytest.
  """
  cache_file = tmp_path / "stablehlo_spec.md"
  primary_url = "https://raw.githubusercontent.com/openxla/stablehlo/main/docs/spec.md"

  def fetch_side_effect(url: str) -> str:
    """Mock fetch side effect simulating primary failure and fallback success.

    Args:
        url: Requested URL.

    Returns:
        Downloaded content string.

    Raises:
        RuntimeError: If primary URL is requested.
    """
    if "raw.githubusercontent.com" in url:
      raise RuntimeError("Rate limited")
    return "# CDN Fallback Spec"

  with patch("scripts.audit_stablehlo_spec.fetch_raw_https", side_effect=fetch_side_effect):
    result = download_spec(primary_url, cache_path=cache_file)
    assert result == "# CDN Fallback Spec"
    assert cache_file.read_text(encoding="utf-8") == "# CDN Fallback Spec"


def test_download_spec_primary_and_fallback_fail(tmp_path: Path) -> None:
  """Test download_spec when both primary URL and fallback URL raise exceptions.

  Args:
      tmp_path: Temporary directory fixture from pytest.
  """
  cache_file = tmp_path / "stablehlo_spec.md"
  primary_url = "https://raw.githubusercontent.com/openxla/stablehlo/main/docs/spec.md"

  with patch("scripts.audit_stablehlo_spec.fetch_raw_https", side_effect=RuntimeError("Both failed")):
    with pytest.raises(RuntimeError, match="Failed to download spec"):
      download_spec(primary_url, cache_path=cache_file, force_download=True)


def test_download_spec_write_cache_exception(tmp_path: Path) -> None:
  """Test download_spec when writing to cache fails with an exception.

  Args:
      tmp_path: Temporary directory fixture from pytest.
  """
  file_as_parent = tmp_path / "file_blocker"
  file_as_parent.write_text("blocking", encoding="utf-8")
  cache_file = file_as_parent / "stablehlo_spec.md"
  with patch("scripts.audit_stablehlo_spec.fetch_raw_https", return_value="# Downloaded Content"):
    res = download_spec("https://example.com/spec.md", cache_path=cache_file)
    assert res == "# Downloaded Content"


def test_download_spec_all_downloads_fail_with_fallback_cache(tmp_path: Path) -> None:
  """Test download_spec uses existing cache when all network requests fail.

  Args:
      tmp_path: Temporary directory fixture from pytest.
  """
  cache_file = tmp_path / "stablehlo_spec.md"
  cache_file.write_text("# Previous Cache", encoding="utf-8")

  with patch("scripts.audit_stablehlo_spec.fetch_raw_https", side_effect=RuntimeError("Network down")):
    result = download_spec("https://example.com/spec.md", cache_path=cache_file, force_download=True)
    assert result == "# Previous Cache"


def test_download_spec_fallback_cache_read_exception(tmp_path: Path) -> None:
  """Test download_spec when fallback cache read raises an exception.

  Args:
      tmp_path: Temporary directory fixture from pytest.
  """
  cache_file = tmp_path / "stablehlo_spec.md"
  cache_file.touch()

  with patch("scripts.audit_stablehlo_spec.fetch_raw_https", side_effect=RuntimeError("Network down")):
    with patch.object(Path, "read_text", side_effect=OSError("Disk error")):
      with pytest.raises(RuntimeError, match="Failed to download spec"):
        download_spec("https://example.com/spec.md", cache_path=cache_file, force_download=True)


def test_download_spec_empty_fallback_cache_raises(tmp_path: Path) -> None:
  """Test download_spec raises when fallback cache exists but is empty or whitespace.

  Args:
      tmp_path: Temporary directory fixture from pytest.
  """
  cache_file = tmp_path / "stablehlo_spec.md"
  cache_file.write_text("   " + chr(10), encoding="utf-8")

  with patch("scripts.audit_stablehlo_spec.fetch_raw_https", side_effect=RuntimeError("Network down")):
    with pytest.raises(RuntimeError, match="Failed to download spec"):
      download_spec("https://example.com/spec.md", cache_path=cache_file, force_download=True)


def test_download_spec_total_failure(tmp_path: Path) -> None:
  """Test download_spec raises RuntimeError when network fails and no cache exists.

  Args:
      tmp_path: Temporary directory fixture from pytest.
  """
  cache_file = tmp_path / "non_existent.md"
  with patch("scripts.audit_stablehlo_spec.fetch_raw_https", side_effect=RuntimeError("Offline")):
    with pytest.raises(RuntimeError, match="Failed to download spec"):
      download_spec("https://example.com/spec.md", cache_path=cache_file, force_download=True)


def test_download_spec_default_cache_path() -> None:
  """Test download_spec when cache_path is omitted and URL has non-spec.md name."""
  with patch("scripts.audit_stablehlo_spec.fetch_raw_https", return_value="# Spec"):
    with patch.object(Path, "exists", return_value=False):
      with patch.object(Path, "write_text"):
        res = download_spec("https://example.com/other.md")
        assert res == "# Spec"


def test_parse_stablehlo_ops() -> None:
  """Test parsing operation names from markdown headings."""
  content = (
    "# StableHLO Spec"
    + chr(10)
    + chr(10)
    + "### add"
    + chr(10)
    + "Performs addition."
    + chr(10)
    + chr(10)
    + "### multiply"
    + chr(10)
    + "Performs multiplication."
    + chr(10)
    + chr(10)
    + "### NotAnOpBecauseUpper"
    + chr(10)
    + "## NotH3"
    + chr(10)
  )
  ops = parse_stablehlo_ops(content)
  assert ops == ["add", "multiply"]


def test_load_implemented_stablehlo_ops(tmp_path: Path) -> None:
  """Test loading implemented StableHLO operations from ODL directory.

  Args:
      tmp_path: Temporary directory fixture from pytest.
  """
  # 1. Non-existent directory returns empty set
  non_existent = tmp_path / "no_odl"
  assert load_implemented_stablehlo_ops(non_existent) == set()

  # 2. Valid ODL directory with yaml files
  odl_dir = tmp_path / "odl"
  odl_dir.mkdir()

  valid_yaml = (
    "name: Add" + chr(10) + "variants:" + chr(10) + "  stablehlo:" + chr(10) + "    api: stablehlo.add" + chr(10)
  )
  (odl_dir / "add.yaml").write_text(valid_yaml, encoding="utf-8")

  non_shlo_yaml = "name: Sub" + chr(10) + "variants:" + chr(10) + "  torch:" + chr(10) + "    api: torch.sub" + chr(10)
  (odl_dir / "sub.yaml").write_text(non_shlo_yaml, encoding="utf-8")

  invalid_yaml = "invalid: yaml: content: :"
  (odl_dir / "bad.yaml").write_text(invalid_yaml, encoding="utf-8")

  # Non-dict YAML (list)
  (odl_dir / "list.yaml").write_text("- item1" + chr(10) + "- item2" + chr(10), encoding="utf-8")

  # variants is not a dict
  (odl_dir / "var_not_dict.yaml").write_text("variants: 'string'" + chr(10), encoding="utf-8")

  # stablehlo variant is not a dict
  (odl_dir / "shlo_not_dict.yaml").write_text("variants:" + chr(10) + "  stablehlo: 'string'" + chr(10), encoding="utf-8")

  # api does not start with stablehlo.
  (odl_dir / "api_no_prefix.yaml").write_text(
    "variants:" + chr(10) + "  stablehlo:" + chr(10) + "    api: other.add" + chr(10), encoding="utf-8"
  )

  implemented = load_implemented_stablehlo_ops(odl_dir)
  assert "add" in implemented
  assert "sub" not in implemented


def test_generate_reports(tmp_path: Path) -> None:
  """Test generating JSON and Markdown audit reports with missing and fully implemented states.

  Args:
      tmp_path: Temporary directory fixture from pytest.
  """
  official = ["add", "multiply"]
  implemented = {"add"}
  missing = ["multiply"]

  json_out = tmp_path / "report.json"
  md_out = tmp_path / "report.md"

  # Case 1: With missing operations
  generate_reports(official, implemented, missing, json_out, md_out)
  assert json_out.exists()
  assert md_out.exists()

  md_content = md_out.read_text(encoding="utf-8")
  assert "## Missing Operations" in md_content
  assert "- [ ] Implement `multiply`" in md_content

  # Case 2: Fully implemented (empty missing list)
  generate_reports(official, {"add", "multiply"}, [], json_out, md_out)
  md_all_content = md_out.read_text(encoding="utf-8")
  assert "## Supported StableHLO Operations" in md_all_content
  assert "- [x] `add`" in md_all_content
  assert "- [x] `multiply`" in md_all_content


def test_main_execution(tmp_path: Path) -> None:
  """Test main execution function in both success and missing rule scenarios.

  Args:
      tmp_path: Temporary directory fixture from pytest.
  """
  odl_dir = tmp_path / "odl"
  odl_dir.mkdir()
  (odl_dir / "add.yaml").write_text(
    "variants:" + chr(10) + "  stablehlo:" + chr(10) + "    api: stablehlo.add" + chr(10),
    encoding="utf-8",
  )

  json_out = tmp_path / "audit.json"
  md_out = tmp_path / "audit.md"

  content_all_match = "### add" + chr(10)

  # Case 1: All ops implemented -> returns 0
  exit_code_0 = main(
    odl_dir=odl_dir,
    json_output=json_out,
    md_output=md_out,
    content=content_all_match,
  )
  assert exit_code_0 == 0

  # Case 2: Missing ops -> returns 1
  content_missing = "### add" + chr(10) + "### missing_op" + chr(10)
  exit_code_1 = main(
    odl_dir=odl_dir,
    json_output=json_out,
    md_output=md_out,
    content=content_missing,
  )
  assert exit_code_1 == 1

  # Case 3: Download failure handling -> returns 1
  with patch("scripts.audit_stablehlo_spec.download_spec", side_effect=RuntimeError("Download failed")):
    exit_code_fail = main(
      odl_dir=odl_dir,
      json_output=json_out,
      md_output=md_out,
      content=None,
    )
    assert exit_code_fail == 1


def test_main_entrypoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  """Test executing scripts/audit_stablehlo_spec.py as __main__ module.

  Args:
      tmp_path: Temporary directory fixture from pytest.
      monkeypatch: Pytest monkeypatch fixture.
  """
  repo_root = Path(__file__).resolve().parent.parent
  script_path = repo_root / "scripts" / "audit_stablehlo_spec.py"

  monkeypatch.chdir(tmp_path)
  monkeypatch.setattr("sys.argv", ["scripts/audit_stablehlo_spec.py"])
  with patch("scripts.audit_stablehlo_spec.main", return_value=0):
    with pytest.raises(SystemExit) as exc:
      runpy.run_path(str(script_path), run_name="__main__")
    assert exc.value.code in (0, 1)
