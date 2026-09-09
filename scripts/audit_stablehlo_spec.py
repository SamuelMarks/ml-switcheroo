"""Audit the StableHLO specification to find missing operations.

This script downloads the StableHLO spec and compares it with
implemented operations to identify and output missing ones.
"""

import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import List, Optional, Set
import urllib.request
import yaml


def get_pip_cache_dir() -> Path:
  """Get a common cache directory location for pip across platforms.

  Returns:
      Path to pip cache directory or platform-appropriate fallback.
  """
  env_cache = os.environ.get("PIP_CACHE_DIR")
  if env_cache:
    return Path(env_cache)
  try:
    home = Path.home()
  except Exception:
    return Path(tempfile.gettempdir()) / "pip"

  if sys.platform == "darwin":
    return home / "Library" / "Caches" / "pip"
  if sys.platform == "win32":
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
      return Path(local_app_data) / "pip" / "cache"
    return home / "AppData" / "Local" / "pip" / "cache"
  xdg_cache = os.environ.get("XDG_CACHE_HOME")
  if xdg_cache:
    return Path(xdg_cache) / "pip"
  return home / ".cache" / "pip"


def get_fallback_raw_url(url: str) -> Optional[str]:
  """Construct a fallback CDN/raw HTTPS URL if GitHub raw is rate-limited.

  Args:
      url: The primary URL to convert.

  Returns:
      A fallback HTTPS raw URL if applicable, or None.
  """
  raw_github_prefix = "https://raw.githubusercontent.com/"
  if url.startswith(raw_github_prefix):
    parts = url[len(raw_github_prefix) :].split("/", 3)
    if len(parts) == 4:
      owner, repo, branch, path = parts
      return f"https://cdn.jsdelivr.net/gh/{owner}/{repo}@{branch}/{path}"
  return None


def fetch_raw_https(url: str) -> str:
  """Download specification text directly via HTTPS raw GET.

  Args:
      url: The URL to download from.

  Returns:
      Downloaded content as string.

  Raises:
      RuntimeError: If download fails.
  """
  req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
  try:
    with urllib.request.urlopen(req) as response:
      return str(response.read().decode("utf-8"))
  except Exception as e:
    raise RuntimeError(f"Failed to download spec: {e}") from e


def download_spec(
  url: str = "https://raw.githubusercontent.com/openxla/stablehlo/main/docs/spec.md",
  cache_path: Optional[Path] = None,
  force_download: bool = False,
) -> str:
  """Download or retrieve cached StableHLO specification markdown content.

  Checks pip common cache location first. If missing or invalid, falls back to
  HTTPS raw GET and stores the result in the cache location.

  Args:
      url: The URL to download the specification from.
      cache_path: Optional custom cache file path. Defaults to pip cache.
      force_download: If True, bypass cache and fetch directly via HTTPS.

  Returns:
      The specification markdown string.

  Raises:
      RuntimeError: If both cache retrieval and HTTPS downloads fail.
  """
  filename = "stablehlo_spec.md" if Path(url).name == "spec.md" else (Path(url).name or "spec.md")
  target_cache = cache_path or (get_pip_cache_dir() / "ml_switcheroo" / filename)

  if not force_download and target_cache.exists():
    try:
      cached_content = target_cache.read_text(encoding="utf-8")
      if cached_content.strip():
        return cached_content
    except Exception:
      pass

  last_error: Optional[Exception] = None
  downloaded_content: Optional[str] = None

  try:
    downloaded_content = fetch_raw_https(url)
  except Exception as primary_err:
    last_error = primary_err
    fallback_url = get_fallback_raw_url(url)
    if fallback_url:
      try:
        downloaded_content = fetch_raw_https(fallback_url)
      except Exception as fallback_err:
        last_error = fallback_err

  if downloaded_content is not None:
    try:
      target_cache.parent.mkdir(parents=True, exist_ok=True)
      target_cache.write_text(downloaded_content, encoding="utf-8")
    except Exception:
      pass
    return downloaded_content

  if target_cache.exists():
    try:
      fallback_content = target_cache.read_text(encoding="utf-8")
      if fallback_content.strip():
        return fallback_content
    except Exception:
      pass

  raise RuntimeError(f"Failed to download spec: {last_error}") from last_error


def parse_stablehlo_ops(content: str) -> List[str]:
  """Parse operation names from markdown headings (### op_name).

  Args:
      content: The StableHLO specification markdown text.

  Returns:
      List of parsed operation names.
  """
  ops: List[str] = []
  for line in content.splitlines():
    m = re.match(r"^###\s+([a-z0-9_]+)$", line)
    if m:
      ops.append(m.group(1))
  return ops


def load_implemented_stablehlo_ops(odl_dir: Path) -> Set[str]:
  """Load implemented StableHLO ops from ODL yaml files.

  Args:
      odl_dir: Path to directory containing ODL definitions.

  Returns:
      Set of operation names implemented for StableHLO.
  """
  implemented: Set[str] = set()
  if not odl_dir.exists():
    return implemented

  for p in odl_dir.glob("*.yaml"):
    try:
      with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        if isinstance(data, dict):
          variants = data.get("variants")
          if isinstance(variants, dict) and "stablehlo" in variants:
            shlo_var = variants["stablehlo"]
            if isinstance(shlo_var, dict):
              api = shlo_var.get("api", "")
              if api.startswith("stablehlo."):
                implemented.add(api[len("stablehlo.") :])
    except Exception:
      pass

  return implemented


def generate_reports(
  official_ops: List[str],
  implemented_ops: Set[str],
  missing_ops: List[str],
  json_output: Path,
  md_output: Path,
) -> None:
  """Write JSON and Markdown audit reports.

  Args:
      official_ops: All official operation names.
      implemented_ops: Set of implemented operation names.
      missing_ops: List of missing operation names.
      json_output: Output path for JSON report.
      md_output: Output path for Markdown report.
  """
  out = {
    "official_ops": sorted(official_ops),
    "implemented": sorted(list(implemented_ops)),
    "missing": missing_ops,
  }

  with open(json_output, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
    f.write(chr(10))

  with open(md_output, "w", encoding="utf-8") as f:
    f.write("# StableHLO Missing Operations" + chr(10))
    if missing_ops:
      f.write(chr(10))
    for op in missing_ops:
      f.write(f"- [ ] Implement `{op}`" + chr(10))


def main(
  spec_url: str = "https://raw.githubusercontent.com/openxla/stablehlo/main/docs/spec.md",
  odl_dir: Path = Path("src/ml_switcheroo/semantics/odl"),
  json_output: Path = Path("audit_stablehlo.json"),
  md_output: Path = Path("audit_stablehlo.md"),
  content: Optional[str] = None,
  cache_path: Optional[Path] = None,
  force_download: bool = False,
) -> int:
  """Main execution function for StableHLO specification audit.

  Args:
      spec_url: URL to StableHLO specification.
      odl_dir: Path to directory containing ODL definitions.
      json_output: Path for JSON output.
      md_output: Path for Markdown output.
      content: Optional pre-loaded spec content string.
      cache_path: Optional custom cache file path. Defaults to pip cache.
      force_download: If True, bypass cache and fetch directly via HTTPS.

  Returns:
      Exit code (0 if all implemented, 1 if missing ops).
  """
  if content is None:
    try:
      content = download_spec(spec_url, cache_path=cache_path, force_download=force_download)
    except RuntimeError as e:
      print(str(e))
      return 1

  ops = parse_stablehlo_ops(content)
  implemented = load_implemented_stablehlo_ops(odl_dir)
  missing = sorted(list(set(ops) - implemented))

  generate_reports(ops, implemented, missing, json_output, md_output)

  print(f"Found {len(ops)} total ops.")
  print(f"Implemented: {len(implemented)}")
  print(f"Missing: {len(missing)}")

  return 1 if missing else 0


if __name__ == "__main__":
  sys.exit(main())
