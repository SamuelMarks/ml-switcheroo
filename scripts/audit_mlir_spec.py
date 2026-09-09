"""Script to audit MLIR spec against implemented Lark grammar rules."""

import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Dict, List, Optional, Set, Tuple
import urllib.request

MANUAL_MAPPING: Dict[str, str] = {
  "string-literal": "string",
  "integer-literal": "number",
  "float-literal": "number",
  "bare-id": "identifier",
  "value-id": "val_id",
  "caret-id": "block_label",
  "symbol-ref-id": "sym_id",
  "type": "type",
  "region": "regions",
  "attribute-value": "attribute",
}

IGNORED_LHS_TERMS: Set[str] = {
  "alternation",
  "sequence",
  "repetition0",
  "repetition1",
  "optionality",
  "grouping",
  "literal",
  "example",
}


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
  url: str = "https://raw.githubusercontent.com/llvm/llvm-project/main/mlir/docs/LangRef.md",
  cache_path: Optional[Path] = None,
  force_download: bool = False,
) -> str:
  """Download or retrieve cached MLIR specification markdown content.

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
  target_cache = cache_path or (get_pip_cache_dir() / "ml_switcheroo" / (Path(url).name or "LangRef.md"))

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


def parse_grammar_rules(content: str) -> Dict[str, str]:
  """Parse BNF-style grammar rules from specification markdown blocks.

  Args:
      content: Markdown content of the specification.

  Returns:
      Dictionary mapping grammar rule names to their syntax specifications.
  """
  grammar_rules: Dict[str, str] = {}
  in_block = False
  block_lines: List[str] = []

  for line in content.splitlines():
    if line.startswith("```"):
      if in_block:
        block_content = chr(10).join(block_lines)
        if "::=" in block_content:
          for rule_line in block_lines:
            stripped = rule_line.strip()
            if "::=" in stripped and not stripped.startswith("//"):
              parts = stripped.split("::=", 1)
              lhs = parts[0].strip()
              rhs = re.sub(r"//.*$", "", parts[1]).strip()
              if lhs and rhs and lhs not in IGNORED_LHS_TERMS:
                grammar_rules[lhs] = rhs
        in_block = False
        block_lines = []
      else:
        in_block = True
        block_lines = []
    elif in_block:
      block_lines.append(line)

  return grammar_rules


def extract_implemented_lark_rules(lark_path: Path) -> Set[str]:
  """Extract implemented rule identifiers from a Lark grammar file.

  Args:
      lark_path: Path to the grammar.lark file.

  Returns:
      Set of lowercased and normalized rule names.
  """
  implemented_rules: Set[str] = set()
  if not lark_path.exists():
    return implemented_rules

  with open(lark_path, "r", encoding="utf-8") as f:
    lark_content = f.read()

  for line in lark_content.splitlines():
    m = re.match(r"^([a-zA-Z_0-9]+)\s*:", line)
    if m:
      name = m.group(1).lower()
      implemented_rules.add(name)
      implemented_rules.add(name.replace("_", "-"))

  for mlir_term, lark_term in MANUAL_MAPPING.items():
    if lark_term in implemented_rules:
      implemented_rules.add(mlir_term)
      implemented_rules.add(mlir_term.replace("-", "_"))

  return implemented_rules


def audit_mlir(
  grammar_rules: Dict[str, str],
  implemented_rules: Set[str],
) -> Tuple[List[str], List[str]]:
  """Audit grammar rules against implemented Lark rules.

  Args:
      grammar_rules: Dictionary of rules parsed from spec.
      implemented_rules: Set of rules implemented in Lark grammar.

  Returns:
      Tuple of (implemented_matches, missing_rules).
  """
  missing_rules: List[str] = []
  implemented_matches: List[str] = []

  for rule in grammar_rules.keys():
    rule_lower = rule.lower()
    normalized = rule_lower.replace("-", "_")
    if rule_lower in implemented_rules or normalized in implemented_rules:
      implemented_matches.append(rule)
    else:
      missing_rules.append(rule)

  return implemented_matches, missing_rules


def generate_reports(
  grammar_rules: Dict[str, str],
  missing_rules: List[str],
  json_output: Path,
  md_output: Path,
) -> None:
  """Write JSON and Markdown audit reports.

  Args:
      grammar_rules: All parsed grammar rules.
      missing_rules: List of missing rule names.
      json_output: Output path for JSON report.
      md_output: Output path for Markdown report.
  """
  with open(json_output, "w", encoding="utf-8") as f:
    json.dump(grammar_rules, f, indent=2)
    f.write(chr(10))

  with open(md_output, "w", encoding="utf-8") as f:
    f.write("# MLIR LangRef Missing Grammar Rules" + chr(10))
    if missing_rules:
      f.write(chr(10))
    for rule in sorted(missing_rules):
      f.write(f"- [ ] Implement `{rule}` (Spec: `{grammar_rules[rule]}`)" + chr(10))


def main(
  spec_url: str = "https://raw.githubusercontent.com/llvm/llvm-project/main/mlir/docs/LangRef.md",
  lark_path: Path = Path("src/ml_switcheroo/core/mlir/grammar.lark"),
  json_output: Path = Path("audit_mlir.json"),
  md_output: Path = Path("audit_mlir.md"),
  content: Optional[str] = None,
  cache_path: Optional[Path] = None,
  force_download: bool = False,
) -> int:
  """Main execution function for MLIR specification audit.

  Args:
      spec_url: URL to MLIR LangRef spec.
      lark_path: Path to Lark grammar file.
      json_output: Path for JSON output.
      md_output: Path for Markdown output.
      content: Optional pre-loaded spec content string.
      cache_path: Optional custom cache file path. Defaults to pip cache.
      force_download: If True, bypass cache and fetch directly via HTTPS.

  Returns:
      Exit code (0 if all implemented, 1 if missing rules).
  """
  if content is None:
    try:
      content = download_spec(spec_url, cache_path=cache_path, force_download=force_download)
    except RuntimeError as e:
      print(str(e))
      return 1

  grammar_rules = parse_grammar_rules(content)
  implemented_rules = extract_implemented_lark_rules(lark_path)
  implemented_matches, missing_rules = audit_mlir(grammar_rules, implemented_rules)

  generate_reports(grammar_rules, missing_rules, json_output, md_output)

  print(f"Found {len(grammar_rules)} total MLIR grammar rules.")
  print(f"Implemented (fuzzy match): {len(implemented_matches)}")
  print(f"Missing: {len(missing_rules)}")

  return 1 if missing_rules else 0


if __name__ == "__main__":
  sys.exit(main())
