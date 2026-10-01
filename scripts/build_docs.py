#!/usr/bin/env python3
"""Documentation Build Script for ml-switcheroo.

This script orchestrates the Sphinx documentation build process, including:
1.  Cleaning previous build artifacts.
2.  Importing root markdown files.
3.  Building a pure-Python Wheel for the WASM demo.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
DOCS_DIR = PROJECT_ROOT / "docs"
BUILD_DIR = DOCS_DIR / "_build"

# Files to copy from root to docs/ to be rendered
ROOT_FILES = (
  "ARCHITECTURE.md",
  "EXTENDING.md",
  "EXTENDING_WITH_DSL.md",
  "IDEAS.md",
  "INTERNALS.md",
  "LICENSE",
  "MAINTENANCE.md",
  "README.md",
)


def clean() -> None:
  """Clean the build directory and temporary artifacts."""
  if BUILD_DIR.exists():
    shutil.rmtree(BUILD_DIR, ignore_errors=True)

  for fname in ROOT_FILES:
    dest = DOCS_DIR / fname
    if dest.exists():
      dest.unlink()

  api_dir = DOCS_DIR / "api"
  if api_dir.exists():
    shutil.rmtree(api_dir, ignore_errors=True)

  # Clean generated Operations documentation to prevent stale files
  # triggering 'document isn't included in any toctree' warnings.
  ops_dir = DOCS_DIR / "ops"
  if ops_dir.exists():
    shutil.rmtree(ops_dir, ignore_errors=True)

  static_dir = DOCS_DIR / "_static"
  if static_dir.exists():
    for whl in static_dir.glob("*.whl"):
      whl.unlink()


def copy_root_files() -> None:
  """Copy essential Markdown files from the project root to the docs directory."""
  print("📋 Copying root Markdown files to docs/...")
  for fname in ROOT_FILES:
    src = PROJECT_ROOT / fname
    dest = DOCS_DIR / fname
    if src.exists():
      shutil.copy2(src, dest)
    else:
      print(f"⚠️  Warning: {fname} not found in root.")


def build_wheel() -> None:
  """Build the pure Python wheel for the WASM demo."""
  print("📦 Building Python Wheel for WASM...")
  dist_dir = PROJECT_ROOT / "dist"
  if dist_dir.exists():
    shutil.rmtree(dist_dir, ignore_errors=True)

  try:
    cmd = ["uv", "build", "--wheel"]
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True, capture_output=True)
    print("✅ Wheel built successfully in dist/")

    static_dir = DOCS_DIR / "_static"
    static_dir.mkdir(exist_ok=True)
    if dist_dir.exists():
      for whl in dist_dir.glob("*.whl"):
        shutil.copy2(whl, static_dir / whl.name)
        print(f"✅ Copied {whl.name} to {static_dir}/")
  except subprocess.CalledProcessError as e:
    print("❌ Failed to build wheel.")
    print("STDERR:", e.stderr.decode())
    sys.exit(1)


def copy_external_wheels() -> None:
  """Download or copy external GitHub wheels specified in requirements into docs/_static.

  Scans requirements.txt for any package pinned to an HTTPS GitHub wheel URL.
  If a matching local wheel exists in a sibling workspace, it is copied over.
  Otherwise, it is downloaded and placed into docs/_static so the WASM demo can
  install it locally without cross-origin or on-the-fly network issues.
  """
  static_dir = DOCS_DIR / "_static"
  static_dir.mkdir(exist_ok=True)

  reqs_file = PROJECT_ROOT / "requirements.txt"
  if not reqs_file.exists():
    return

  import urllib.request
  from ml_switcheroo.sphinx_ext.hooks import find_local_wheel

  with open(reqs_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

  for raw_line in lines:
    line = raw_line.strip()
    if not line or line.startswith("#"):
      continue
    if " @ " in line:
      pkg, url_or_spec = line.split(" @ ", 1)
      pkg = pkg.strip()
      url_or_spec = url_or_spec.strip()

      if (
        ("http://" in url_or_spec or "https://" in url_or_spec) and "github.com" in url_or_spec and ".whl" in url_or_spec
      ):
        whl_filename = url_or_spec.split("/")[-1].split("?")[0]
        dest_path = static_dir / whl_filename

        # First check if local wheel exists in sibling repo
        local_wheel = find_local_wheel(PROJECT_ROOT, pkg)
        if local_wheel is not None and local_wheel.exists():
          if dest_path.exists() and dest_path.stat().st_mtime >= local_wheel.stat().st_mtime:
            continue
          shutil.copy2(local_wheel, dest_path)
          print(f"✅ Copied local wheel {local_wheel.name} to {static_dir}/")
          continue

        # Otherwise download from GitHub release if not already present
        if not dest_path.exists():
          print(f"⬇️  Downloading {url_or_spec} to {dest_path}...")
          try:
            req = urllib.request.Request(url_or_spec, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as resp, open(dest_path, "wb") as out:
              shutil.copyfileobj(resp, out)
            print(f"✅ Downloaded {whl_filename} to {static_dir}/")
          except Exception as e:
            print(f"⚠️  Warning: Failed to download {url_or_spec}: {e}")


def calculate_unique_variants() -> None:
  """Calculate the unique cross-framework variants across all semantics files."""
  try:
    from ml_switcheroo.semantics.manager import SemanticsManager

    mgr = SemanticsManager()
    total_variants = len(mgr._reverse_index)

    print(f"📊 Calculated unique cross-framework variants: {total_variants}")
    if os.environ.get("CI") == "true" and total_variants < 1860:
      print(f"❌ CI Assertion Failed: Unique operator count ({total_variants}) is below the required 1860 limit.")
      sys.exit(1)
  except Exception as e:
    print(f"⚠️ Failed to calculate variants: {e}")


def build(build_all: bool = False) -> int:
  """Execute the Sphinx build process.

  Args:
      build_all: Whether to perform a full build.

  Returns:
      The return code from the sphinx-build command.
  """
  calculate_unique_variants()
  build_wheel()
  copy_external_wheels()

  print("🏗️  Building Sphinx documentation...")
  cmd = [
    sys.executable,
    "-m",
    "sphinx",
    "-j",
    "auto",
    "-b",
    "html",
    str(DOCS_DIR),
    str(BUILD_DIR / "html"),
  ]

  env = os.environ.copy()
  is_full_build = build_all and env.get("BUILD_ALL_DOCS") == "1"

  if not is_full_build:
    cmd.append(str(DOCS_DIR / "index.md"))
    # Setting an environment variable so Sphinx extensions know it's not a full build
    # We still use an env var to pass state to conf.py and sphinx_ext
    env["BUILD_ALL_DOCS"] = "0"

  result = subprocess.run(cmd, env=env)
  return result.returncode


def main() -> None:
  """Run the documentation build process."""
  import argparse

  parser = argparse.ArgumentParser(description="Build ml-switcheroo documentation.")
  parser.add_argument(
    "--build-all", action="store_true", help="Build all documentation. Requires BUILD_ALL_DOCS=1 env var."
  )
  args = parser.parse_args()

  try:
    clean()
    copy_root_files()
    ret = build(build_all=args.build_all)

    if ret == 0:
      index_path = BUILD_DIR / "html" / "index.html"
      print("\n✨ Documentation built successfully!")
      print(f"🌍 Open index at: {index_path.resolve()}")
      print("   (Serve with: python3 -m http.server --directory docs/_build/html)")
  finally:
    for fname in ROOT_FILES:
      dest = DOCS_DIR / fname
      if dest.exists():
        dest.unlink()

  sys.exit(ret)


if __name__ == "__main__":
  main()
