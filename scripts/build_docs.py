#!/usr/bin/env python3
"""
Documentation Build Script for ml-switcheroo.

This script orchestrates the Sphinx documentation build process, including:
1.  Cleaning previous build artifacts.
2.  Importing root markdown files.
3.  Building a pure-Python Wheel for the WASM demo.
4.  Downloading static assets for TikZJax from a public CDN (jsDelivr) to avoid 403 errors.
"""

import gzip
import shutil
import subprocess
import sys
import urllib.request
import urllib.error
from pathlib import Path

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"
BUILD_DIR = DOCS_DIR / "_build"

# Files to copy from root to docs/ to be rendered
ROOT_FILES = [
  "README.md",
  "ARCHITECTURE.md",
  "EXTENDING.md",
  "EXTENDING_WITH_DSL.md",
  "IDEAS.md",
  "MAINTENANCE.md",
  "LICENSE",
]

# TikZJax Assets configuration
# Switch to jsDelivr CDN to avoid HTTP 403 Forbidden on the main site
TIKZ_BASE_URL = "https://cdn.jsdelivr.net/npm/tikzjax@1.0.3/dist"
TIKZ_ASSETS = [
  # Core Javascript
  {"name": "tikzjax.js", "remote": "tikzjax.js"},
  # WASM Runtime (Fetch raw, browser handles loading)
  {"name": "tex.wasm", "remote": "tex.wasm"},
  # TeX Memory Dump
  {"name": "core.dump", "remote": "core.dump"},
]


def clean() -> None:
  """
  Cleans the build directory and temporary artifacts.
  """
  if BUILD_DIR.exists():
    shutil.rmtree(BUILD_DIR)

  for fname in ROOT_FILES:
    dest = DOCS_DIR / fname
    if dest.exists():
      dest.unlink()

  api_dir = DOCS_DIR / "api"
  if api_dir.exists():
    shutil.rmtree(api_dir)


def copy_root_files() -> None:
  """
  Copies essential Markdown files from the project root to the docs directory.
  """
  print("📋 Copying root Markdown files to docs/...")
  for fname in ROOT_FILES:
    src = PROJECT_ROOT / fname
    dest = DOCS_DIR / fname
    if src.exists():
      shutil.copy2(src, dest)
    else:
      print(f"⚠️  Warning: {fname} not found in root.")


def download_vendor_assets() -> None:
  """
  Downloads TikZJax assets to docs/_static/tikzjax using a robust CDN.
  """
  target_dir = DOCS_DIR / "_static" / "tikzjax"
  target_dir.mkdir(parents=True, exist_ok=True)

  print(f"⬇️  Downloading TikZJax assets to {target_dir}...")

  for asset in TIKZ_ASSETS:
    local_name = asset["name"]
    remote_name = asset["remote"]

    url = f"{TIKZ_BASE_URL}/{remote_name}"
    dest = target_dir / local_name

    print(f"   Fetching {remote_name} -> {local_name}...")

    try:
      # Set a standard User-Agent to avoid generic blocks
      req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
      with urllib.request.urlopen(req) as response:
        with open(dest, "wb") as out_file:
          shutil.copyfileobj(response, out_file)

    except urllib.error.URLError as e:
      print(f"❌ Failed to download {remote_name}: {e}")
      print("   > The WASM demo might not render TikZ diagrams correctly locally.")
    except Exception as e:
      print(f"❌ Error processing {local_name}: {e}")


def build_wheel() -> None:
  """
  Builds the pure Python wheel for the WASM demo.
  """
  print("📦 Building Python Wheel for WASM...")
  dist_dir = PROJECT_ROOT / "dist"
  if dist_dir.exists():
    shutil.rmtree(dist_dir)

  try:
    cmd = [sys.executable, "-m", "build", ".", "--wheel"]
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True, capture_output=True)
    print("✅ Wheel built successfully in dist/")
  except subprocess.CalledProcessError as e:
    print("❌ Failed to build wheel.")
    print("STDERR:", e.stderr.decode())
    sys.exit(1)


def build() -> int:
  """
  Executes the Sphinx build process.
  """
  build_wheel()
  download_vendor_assets()

  print("🏗️  Building Sphinx documentation...")
  cmd = [
    sys.executable,
    "-m",
    "sphinx",
    "-b",
    "html",
    str(DOCS_DIR),
    str(BUILD_DIR / "html"),
  ]

  result = subprocess.run(cmd)
  return result.returncode


def main() -> None:
  try:
    clean()
    copy_root_files()
    ret = build()

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
