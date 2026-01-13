#!/usr/bin/env python3
"""
Documentation Build Script for ml-switcheroo.

This script orchestrates the Sphinx documentation build process, including:
1.  Cleaning previous build artifacts.
2.  Importing root markdown files.
3.  Building a pure-Python Wheel for the WASM demo.
"""

import shutil
import subprocess
import sys
from pathlib import Path

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent
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

  # Clean generated Operations documentation to prevent stale files
  # triggering 'document isn't included in any toctree' warnings.
  ops_dir = DOCS_DIR / "ops"
  if ops_dir.exists():
    shutil.rmtree(ops_dir)


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
