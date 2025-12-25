#!/usr/bin/env python3
"""
Documentation Build Script for ml-switcheroo.

This script orchestrates the Sphinx documentation build process, including:
1.  Cleaning previous build artifacts.
2.  Copying root-level Markdown files (README, ARCHITECTURE, etc.) into the docs directory.
3.  Building a pure-Python Wheel (.whl) of the package to support the interactive WASM demo.
4.  Invoking `sphinx-build` to generate the HTML site.
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
ROOT_FILES = [
  "README.md",
  "ARCHITECTURE.md",
  "EXTENDING.md",
  "EXTENDING_WITH_DSL.md",  # Added to fix cross-reference warning
  "IDEAS.md",
  "MAINTENANCE.md",
  "LICENSE",
]


def clean() -> None:
  """
  Cleans the build directory and temporary artifacts.

  Removes:
  - The `_build` directory.
  - Copied root Markdown files in `docs/`.
  - Auto-generated API documentation helpers.
  """
  if BUILD_DIR.exists():
    shutil.rmtree(BUILD_DIR)

  # Remove copied root files
  for fname in ROOT_FILES:
    dest = DOCS_DIR / fname
    if dest.exists():
      dest.unlink()

  # Remove autoapi helper dir
  api_dir = DOCS_DIR / "api"
  if api_dir.exists():
    shutil.rmtree(api_dir)


def copy_root_files() -> None:
  """
  Copies essential Markdown files from the project root to the docs directory.

  This allows files like `README.md` and `ARCHITECTURE.md` to be included
  in the Sphinx toctree without duplication.
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

  Delegates to the standard `python -m build` command. The resulting `.whl` file
  is placed in `dist/` and later picked up by the Sphinx extension to be embedded
  in the static site.

  Exits:
      sys.exit(1): If the build process fails.
  """
  print("📦 Building Python Wheel for WASM...")

  # Ensure 'dist' is clean to avoid grabbing old versions
  dist_dir = PROJECT_ROOT / "dist"
  if dist_dir.exists():
    shutil.rmtree(dist_dir)

  try:
    # Standard PEP 517 build
    cmd = [sys.executable, "-m", "build", ".", "--wheel"]
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True, capture_output=True)
    print("✅ Wheel built successfully in dist/")
  except subprocess.CalledProcessError as e:
    print("❌ Failed to build wheel.")
    print("STDERR:", e.stderr.decode())
    print("\n💡 Tip: Ensure 'build' is installed: pip install build")
    sys.exit(1)


def build() -> int:
  """
  Executes the Sphinx build process.

  1. Builds the package wheel.
  2. Runs `sphinx-build -b html`.

  Returns:
      int: The exit code from sphinx-build (0 for success).
  """
  # 1. Build the wheel first so the sphinx extension can find it
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
  """
  Main entry point. Orchestrates clean, copy, and build steps.
  """
  try:
    clean()
    copy_root_files()
    ret = build()

    if ret == 0:
      index_path = BUILD_DIR / "html" / "index.html"
      print("\n✨ Documentation built successfully!")
      print(f"🌍 Open index at: {index_path.resolve()}")
  finally:
    # Optional: cleanup copied files after build
    for fname in ROOT_FILES:
      dest = DOCS_DIR / fname
      if dest.exists():
        dest.unlink()

  sys.exit(ret)


if __name__ == "__main__":
  main()
