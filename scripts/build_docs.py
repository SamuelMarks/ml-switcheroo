#!/usr/bin/env python3
import os
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
  "IDEAS.md",
  "MAINTENANCE.md",
]


def clean():
  """Clean build directory and temp files."""
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


def copy_root_files():
  """Copies markdown files from project root to docs dir."""
  print("📋 Copying root Markdown files to docs/...")
  for fname in ROOT_FILES:
    src = PROJECT_ROOT / fname
    dest = DOCS_DIR / fname
    if src.exists():
      shutil.copy2(src, dest)
    else:
      print(f"⚠️  Warning: {fname} not found in root.")


def build():
  """Runs sphinx-build."""
  print("🏗️  Building Sphinx documentation...")
  cmd = [
    sys.executable, "-m", "sphinx",
    "-b", "html",
    str(DOCS_DIR),
    str(BUILD_DIR / "html")
  ]

  result = subprocess.run(cmd)
  return result.returncode


def main():
  try:
    clean()
    copy_root_files()
    ret = build()

    if ret == 0:
      index_path = BUILD_DIR / "html" / "index.html"
      print(f"\n✨ Documentation built successfully!")
      print(f"🌍 Open index at: {index_path.resolve()}")
  finally:
    # Optional: cleanup copied files after build
    # Comment this out if you want to inspect what was built
    for fname in ROOT_FILES:
      dest = DOCS_DIR / fname
      if dest.exists():
        dest.unlink()

  sys.exit(ret)


if __name__ == "__main__":
  main()
