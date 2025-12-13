#!/usr/bin/env python3
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
  "LICENSE",  # Added to fix cross-reference warning
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


def build_wheel():
  """
  Builds the pure python wheel for the WASM demo.
  It delegates to 'python -m build' (generic frontend) or 'hatch build'.
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


def build():
  """Runs sphinx-build."""

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


def main():
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
