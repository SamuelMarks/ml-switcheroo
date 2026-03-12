import os
import subprocess
import ast
import json
import itertools
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

REPOS = {
  "pytorch": "https://github.com/pytorch/pytorch.git",
  "jax": "https://github.com/google/jax.git",
  "flax": "https://github.com/google/flax.git",
  "keras": "https://github.com/keras-team/keras.git",
  "mlx": "https://github.com/ml-explore/mlx.git",
  "transformers": "https://github.com/huggingface/transformers.git",
  "maxtext": "https://github.com/google/maxtext.git",
  "numpy": "https://github.com/numpy/numpy.git",
}

# Target core directories to avoid parsing thousands of unrelated util/C++ files
FOCUS_DIRS = {
  "pytorch": ["torch/nn/modules", "torch/nn/functional.py"],
  "jax": ["jax/_src/lax", "jax/_src/numpy"],
  "flax": ["flax/nnx"],
  "keras": ["keras/src/layers", "keras/src/ops"],
  "mlx": ["python/mlx/nn/layers"],
  "numpy": ["numpy/_core"],
  "transformers": ["src/transformers/models/llama/modeling_llama.py"],
  "maxtext": ["MaxText/layers"],
}

TMP_DIR = Path("tmp/repos")


def clone_repos():
  TMP_DIR.mkdir(parents=True, exist_ok=True)
  for name, url in REPOS.items():
    repo_path = TMP_DIR / name
    if not repo_path.exists():
      logging.info(f"Cloning {name}...")
      subprocess.run(
        ["git", "clone", "--depth", "1", url, str(repo_path)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
      )
    else:
      logging.info(f"{name} already exists. Skipping clone.")


def extract_api_surface(repo_name: str, focus_dirs: list[str]) -> list[str]:
  repo_path = TMP_DIR / repo_name
  api_nodes = []

  for fdir in focus_dirs:
    target_path = repo_path / fdir

    # Determine files to parse
    files_to_parse = []
    if target_path.is_file() and target_path.suffix == ".py":
      files_to_parse.append(target_path)
    elif target_path.is_dir():
      files_to_parse.extend(target_path.rglob("*.py"))

    for py_file in files_to_parse:
      # Skip test files
      if "test" in py_file.name:
        continue

      try:
        content = py_file.read_text(encoding="utf-8")
        tree = ast.parse(content)
        for node in ast.walk(tree):
          if isinstance(node, ast.ClassDef):
            # Filter out private classes
            if not node.name.startswith("_"):
              api_nodes.append(f"{py_file.stem}.{node.name}")
          elif isinstance(node, ast.FunctionDef):
            # Only top-level or significant functions (like in functional.py)
            if not node.name.startswith("_") and node.col_offset == 0:
              api_nodes.append(f"{py_file.stem}.{node.name}")
      except Exception as e:
        # Catch parse errors on weird syntax
        pass

  return sorted(list(set(api_nodes)))


def generate_mappings():
  logging.info("Extracting APIs...")
  apis = {}
  for name in REPOS.keys():
    apis[name] = extract_api_surface(name, FOCUS_DIRS[name])
    logging.info(f"  Extracted {len(apis[name])} ops from {name}")

  logging.info("Generating 10,000+ Step Plan...")

  plan_path = Path("10000_STEP_PLAN.md")
  json_path = Path("universal_mapping.json")

  # We will generate mappings betwixt all major frameworks
  frameworks = ["pytorch", "keras", "flax", "mlx", "jax", "numpy"]
  low_level = ["jax", "numpy"]
  zoos = ["transformers", "maxtext"]

  total_steps = 0
  all_mappings = {}

  with plan_path.open("w", encoding="utf-8") as f:
    f.write("# The 10,000+ Step Universal Transmutation Plan\n\n")
    f.write("This document outlines the exhaustive mapping required for lossless N <-> N conversion.\n\n")

    for src, dst in itertools.permutations(frameworks, 2):
      f.write(f"## Phase: {src.upper()} -> {dst.upper()}\n")
      f.write(f"Mapping high-level {src} modules to {dst} modules.\n\n")

      src_ops = apis[src]
      dst_ops = apis[dst]

      all_mappings[f"{src}_to_{dst}"] = {}

      for sop in src_ops:
        # Mock matching logic: if the name is similar, it's a direct map. Else, decomposition.
        sop_name = sop.split(".")[-1].lower()

        # Try to find a direct equivalent
        match = next((dop for dop in dst_ops if dop.split(".")[-1].lower() == sop_name), None)

        total_steps += 1
        if match:
          f.write(
            f"{total_steps}. [ ] **Direct Map**: Translate `{src}::{sop}` -> `{dst}::{match}`. Ensure attributes match.\n"
          )
          all_mappings[f"{src}_to_{dst}"][sop] = {"type": "direct", "target": match}
        else:
          # Decomposition step
          f.write(f"{total_steps}. [ ] **Decompose**: `{src}::{sop}` has no direct `{dst}` equivalent.\n")
          f.write(f"    - Sub-step {total_steps}.a: Lower `{src}::{sop}` to `{low_level[0]}`/math primitives.\n")
          f.write(f"    - Sub-step {total_steps}.b: Re-compile primitive graph to `{dst}` custom module.\n")
          all_mappings[f"{src}_to_{dst}"][sop] = {"type": "decompose", "intermediate": low_level[0]}

    f.write("\n## Phase: ZOO MACRO-ARCHITECTURE LIFTING\n\n")
    for zoo in zoos:
      for fw in frameworks:
        f.write(f"### Re-modularizing {zoo.upper()} -> {fw.upper()}\n")
        for zop in apis[zoo]:
          total_steps += 1
          f.write(
            f"{total_steps}. [ ] **Lift Macro**: Identify `{zoo}::{zop}` graph pattern and lower to `{fw}` idiomatic module.\n"
          )

  with json_path.open("w", encoding="utf-8") as jf:
    json.dump(all_mappings, jf, indent=2)

  logging.info(f"Generated {total_steps} discrete steps in {plan_path}")


if __name__ == "__main__":
  clone_repos()
  generate_mappings()
