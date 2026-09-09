"""Parse tablegen definitions."""

import json
import os
import re
from typing import Dict, List, Set

DIR_TO_DIALECT: Dict[str, str] = {
  "arith": "arith",
  "builtin": "builtin",
  "func": "func",
  "gpu": "gpu",
  "linalg": "linalg",
  "math": "math",
  "memref": "memref",
  "scf": "scf",
  "shape": "shape",
  "tensor": "tensor",
  "vector": "vector",
  "controlflow": "cf",
  "spirv": "spirv",
  "llvmir": "llvm",
  "nvvm": "nvvm",
  "rocdl": "rocdl",
  "openmp": "omp",
  "openacc": "acc",
  "affine": "affine",
  "amdgpu": "amdgpu",
  "async": "async",
  "complex": "complex",
  "emitc": "emitc",
  "index": "index",
  "pdl": "pdl",
  "quant": "quant",
  "sparse_tensor": "sparse_tensor",
  "tosa": "tosa",
  "transform": "transform",
}

RE1 = re.compile(r'def\s+[A-Za-z0-9_]+Op\s*:\s*[A-Za-z0-9_]+<\s*"([a-zA-Z0-9_.-]+)"')
RE2 = re.compile(r'def\s+[A-Za-z0-9_]+Op\s*:\s*Op<\s*[A-Za-z0-9_]+,\s*"([a-zA-Z0-9_.-]+)"')


def parse_td_files(root_dir: str) -> Dict[str, List[str]]:
  """Parse TableGen definitions from directory.

  Args:
      root_dir: Directory containing .td TableGen files.

  Returns:
      Dictionary mapping dialect names to sorted lists of operation names.
  """
  ops_by_dialect: Dict[str, Set[str]] = {}

  for root, _, files in os.walk(root_dir):
    for f in files:
      if f.endswith(".td"):
        path = os.path.join(root, f)

        if "/Dialect/" in path:
          dir_name = path.split("/Dialect/")[1].split("/")[0].lower()
          dialect = DIR_TO_DIALECT.get(dir_name, dir_name) if dir_name else "unknown"
        elif "/IR/" in path:
          dialect = "builtin"
        else:
          dialect = "unknown"

        with open(path, "r", encoding="utf-8") as file:
          content = file.read()

        ops_found: Set[str] = set()
        for m in RE1.finditer(content):
          ops_found.add(m.group(1))
        for m in RE2.finditer(content):
          ops_found.add(m.group(1))

        for op in ops_found:
          ops_by_dialect.setdefault(dialect, set()).add(op)

  final_ops: Dict[str, List[str]] = {d: sorted(list(ops)) for d, ops in ops_by_dialect.items()}

  return final_ops


def main(
  llvm_include_dir: str = "/tmp/llvm-project/mlir/include/mlir",
  output_path: str = "mlir_official_ops.json",
) -> int:
  """Main execution function for parsing TableGen files.

  Args:
      llvm_include_dir: Path to MLIR include directory.
      output_path: Path to output JSON file.

  Returns:
      Exit code (0 for success).
  """
  final_ops = parse_td_files(llvm_include_dir)
  with open(output_path, "w", encoding="utf-8") as file:
    json.dump(final_ops, file, indent=2)

  total_ops = sum(len(v) for v in final_ops.values())
  print(f"Found {total_ops} operations across {len(final_ops)} dialects.")
  return 0


if __name__ == "__main__":
  import sys

  sys.exit(main())
