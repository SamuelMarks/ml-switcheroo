"""Parse tablegen definitions."""

import os
import re
import json

ops_by_dialect: dict[str, set[str]] = {}

re1 = re.compile(r'def\s+[A-Za-z0-9_]+Op\s*:\s*[A-Za-z0-9_]+<\s*"([a-zA-Z0-9_.-]+)"')
re2 = re.compile(r'def\s+[A-Za-z0-9_]+Op\s*:\s*Op<\s*[A-Za-z0-9_]+,\s*"([a-zA-Z0-9_.-]+)"')

# Some known directories to real dialect names mapping
dir_to_dialect = {
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

for root, _, files in os.walk("/tmp/llvm-project/mlir/include/mlir"):
  for f in files:
    if f.endswith(".td"):
      path = os.path.join(root, f)

      # Identify dialect from path
      dialect = "unknown"
      if "/Dialect/" in path:
        parts = path.split("/Dialect/")
        if len(parts) > 1:
          dir_name = parts[1].split("/")[0].lower()
          dialect = dir_to_dialect.get(dir_name, dir_name)
      elif "/IR/" in path:
        dialect = "builtin"

      with open(path, "r", encoding="utf-8") as file:
        content = file.read()

        ops_found = set()
        for m in re1.finditer(content):
          ops_found.add(m.group(1))
        for m in re2.finditer(content):
          ops_found.add(m.group(1))

        for op in ops_found:
          ops_by_dialect.setdefault(dialect, set()).add(op)

final_ops = {}
for d, ops in ops_by_dialect.items():
  if len(ops) > 0:
    final_ops[d] = sorted(list(ops))

with open("mlir_official_ops.json", "w") as file:
  json.dump(final_ops, file, indent=2)

print(f"Found {sum(len(v) for v in final_ops.values())} operations across {len(final_ops)} dialects.")
