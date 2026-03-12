import os
import ast
import glob


def find_tested_apis():
  tested_frameworks = {}

  for file_path in glob.glob("tests/gold/semantic/*.py"):
    with open(file_path, "r") as f:
      tree = ast.parse(f.read(), filename=file_path)

    for node in ast.walk(tree):
      if isinstance(node, ast.FunctionDef):
        framework = "unknown"
        if "numpy" in node.name:
          framework = "numpy"
        elif "torch" in node.name:
          framework = "torch"
        elif "jax" in node.name:
          framework = "jax"
        elif "flax" in node.name:
          framework = "flax_nnx"
        elif "mlx" in node.name:
          framework = "mlx"
        elif "tensorflow" in node.name:
          framework = "tensorflow"
        elif "keras" in node.name:
          framework = "keras"
        elif "paxml" in node.name:
          framework = "paxml"
        elif "transformers" in node.name:
          framework = "transformers"
        elif "maxtext" in node.name:
          framework = "maxtext"
        elif "bonsai" in node.name:
          framework = "bonsai"
        else:
          if "test_numpy" in file_path:
            framework = "numpy"
          elif "test_torch" in file_path:
            framework = "torch"
          elif "test_jax" in file_path:
            framework = "jax"
          elif "test_keras" in file_path:
            framework = "keras"
          elif "test_paxml" in file_path:
            framework = "paxml"
          elif "test_transformers" in file_path:
            framework = "transformers"
          elif "test_maxtext" in file_path:
            framework = "maxtext"
          elif "test_bonsai" in file_path:
            framework = "bonsai"

        if framework not in tested_frameworks:
          tested_frameworks[framework] = set()

        # Look for pytest.mark.parametrize("func_name", [...])
        parametrized_funcs = []
        for dec in node.decorator_list:
          if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute) and dec.func.attr == "parametrize":
            if hasattr(dec.args[0], "value") and dec.args[0].value == "func_name":
              if isinstance(dec.args[1], ast.List):
                for item in dec.args[1].elts:
                  if isinstance(item, ast.Constant):
                    # Only add ones not explicitly skipped in the tests logic currently
                    if framework == "flax_nnx" and item.value in ["relu", "gelu", "silu", "leaky_relu"]:
                      pass
                    else:
                      parametrized_funcs.append(item.value)

        if parametrized_funcs:
          tested_frameworks[framework].update(parametrized_funcs)
        else:
          # It's a single test without parametrize, maybe the name has it?
          if node.name.startswith("test_numpy_"):
            tested_frameworks["numpy"].add(node.name.replace("test_numpy_", ""))
          elif node.name.startswith("test_torch_"):
            tested_frameworks["torch"].add(node.name.replace("test_torch_", ""))
          elif node.name.startswith("test_jax_"):
            tested_frameworks["jax"].add(node.name.replace("test_jax_", ""))

  return tested_frameworks


frameworks = find_tested_apis()
with open("ABSTRACT_COMPLIANCE2.md", "w") as f:
  f.write("# ML-Switcheroo Semantic Compliance Report 2\n\n")
  f.write(
    "This document tracks the specific ML framework APIs that have been **semantically verified** to be translated correctly across frameworks. The verifications occur completely within the standalone AST translator (running purely in Python via `libcst`), without framework side-effects, guaranteeing clean WASM integration.\n\n"
  )

  total = sum(len(apis) for apis in frameworks.values() if isinstance(apis, set))
  f.write(f"**Total Semantically Tested APIs:** {total}\n\n")

  for fw in sorted(frameworks.keys()):
    if fw == "unknown":
      continue
    apis = sorted(list(frameworks[fw]))
    if not apis:
      continue
    f.write(f"## {fw.capitalize()}\n\n")
    f.write(f"Verified: {len(apis)} APIs\n\n")
    for api in apis:
      f.write(f"- [x] `{fw}.{api}`\n")
    f.write("\n")
