"""Doc."""

import pytest
import libcst as cst
import importlib
import pkgutil
import inspect
from unittest.mock import MagicMock
import ml_switcheroo

pytest.skip("Too slow for pre-commit", allow_module_level=True)


def get_all_classes_and_funcs():
  """Doc."""
  callables = []
  for loader, module_name, is_pkg in pkgutil.walk_packages(ml_switcheroo.__path__, ml_switcheroo.__name__ + "."):
    try:
      module = importlib.import_module(module_name)
      for name, obj in inspect.getmembers(module):
        if (
          (inspect.isfunction(obj) or inspect.isclass(obj))
          and obj.__module__
          and obj.__module__.startswith("ml_switcheroo")
        ):
          callables.append(obj)
    except Exception:
      pass
  return set(callables)


@pytest.mark.skip(reason="Too slow for pre-commit")
def test_improved_fuzz():
  """Doc."""
  with open("massive_code.py", "r") as f:
    code = f.read()

  # Also include all source code from the project to maximize CST types!
  import os

  for root, dirs, files in os.walk("src/ml_switcheroo"):
    for file in files:
      if file.endswith(".py"):
        with open(os.path.join(root, file), "r") as f:
          code += f.read() + "\n\n"

  tree = cst.parse_module(code)

  nodes = []

  class NodeCollector(cst.CSTVisitor):
    """Docstring."""

    def on_visit(self, node):
      """Docstring."""
      nodes.append(node)
      return True

  tree.visit(NodeCollector())

  # select a subset of nodes (e.g., max 100 of each type)
  from collections import defaultdict

  node_by_type = defaultdict(list)
  for n in nodes:
    if len(node_by_type[type(n)]) < 10:
      node_by_type[type(n)].append(n)

  reduced_nodes = []
  for type_nodes in node_by_type.values():
    reduced_nodes.extend(type_nodes)

  callables = get_all_classes_and_funcs()

  mock_ctx = MagicMock()
  mock_semantics = MagicMock()

  for obj in callables:
    # If it's a Visitor/Transformer
    if inspect.isclass(obj) and issubclass(obj, (cst.CSTVisitor, cst.CSTTransformer)):
      try:
        inst = obj()
      except Exception:
        try:
          inst = obj(mock_ctx)
        except Exception:
          try:
            inst = obj(mock_semantics)
          except Exception:
            try:
              inst = obj(mock_semantics, "torch")
            except Exception:
              continue

      # Visit the whole tree
      try:
        tree.visit(inst)
      except Exception:
        pass

    # If it's a normal function, try calling it with different node types
    elif inspect.isfunction(obj):
      for n in reduced_nodes:
        try:
          obj(n)
        except Exception:
          pass
        try:
          obj(n, mock_ctx)
        except Exception:
          pass
        try:
          obj(n, mock_semantics)
        except Exception:
          pass
        try:
          obj(n, ctx=mock_ctx)
        except Exception:
          pass

      # Try with Nones
      try:
        obj(None)
      except Exception:
        pass
      try:
        obj(None, mock_ctx)
      except Exception:
        pass
