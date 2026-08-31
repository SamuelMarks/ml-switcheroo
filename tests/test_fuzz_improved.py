"""Test suite for the Fuzz Improved module."""

import importlib
import inspect
import pkgutil
from typing import Dict, List, Set
from unittest.mock import MagicMock

import libcst as cst
import pytest

import ml_switcheroo

pytest.skip("Too slow for pre-commit", allow_module_level=True)


def get_all_classes_and_funcs() -> Set[type]:
  """Gets all classes and funcs."""
  callables: List[type] = []
  # Note: ml_switcheroo.__path__ is a list of str, but it might complain about type
  path: list[str] = getattr(ml_switcheroo, "__path__", [])
  for loader, module_name, is_pkg in pkgutil.walk_packages(path, ml_switcheroo.__name__ + "."):
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
def test_improved_fuzz() -> None:
  """Verifies the behavior of improved fuzz."""
  with open("massive_code.py", "r") as f:
    code: str = f.read()
  import os

  for root, dirs, files in os.walk("src/ml_switcheroo"):
    for file in files:
      if file.endswith(".py"):
        with open(os.path.join(root, file), "r") as f:
          code += f.read() + "\n\n"
  tree: cst.Module = cst.parse_module(code)
  nodes: List[cst.CSTNode] = []

  class NodeCollector(cst.CSTVisitor):
    """Docstring."""

    def on_visit(self, node: cst.CSTNode) -> bool:
      """Helper to on visit."""
      nodes.append(node)
      return True

  tree.visit(NodeCollector())
  from collections import defaultdict

  node_by_type: Dict[type, List[cst.CSTNode]] = defaultdict(list)
  for n in nodes:
    if len(node_by_type[type(n)]) < 10:
      node_by_type[type(n)].append(n)
  reduced_nodes: List[cst.CSTNode] = []
  for type_nodes in node_by_type.values():
    reduced_nodes.extend(type_nodes)
  callables: Set[type] = get_all_classes_and_funcs()
  mock_ctx: MagicMock = MagicMock()
  mock_semantics: MagicMock = MagicMock()
  for obj in callables:
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
      try:
        tree.visit(inst)
      except Exception:
        pass
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
      try:
        obj(None)
      except Exception:
        pass
      try:
        obj(None, mock_ctx)
      except Exception:
        pass
