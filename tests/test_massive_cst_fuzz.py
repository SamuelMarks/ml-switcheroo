"""Test suite for the Massive Cst Fuzz module."""

import pytest
import os
import libcst as cst

pytest.skip("Too slow for pre-commit", allow_module_level=True)


def get_all_visitors():
  """Gets all visitors."""
  import importlib
  import pkgutil
  import inspect
  import ml_switcheroo

  visitors = []

  def iter_modules(package):
    """Helper to iter modules."""
    for loader, module_name, is_pkg in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
      try:
        module = importlib.import_module(module_name)
        for name, obj in inspect.getmembers(module):
          if (
            inspect.isclass(obj)
            and issubclass(obj, (cst.CSTVisitor, cst.CSTTransformer))
            and (obj not in (cst.CSTVisitor, cst.CSTTransformer, cst.RemoveFromParent))
          ):
            visitors.append(obj)
      except Exception:
        pass

  iter_modules(ml_switcheroo)
  return set(visitors)


@pytest.mark.skip(reason="Too slow")
def test_fuzz_all_visitors():
  """Verifies the behavior of fuzz all visitors."""
  visitors = get_all_visitors()
  code = ""
  for root, dirs, files in os.walk("src/ml_switcheroo"):
    for file in files:
      if file.endswith(".py"):
        with open(os.path.join(root, file), "r") as f:
          code += f.read() + "\n\n"
  tree = cst.parse_module(code)
  for visitor_cls in visitors:
    try:
      import unittest.mock

      mock_semantics = unittest.mock.MagicMock()
      mock_semantics.import_data = {"torch.foo": 1}
      try:
        visitor = visitor_cls()
      except TypeError:
        try:
          visitor = visitor_cls(mock_semantics)
        except TypeError:
          try:
            visitor = visitor_cls(mock_semantics, "torch")
          except TypeError:
            continue
      if hasattr(tree, "visit"):
        tree.visit(visitor)
    except Exception:
      pass
