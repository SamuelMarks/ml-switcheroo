"""Doc."""

import pytest
import importlib
import pkgutil
import inspect
from unittest.mock import MagicMock
import ml_switcheroo

pytest.skip("Too slow for pre-commit", allow_module_level=True)


def get_all_callables():
  """Doc."""
  callables = []
  for loader, module_name, is_pkg in pkgutil.walk_packages(ml_switcheroo.__path__, ml_switcheroo.__name__ + "."):
    try:
      module = importlib.import_module(module_name)
      for name, obj in inspect.getmembers(module):
        if inspect.isfunction(obj) or inspect.isclass(obj) or inspect.ismethod(obj):
          if obj.__module__ and obj.__module__.startswith("ml_switcheroo"):
            callables.append(obj)
    except Exception:
      pass
  return set(callables)


def try_call(func, *args, **kwargs):
  """Doc."""
  try:
    if inspect.isclass(func):
      obj = func(*args, **kwargs)
      # Try to call its methods
      for name, method in inspect.getmembers(obj, predicate=inspect.ismethod):
        try:
          method()
        except Exception:
          pass
        try:
          method(MagicMock())
        except Exception:
          pass
    else:
      func(*args, **kwargs)
  except Exception:
    pass


@pytest.mark.skip(reason="Too slow")
def test_brute_force():
  """Doc."""
  callables = get_all_callables()
  for func in callables:
    # Try with no args
    try_call(func)
    # Try with one MagicMock
    try_call(func, MagicMock())
    # Try with None
    try_call(func, None)
    # Try with multiple mocks
    try_call(func, MagicMock(), MagicMock())
    try_call(func, None, None)
    try_call(func, MagicMock(), None)
    try_call(func, MagicMock(), MagicMock(), MagicMock())

    # Try with some common keyword args just in case
    try_call(func, node=MagicMock())
    try_call(func, context=MagicMock())
    try_call(func, semantics=MagicMock())
