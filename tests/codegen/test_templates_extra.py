"""Docstring."""

from ml_switcheroo.generated_tests.templates import get_template
from unittest.mock import MagicMock


def test_get_template_exception():
  """Docstring."""
  manager = MagicMock()
  manager.get_test_template.side_effect = Exception("err")
  tmpl = get_template("torch", manager)
  assert tmpl is not None


def test_is_static_arg():
  """Docstring."""
  from ml_switcheroo.generated_tests.templates import is_static_arg

  assert is_static_arg({"type": "int"}) is True
  assert is_static_arg({"type": "bool"}) is True
  assert is_static_arg({"type": "str"}) is True
  assert is_static_arg({"type": "list[int]"}) is True
  assert is_static_arg({"type": "tuple[int]"}) is True
  assert is_static_arg({"name": "axis"}) is True
  assert is_static_arg({"name": "dim"}) is True
  assert is_static_arg({"name": "keepdims"}) is True
  assert is_static_arg({"type": "Array"}) is False
  assert is_static_arg({"name": "x", "type": "Tensor"}) is False


def test_get_template_no_manager():
  """Docstring."""
  from ml_switcheroo.generated_tests.templates import get_template

  tmpl = get_template(None, "torch")
  assert tmpl is not None
