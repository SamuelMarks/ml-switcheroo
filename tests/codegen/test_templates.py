"""Docstring."""

import typing
from unittest.mock import MagicMock

from ml_switcheroo.generated_tests.templates import get_template


def test_get_template_exception() -> None:
  """Docstring."""
  manager: MagicMock = MagicMock()
  manager.get_test_template.side_effect = Exception("err")
  tmpl: typing.Optional[dict[str, str]] = get_template("torch", manager)
  assert tmpl is not None


def test_is_static_arg() -> None:
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


def test_get_template_no_manager() -> None:
  """Docstring."""
  from ml_switcheroo.generated_tests.templates import get_template

  tmpl: typing.Optional[dict[str, str]] = get_template(None, "torch")
  assert tmpl is not None
