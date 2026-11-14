"""Test suite for templates."""

from unittest.mock import MagicMock

from ml_switcheroo.generated_tests.templates import get_template
from ml_switcheroo.semantics.manager import SemanticsManager


def test_get_template_exception() -> None:
  """Verify `get_template` falls back cleanly when manager lookup raises."""
  manager = MagicMock(spec=SemanticsManager)
  manager.get_test_template.side_effect = RuntimeError("err")
  tmpl = get_template(manager, "torch")
  assert tmpl is not None
  assert tmpl["import"] == "import torch"


def test_is_static_arg() -> None:
  """Verify `is_static_arg` produces expected results across templates inputs."""
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
  """Verify `get_template` handles no manager as expected."""
  from ml_switcheroo.generated_tests.templates import get_template

  tmpl = get_template(None, "torch")
  assert tmpl is not None
  assert tmpl["import"] == "import torch"
