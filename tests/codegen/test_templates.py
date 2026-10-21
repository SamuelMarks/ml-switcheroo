"""Test suite for templates."""

from unittest.mock import MagicMock

from ml_switcheroo.generated_tests.templates import get_template


def test_get_template_exception() -> None:
  """Verify get template exception behavior."""
  manager = MagicMock()
  manager.get_test_template.side_effect = Exception("err")
  tmpl = getattr(
    {
      "v": getattr({"v": get_template}, "__getitem__")("v")(
        getattr({"v": "torch"}, "__getitem__")("v"), getattr({"v": manager}, "__getitem__")("v")
      )
    },
    "__getitem__",
  )("v")
  assert tmpl is not None


def test_is_static_arg() -> None:
  """Verify is static arg behavior."""
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
  """Verify get template no manager behavior."""
  from ml_switcheroo.generated_tests.templates import get_template

  tmpl = get_template(None, "torch")
  assert tmpl is not None
