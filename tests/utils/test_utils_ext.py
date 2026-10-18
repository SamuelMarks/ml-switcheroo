"""Test suite for the Utils Coverage module."""

import typing


def test_code_extractor_error() -> None:
  """Verify code extractor error behavior."""
  import pytest

  from ml_switcheroo.utils.code_extractor import CodeExtractor

  ce: typing.Any = CodeExtractor()
  with __import__("unittest.mock").mock.patch("inspect.getsource", side_effect=OSError("fail")):
    with pytest.raises(OSError):
      ce.extract_class(CodeExtractor)


def test_code_extractor_normalize_harness_imports() -> None:
  """Verify code extractor normalize harness imports behavior."""
  from ml_switcheroo.utils.code_extractor import CodeExtractor

  ce: typing.Any = CodeExtractor()
  res: typing.Any = ce.normalize_harness_imports("pass", ["numpy", "torch.nn"])
  assert "import torch.nn" in res


def test_doc_context_branches() -> None:
  """Verifies the behavior of documentation context branches."""
  from ml_switcheroo.utils.doc_context import DocContextBuilder

  class DummySM:
    """Test helper class DummySM."""

    def get_all_operations(self) -> typing.Any:
      """Mock implementation of get all operations.

      Returns:
          Computed test result or fixture value.
      """
      return {}

  b: typing.Any = DocContextBuilder(DummySM())
  res: typing.Any = b.build(
    "foo", {"variants": {"jax": None, "torch": {"transformation_type": "inline_lambda"}, "mlx": {"something": "else"}}}
  )
  assert len(res["variants"]) == 2
  for v in res["variants"]:
    if v["framework"] == "torch":
      assert v["type"] == "Inline Lambda"
    elif v["framework"] == "mlx":
      assert v["type"] == "Custom / Partial"


def test_code_extractor_error_more() -> None:
  """Verify code extractor error more behavior."""
  import pytest

  from ml_switcheroo.utils.code_extractor import CodeExtractor

  ce: typing.Any = CodeExtractor()
  with pytest.raises(TypeError):
    ce.extract_class(lambda: None)


def test_doc_context_more() -> None:
  """Verifies the behavior of documentation context more."""
  from ml_switcheroo.utils.doc_context import DocContextBuilder

  class DummySM:
    """Test helper class DummySM."""

    pass

  b: typing.Any = DocContextBuilder(DummySM())
  res: typing.Any = b.build(
    "foo",
    {
      "std_args": ["a", ("b", "int"), {"name": "c", "type": "float", "default": 1.0}],
      "variants": {
        "jax": {"requires_plugin": "foo", "api": "a"},
        "torch": {"type_map": "b", "api": "a"},
        "mlx": {"args": "c", "api": "a"},
      },
    },
  )
  assert "a" in res["args"][0]
  assert "b: int" in res["args"][1]
  assert "c: float = 1.0" in res["args"][2]


def test_doc_context_more_variants() -> None:
  """Verifies the behavior of documentation context more variants."""
  from ml_switcheroo.utils.doc_context import DocContextBuilder

  class DummySM:
    """Test helper class DummySM."""

    pass

  b: typing.Any = DocContextBuilder(DummySM())
  res: typing.Any = b.build(
    "foo", {"variants": {"a": {"macro_template": "foo"}, "b": {"transformation_type": "infix", "operator": "+"}}}
  )
  for v in res["variants"]:
    if v["framework"] == "a":
      assert "Macro" in v.get("implementation_type", "")
    elif v["framework"] == "b":
      assert "Infix" in v.get("implementation_type", "")
