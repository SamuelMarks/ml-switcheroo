"""Test suite for the Utils Coverage module."""

from pydantic import JsonValue


def test_code_extractor_error() -> None:
  """Verify code extractor error behavior."""
  import pytest

  from ml_switcheroo.utils.code_extractor import CodeExtractor

  ce = CodeExtractor()
  with __import__("unittest.mock").mock.patch("inspect.getsource", side_effect=OSError("fail")):
    with pytest.raises(OSError):
      ce.extract_class(CodeExtractor)


def test_code_extractor_normalize_harness_imports() -> None:
  """Verify code extractor normalize harness imports behavior."""
  from ml_switcheroo.utils.code_extractor import CodeExtractor

  ce = CodeExtractor()
  res = ce.normalize_harness_imports("pass", ["numpy", "torch.nn"])
  assert "import torch.nn" in res


def test_doc_context_branches() -> None:
  """Verifies the behavior of documentation context branches."""
  from ml_switcheroo.semantics.manager import SemanticsManager
  from ml_switcheroo.utils.doc_context import DocContextBuilder

  class DummySM(SemanticsManager):
    """Test helper class DummySM."""

    def get_all_operations(self) -> dict[str, dict[str, JsonValue]]:
      """Mock implementation of get all operations.

      Returns:
          Computed test result or fixture value.
      """
      return {}

  b = DocContextBuilder(DummySM())
  res = b.build(
    "foo", {"variants": {"jax": None, "torch": {"transformation_type": "inline_lambda"}, "mlx": {"something": "else"}}}
  )
  variants = res["variants"]
  assert isinstance(variants, list)
  assert len(variants) == 2
  for v in variants:
    assert isinstance(v, dict)
    if v["framework"] == "torch":
      assert v["type"] == "Inline Lambda"
    elif v["framework"] == "mlx":
      assert v["type"] == "Custom / Partial"


def test_code_extractor_error_more() -> None:
  """Verify code extractor error more behavior."""
  import pytest

  from ml_switcheroo.utils.code_extractor import CodeExtractor

  ce = CodeExtractor()
  with pytest.raises(TypeError):
    ce.extract_class(lambda: None)


def test_doc_context_more() -> None:
  """Verifies the behavior of documentation context more."""
  from ml_switcheroo.semantics.manager import SemanticsManager
  from ml_switcheroo.utils.doc_context import DocContextBuilder

  class DummySM(SemanticsManager):
    """Test helper class DummySM."""

    pass

  b = DocContextBuilder(DummySM())
  res = b.build(
    "foo",
    {
      "std_args": ["a", ["b", "int"], {"name": "c", "type": "float", "default": 1.0}],
      "variants": {
        "jax": {"requires_plugin": "foo", "api": "a"},
        "torch": {"type_map": "b", "api": "a"},
        "mlx": {"args": "c", "api": "a"},
      },
    },
  )
  args = res["args"]
  assert isinstance(args, list)
  assert isinstance(args[0], str) and "a" in args[0]
  assert isinstance(args[1], str) and "b: int" in args[1]
  assert isinstance(args[2], str) and "c: float = 1.0" in args[2]


def test_doc_context_more_variants() -> None:
  """Verifies the behavior of documentation context more variants."""
  from ml_switcheroo.semantics.manager import SemanticsManager
  from ml_switcheroo.utils.doc_context import DocContextBuilder

  class DummySM(SemanticsManager):
    """Test helper class DummySM."""

    pass

  b = DocContextBuilder(DummySM())
  res = b.build(
    "foo", {"variants": {"a": {"macro_template": "foo"}, "b": {"transformation_type": "infix", "operator": "+"}}}
  )
  variants = res["variants"]
  assert isinstance(variants, list)
  for v in variants:
    assert isinstance(v, dict)
    impl_type = v.get("implementation_type", "")
    assert isinstance(impl_type, str)
    if v["framework"] == "a":
      assert "Macro" in impl_type
    elif v["framework"] == "b":
      assert "Infix" in impl_type
