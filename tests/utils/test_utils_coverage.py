"""Test suite for the Utils Coverage module."""

from typing import Any, Dict, List


def test_code_extractor_error() -> None:
  """Docstring."""
  import pytest

  from ml_switcheroo.utils.code_extractor import CodeExtractor

  ce: CodeExtractor = CodeExtractor()
  with __import__("unittest.mock").mock.patch("inspect.getsource", side_effect=OSError("fail")):
    with pytest.raises(OSError):
      ce.extract_class(CodeExtractor)


def test_code_extractor_normalize_harness_imports() -> None:
  """Docstring."""
  from ml_switcheroo.utils.code_extractor import CodeExtractor

  ce: CodeExtractor = CodeExtractor()
  res: List[str] = ce.normalize_harness_imports("pass", ["numpy", "torch.nn"])
  assert "import torch.nn" in res


def test_doc_context_branches() -> None:
  """Verifies the behavior of documentation context branches."""
  from ml_switcheroo.utils.doc_context import DocContextBuilder

  class DummySM:
    """Docstring."""

    def get_all_operations(self) -> Dict[str, Any]:
      """Mock implementation of get all operations."""
      return {}

  b: DocContextBuilder = DocContextBuilder(DummySM())
  res: Dict[str, Any] = b.build(
    "foo", {"variants": {"jax": None, "torch": {"transformation_type": "inline_lambda"}, "mlx": {"something": "else"}}}
  )
  assert len(res["variants"]) == 2
  for v in res["variants"]:
    if v["framework"] == "torch":
      assert v["type"] == "Inline Lambda"
    elif v["framework"] == "mlx":
      assert v["type"] == "Custom / Partial"


def test_code_extractor_error_more() -> None:
  """Docstring."""
  import pytest

  from ml_switcheroo.utils.code_extractor import CodeExtractor

  ce: CodeExtractor = CodeExtractor()
  with pytest.raises(TypeError):
    ce.extract_class(lambda: None)


def test_doc_context_more() -> None:
  """Verifies the behavior of documentation context more."""
  from ml_switcheroo.utils.doc_context import DocContextBuilder

  class DummySM:
    """Docstring."""

    pass

  b: DocContextBuilder = DocContextBuilder(DummySM())
  res: Dict[str, Any] = b.build(
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
    """Docstring."""

    pass

  b: DocContextBuilder = DocContextBuilder(DummySM())
  res: Dict[str, Any] = b.build(
    "foo", {"variants": {"a": {"macro_template": "foo"}, "b": {"transformation_type": "infix", "operator": "+"}}}
  )
  for v in res["variants"]:
    if v["framework"] == "a":
      assert "Macro" in v.get("implementation_type", "")
    elif v["framework"] == "b":
      assert "Infix" in v.get("implementation_type", "")
