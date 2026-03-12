def test_code_extractor_error():
  from ml_switcheroo.utils.code_extractor import CodeExtractor
  import pytest
  import inspect

  ce = CodeExtractor()
  with __import__("unittest.mock").mock.patch("inspect.getsource", side_effect=OSError("fail")):
    with pytest.raises(OSError):
      ce.extract_class(CodeExtractor)


def test_code_extractor_normalize_harness_imports():
  from ml_switcheroo.utils.code_extractor import CodeExtractor

  ce = CodeExtractor()
  res = ce.normalize_harness_imports("pass", ["numpy", "torch.nn"])
  assert "import torch.nn" in res


def test_doc_context_branches():
  from ml_switcheroo.utils.doc_context import DocContextBuilder

  class DummySM:
    def get_all_operations(self):
      return {}

  b = DocContextBuilder(DummySM())

  res = b.build(
    "foo", {"variants": {"jax": None, "torch": {"transformation_type": "inline_lambda"}, "mlx": {"something": "else"}}}
  )

  assert len(res["variants"]) == 2
  for v in res["variants"]:
    if v["framework"] == "torch":
      assert v["type"] == "Inline Lambda"
    elif v["framework"] == "mlx":
      assert v["type"] == "Custom / Partial"


def test_code_extractor_error_more():
  from ml_switcheroo.utils.code_extractor import CodeExtractor
  import pytest

  ce = CodeExtractor()
  with pytest.raises(TypeError):
    ce.extract_class(lambda: None)


def test_doc_context_more():
  from ml_switcheroo.utils.doc_context import DocContextBuilder

  class DummySM:
    pass

  b = DocContextBuilder(DummySM())

  res = b.build(
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

  # Coverage for formatting args list
  assert "a" in res["args"][0]
  assert "b: int" in res["args"][1]
  assert "c: float = 1.0" in res["args"][2]


def test_doc_context_more_variants():
  from ml_switcheroo.utils.doc_context import DocContextBuilder

  class DummySM:
    pass

  b = DocContextBuilder(DummySM())

  res = b.build(
    "foo", {"variants": {"a": {"macro_template": "foo"}, "b": {"transformation_type": "infix", "operator": "+"}}}
  )

  for v in res["variants"]:
    if v["framework"] == "a":
      assert "Macro" in v.get("implementation_type", "")
    elif v["framework"] == "b":
      assert "Infix" in v.get("implementation_type", "")
