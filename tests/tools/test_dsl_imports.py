"""Test suite for the Dsl Imports module."""

from ml_switcheroo.core.dsl import FrameworkVariant, ImportReq


def test_variant_imports_strings() -> None:
  """Verify `FrameworkVariant` emits `import os` for variant imports strings."""
  v: FrameworkVariant = FrameworkVariant(api="foo", required_imports=["import os"])
  assert "import os" in v.required_imports


def test_variant_imports_structured() -> None:
  """Verify `ImportReq` returns `1` for variant imports structured."""
  req: ImportReq = ImportReq(module="numpy", alias="np")
  v: FrameworkVariant = FrameworkVariant(api="foo", required_imports=[req])
  assert len(v.required_imports) == 1
  item = v.required_imports[0]
  assert isinstance(item, ImportReq)
  assert item.module == "numpy"
  assert item.alias == "np"


def test_variant_imports_dict_coercion() -> None:
  """Verify `FrameworkVariant.model_validate` handles variant imports dict coercion as expected."""
  data = {"api": "foo", "required_imports": [{"module": "pandas", "alias": "pd"}]}
  v: FrameworkVariant = FrameworkVariant.model_validate(data)
  assert isinstance(v.required_imports[0], ImportReq)
  assert v.required_imports[0].alias == "pd"


def test_variant_imports_mixed() -> None:
  """Verify `FrameworkVariant` returns `2` for variant imports mixed."""
  imports: list[str | ImportReq] = ["import cv2", ImportReq(module="PIL", alias="Image")]
  v: FrameworkVariant = FrameworkVariant(api="image_op", required_imports=imports)
  assert len(v.required_imports) == 2
  assert "import cv2" in v.required_imports
  assert isinstance(v.required_imports[1], ImportReq)


def test_variant_imports_default_empty() -> None:
  """Verify `FrameworkVariant` handles variant imports default empty as expected."""
  v: FrameworkVariant = FrameworkVariant(api="basic_op")
  assert v.required_imports == []
