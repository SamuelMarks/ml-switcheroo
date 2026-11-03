"""Test suite for the Dsl Imports module."""

from tests.conftest import _test_getattr

from ml_switcheroo.core.dsl import FrameworkVariant, ImportReq


def test_variant_imports_strings() -> None:
  """Verifies the behavior of variant imports strings."""
  v: FrameworkVariant = FrameworkVariant(api="foo", required_imports=["import os"])
  assert "import os" in _test_getattr(v, "required_imports")


def test_variant_imports_structured() -> None:
  """Verifies the behavior of variant imports structured."""
  req: ImportReq = ImportReq(module="numpy", alias="np")
  v: FrameworkVariant = FrameworkVariant(api="foo", required_imports=[req])
  assert len(_test_getattr(v, "required_imports")) == 1
  item = _test_getattr(v, "required_imports")[0]
  assert isinstance(item, ImportReq)
  assert _test_getattr(item, "module") == "numpy"
  assert _test_getattr(item, "alias") == "np"


def test_variant_imports_dict_coercion() -> None:
  """Verifies the behavior of variant imports dictionary coercion."""
  data = {"api": "foo", "required_imports": [{"module": "pandas", "alias": "pd"}]}
  v: FrameworkVariant = FrameworkVariant.model_validate(data)
  assert isinstance(_test_getattr(v, "required_imports")[0], ImportReq)
  assert _test_getattr(_test_getattr(v, "required_imports")[0], "alias") == "pd"


def test_variant_imports_mixed() -> None:
  """Verifies the behavior of variant imports mixed."""
  imports: list[str | ImportReq] = ["import cv2", ImportReq(module="PIL", alias="Image")]
  v: FrameworkVariant = FrameworkVariant(api="image_op", required_imports=imports)
  assert len(_test_getattr(v, "required_imports")) == 2
  assert "import cv2" in _test_getattr(v, "required_imports")
  assert isinstance(_test_getattr(v, "required_imports")[1], ImportReq)


def test_variant_imports_default_empty() -> None:
  """Verifies the behavior of variant imports default empty."""
  v: FrameworkVariant = FrameworkVariant(api="basic_op")
  assert _test_getattr(v, "required_imports") == []
