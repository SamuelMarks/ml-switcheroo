"""Test suite for the Dsl Imports module."""

from ml_switcheroo.core.dsl import FrameworkVariant, ImportReq
from typing import Dict, Any


def test_variant_imports_strings() -> None:
  """Verifies the behavior of variant imports strings."""
  v: FrameworkVariant = FrameworkVariant(api="foo", required_imports=["import os"])
  assert "import os" in getattr(v, "required_imports")


def test_variant_imports_structured() -> None:
  """Verifies the behavior of variant imports structured."""
  req: ImportReq = ImportReq(module="numpy", alias="np")
  v: FrameworkVariant = FrameworkVariant(api="foo", required_imports=[req])
  assert len(getattr(v, "required_imports")) == 1
  item: Any = getattr(v, "required_imports")[0]
  assert isinstance(item, ImportReq)
  assert getattr(item, "module") == "numpy"
  assert getattr(item, "alias") == "np"


def test_variant_imports_dict_coercion() -> None:
  """Verifies the behavior of variant imports dictionary coercion."""
  data: Dict[str, Any] = {"api": "foo", "required_imports": [{"module": "pandas", "alias": "pd"}]}
  v: FrameworkVariant = FrameworkVariant.model_validate(data)
  assert isinstance(getattr(v, "required_imports")[0], ImportReq)
  assert getattr(getattr(v, "required_imports")[0], "alias") == "pd"


def test_variant_imports_mixed() -> None:
  """Verifies the behavior of variant imports mixed."""
  imports: list[Any] = ["import cv2", ImportReq(module="PIL", alias="Image")]
  v: FrameworkVariant = FrameworkVariant(api="image_op", required_imports=imports)
  assert len(getattr(v, "required_imports")) == 2
  assert "import cv2" in getattr(v, "required_imports")
  assert isinstance(getattr(v, "required_imports")[1], ImportReq)


def test_variant_imports_default_empty() -> None:
  """Verifies the behavior of variant imports default empty."""
  v: FrameworkVariant = FrameworkVariant(api="basic_op")
  assert getattr(v, "required_imports") == []
