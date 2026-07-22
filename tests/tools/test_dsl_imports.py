"""Test suite for the Dsl Imports module."""

from ml_switcheroo.core.dsl import FrameworkVariant, ImportReq


def test_variant_imports_strings():
  """Verifies the behavior of variant imports strings."""
  v = FrameworkVariant(api="foo", required_imports=["import os"])
  assert "import os" in v.required_imports


def test_variant_imports_structured():
  """Verifies the behavior of variant imports structured."""
  req = ImportReq(module="numpy", alias="np")
  v = FrameworkVariant(api="foo", required_imports=[req])
  assert len(v.required_imports) == 1
  item = v.required_imports[0]
  assert isinstance(item, ImportReq)
  assert item.module == "numpy"
  assert item.alias == "np"


def test_variant_imports_dict_coercion():
  """Verifies the behavior of variant imports dictionary coercion."""
  data = {"api": "foo", "required_imports": [{"module": "pandas", "alias": "pd"}]}
  v = FrameworkVariant.model_validate(data)
  assert isinstance(v.required_imports[0], ImportReq)
  assert v.required_imports[0].alias == "pd"


def test_variant_imports_mixed():
  """Verifies the behavior of variant imports mixed."""
  imports = ["import cv2", ImportReq(module="PIL", alias="Image")]
  v = FrameworkVariant(api="image_op", required_imports=imports)
  assert len(v.required_imports) == 2
  assert "import cv2" in v.required_imports
  assert isinstance(v.required_imports[1], ImportReq)


def test_variant_imports_default_empty():
  """Verifies the behavior of variant imports default empty."""
  v = FrameworkVariant(api="basic_op")
  assert v.required_imports == []
