"""
Tests for ODL Schema Extension: Explicit Import Dependencies.

Verifies that the FrameworkVariant schema correctly accepts and validates
declarative import requirements, enabling plugins to access non-standard dependencies.

Updated for Feature 13: Structured Import Aliasing.
"""

import pytest
from ml_switcheroo.core.dsl import FrameworkVariant, ImportReq
from pydantic import ValidationError


def test_variant_imports_strings():
  """
  Verify that 'required_imports' stores a list of import strings (Legacy support).
  """
  v = FrameworkVariant(api="foo", required_imports=["import os"])
  assert "import os" in v.required_imports


def test_variant_imports_structured():
  """
  Verify support for structured ImportReq objects {module, alias}.
  """
  req = ImportReq(module="numpy", alias="np")
  v = FrameworkVariant(api="foo", required_imports=[req])

  assert len(v.required_imports) == 1
  item = v.required_imports[0]
  assert isinstance(item, ImportReq)
  assert item.module == "numpy"
  assert item.alias == "np"


def test_variant_imports_dict_coercion():
  """
  Verify Pydantic coerces dicts to ImportReq objects automatically.
  This mimics loading from JSON.
  """
  data = {"api": "foo", "required_imports": [{"module": "pandas", "alias": "pd"}]}
  v = FrameworkVariant.model_validate(data)

  assert isinstance(v.required_imports[0], ImportReq)
  assert v.required_imports[0].alias == "pd"


def test_variant_imports_mixed():
  """
  Verify that multiple imports of mixed types (str and object) work.
  """
  imports = ["import cv2", ImportReq(module="PIL", alias="Image")]
  v = FrameworkVariant(api="image_op", required_imports=imports)

  assert len(v.required_imports) == 2
  assert "import cv2" in v.required_imports
  assert isinstance(v.required_imports[1], ImportReq)


def test_variant_imports_default_empty():
  """
  Verify default behavior is an empty list (safe for most ops).
  """
  v = FrameworkVariant(api="basic_op")
  assert v.required_imports == []
