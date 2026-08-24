"""Test module."""

import libcst as cst
from ml_switcheroo.core.import_fixer.utils import (
  get_root_name,
  create_dotted_name,
  get_signature,
  is_docstring,
  is_future_import,
)


def test_get_root_name():
  """Test element."""
  node = cst.Name("torch")
  assert get_root_name(node) == "torch"

  node = cst.Attribute(value=cst.Name("torch"), attr=cst.Name("nn"))
  assert get_root_name(node) == "torch"

  node = cst.Attribute(value=cst.Attribute(value=cst.Name("torch"), attr=cst.Name("nn")), attr=cst.Name("Module"))
  assert get_root_name(node) == "torch"

  # test fallback
  assert get_root_name(cst.Pass()) == ""


def test_create_dotted_name():
  """Test element."""
  node = create_dotted_name("torch")
  assert isinstance(node, cst.Name)
  assert node.value == "torch"

  node = create_dotted_name("torch.nn.Module")
  assert isinstance(node, cst.Attribute)
  assert node.attr.value == "Module"
  assert isinstance(node.value, cst.Attribute)
  assert node.value.attr.value == "nn"
  assert isinstance(node.value.value, cst.Name)
  assert node.value.value.value == "torch"


def test_get_signature():
  """Test element."""
  # Test simple statement unwrapping
  import_stmt = cst.SimpleStatementLine(body=[cst.Import(names=[cst.ImportAlias(name=cst.Name("torch"))])])
  sig = get_signature(import_stmt)
  assert sig == "import torch"

  import_stmt_multi_space = cst.Import(names=[cst.ImportAlias(name=cst.Name("torch"))])
  # The normaliser should trim spaces but libcst generates exactly what we ask.
  # We will test normal behavior.
  sig = get_signature(import_stmt_multi_space)
  assert sig == "import torch"


def test_is_docstring():
  """Test element."""
  # A standard docstring expression
  doc_node = cst.SimpleStatementLine(body=[cst.Expr(value=cst.SimpleString('"""Doc"""'))])
  assert is_docstring(doc_node, 0) is True
  # Not docstring because idx is not 0
  assert is_docstring(doc_node, 1) is False

  # Not docstring because it's not string
  non_str_node = cst.SimpleStatementLine(body=[cst.Expr(value=cst.Name("x"))])
  assert is_docstring(non_str_node, 0) is False

  # Not a simple statement line
  assert is_docstring(cst.Pass(), 0) is False

  # Concatenated string
  concat_node = cst.SimpleStatementLine(
    body=[cst.Expr(value=cst.ConcatenatedString(left=cst.SimpleString('"A"'), right=cst.SimpleString('"B"')))]
  )
  assert is_docstring(concat_node, 0) is True


def test_is_future_import():
  """Test element."""
  future_node = cst.SimpleStatementLine(
    body=[cst.ImportFrom(module=cst.Name("__future__"), names=[cst.ImportAlias(name=cst.Name("annotations"))])]
  )
  assert is_future_import(future_node) is True

  regular_from_node = cst.SimpleStatementLine(
    body=[cst.ImportFrom(module=cst.Name("torch"), names=[cst.ImportAlias(name=cst.Name("nn"))])]
  )
  assert is_future_import(regular_from_node) is False

  regular_import_node = cst.SimpleStatementLine(body=[cst.Import(names=[cst.ImportAlias(name=cst.Name("torch"))])])
  assert is_future_import(regular_import_node) is False

  non_stmt = cst.Pass()
  assert is_future_import(non_stmt) is False
