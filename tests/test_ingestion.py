"""Test suite for ingestion."""

from unittest.mock import MagicMock, patch

import libcst as cst
import pytest

from ml_switcheroo.core.ingestion import ingest_code
from ml_switcheroo.exceptions import IngestionParseError
from ml_switcheroo.frameworks.base import CustomParserProtocol


class DummyAdapter:
  """Test helper class DummyAdapter."""

  def __init__(self, parser_class: type["DummyParser | DummyFailingParser"]) -> None:
    """Initialize test helper instance.

    Args:
        parser_class: Parser class to instantiate.
    """
    self.parser_class = parser_class

  def create_parser(self, code: str) -> CustomParserProtocol:
    """Execute create parser helper logic.

    Args:
        code: Source code string.

    Returns:
        Instantiated parser.
    """
    return self.parser_class(code)


class DummyParser:
  """Test helper class DummyParser."""

  def __init__(self, code: str) -> None:
    """Initialize test helper instance.

    Args:
        code: Source code string.
    """
    self.code = code

  def parse(self) -> cst.Module:
    """Execute parse helper logic.

    Returns:
        Parsed CST module.
    """
    return cst.parse_module("a = 1")


class DummyFailingParser:
  """Test helper class DummyFailingParser."""

  def __init__(self, code: str) -> None:
    """Initialize test helper instance.

    Args:
        code: Source code string.
    """
    self.code = code

  def parse(self) -> cst.Module:
    """Execute parse helper logic.

    Returns:
        Raises ValueError on parse failure.
    """
    raise ValueError("Failed")


def test_ingest_code_adapter() -> None:
  """Verify ingest code adapter behavior."""
  tracer = MagicMock()
  adapter = DummyAdapter(DummyParser)
  tree = ingest_code("dummy code", "dummy", "torch", adapter, tracer)
  assert isinstance(tree, cst.Module)
  tracer.start_phase.assert_called_with("Custom Ingest", "dummy Parser")


def test_ingest_code_adapter_fail() -> None:
  """Verify ingest code adapter fail behavior."""
  tracer = MagicMock()
  adapter = DummyAdapter(DummyFailingParser)
  with pytest.raises(IngestionParseError):
    ingest_code("dummy code", "dummy", "torch", adapter, tracer)
  tracer.end_phase.assert_called()


@patch("ml_switcheroo.core.ingestion.MlirParser")
@patch("ml_switcheroo.core.ingestion.MlirToPythonGenerator")
def test_ingest_code_mlir(mock_gen: MagicMock, mock_parser: MagicMock) -> None:
  """Verify ingest code mlir behavior.

  Args:
      mock_gen: Mock object for gen.
      mock_parser: Mock object for parser.
  """
  tracer = MagicMock()
  mock_parser_inst = MagicMock()
  mock_parser.return_value = mock_parser_inst
  mock_parser_inst.parse.return_value = "mlir_mod"

  mock_gen_inst = MagicMock()
  mock_gen.return_value = mock_gen_inst
  mock_gen_inst.generate.return_value = cst.parse_module("a = 2")

  tree = ingest_code("module {}", "mlir", "torch", None, tracer)
  assert isinstance(tree, cst.Module)
  tracer.start_phase.assert_called_with("MLIR Ingest", "MLIR Text -> Python CST")


@patch("ml_switcheroo.core.ingestion.StableHloParser")
@patch("ml_switcheroo.core.ingestion.MlirToPythonGenerator")
def test_ingest_code_stablehlo(mock_gen: MagicMock, mock_parser: MagicMock) -> None:
  """Verify ingest code stablehlo behavior.

  Args:
      mock_gen: Mock object for gen.
      mock_parser: Mock object for parser.
  """
  tracer = MagicMock()
  mock_parser_inst = MagicMock()
  mock_parser.return_value = mock_parser_inst
  mock_parser_inst.parse.return_value = "mlir_mod"

  mock_gen_inst = MagicMock()
  mock_gen.return_value = mock_gen_inst
  mock_gen_inst.generate.return_value = cst.parse_module("a = 3")

  tree = ingest_code("module {}", "stablehlo", "torch", None, tracer)
  assert isinstance(tree, cst.Module)
  tracer.start_phase.assert_called_with("STABLEHLO Ingest", "STABLEHLO Text -> Python CST")


@patch("ml_switcheroo.core.ingestion.MlirParser")
def test_ingest_code_mlir_fail(mock_parser: MagicMock) -> None:
  """Verify ingest code mlir fail behavior.

  Args:
      mock_parser: Mock object for parser.
  """
  tracer = MagicMock()
  mock_parser.side_effect = ValueError("fail")
  with pytest.raises(ValueError):
    ingest_code("module {}", "mlir", "torch", None, tracer)
  tracer.end_phase.assert_called()


@patch("ml_switcheroo.core.ingestion.TikzParser")
@patch("ml_switcheroo.core.tikz.parser._logical_from_tikz_graph")
@patch("ml_switcheroo.core.ingestion.PythonBackend")
def test_ingest_code_tikz(mock_backend: MagicMock, mock_logical: MagicMock, mock_parser: MagicMock) -> None:
  """Verify ingest code tikz behavior.

  Args:
      mock_backend: Mock object for backend.
      mock_logical: Mock object for logical.
      mock_parser: Mock object for parser.
  """
  tracer = MagicMock()

  mock_parser_inst = MagicMock()
  mock_parser.return_value = mock_parser_inst
  mock_parser_inst.parse.return_value = "tikz_graph"

  mock_logical.return_value = "logical_graph"

  mock_backend_inst = MagicMock()
  mock_backend.return_value = mock_backend_inst
  mock_backend_inst.generate.return_value = "a = 4"

  tree = ingest_code("\\node", "tikz", "jax", None, tracer)
  assert isinstance(tree, cst.Module)
  tracer.start_phase.assert_called_with("TikZ Ingest", "TikZ Text -> Logical Graph -> Python CST")
  mock_backend.assert_called_with(framework="jax")

  tree2 = ingest_code("\\node", "tikz", "torch", None, tracer)
  assert isinstance(tree2, cst.Module)
  mock_backend.assert_called_with(framework="torch")


@patch("ml_switcheroo.core.ingestion.TikzParser")
def test_ingest_code_tikz_fail(mock_parser: MagicMock) -> None:
  """Verify ingest code tikz fail behavior.

  Args:
      mock_parser: Mock object for parser.
  """
  tracer = MagicMock()
  mock_parser.side_effect = ValueError("fail")
  with pytest.raises(ValueError):
    ingest_code("\\node", "tikz", "torch", None, tracer)
  tracer.end_phase.assert_called()


def test_ingest_code_python() -> None:
  """Verify ingest code python behavior."""
  tracer = MagicMock()
  tree = ingest_code("a = 5", "torch", "torch", None, tracer)
  assert isinstance(tree, cst.Module)
  tracer.start_phase.assert_called_with("Preprocessing", "Parsing & Analysis")


def test_ingest_code_python_fail() -> None:
  """Verify ingest code python fail behavior."""
  tracer = MagicMock()
  with pytest.raises(IngestionParseError):
    ingest_code("a = ", "torch", "torch", None, tracer)
  tracer.end_phase.assert_called()
