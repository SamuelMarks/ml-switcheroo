"""Docstring."""

from unittest.mock import MagicMock, patch

import libcst as cst
import pytest

from ml_switcheroo.core.ingestion import ingest_code
from ml_switcheroo.frameworks.base import FrameworkAdapter


class DummyAdapter(FrameworkAdapter):
  """Docstring."""

  def __init__(self, parser_class: type) -> None:
    """Docstring."""
    self.parser_class: type = parser_class

  def create_parser(self, code: str) -> object:
    """Docstring."""
    return self.parser_class(code)


class DummyParser:
  """Docstring."""

  def __init__(self, code: str) -> None:
    """Docstring."""
    self.code: str = code

  def parse(self) -> cst.Module:
    """Docstring."""
    return cst.parse_module("a = 1")


class DummyFailingParser:
  """Docstring."""

  def __init__(self, code: str) -> None:
    """Docstring."""
    self.code: str = code

  def parse(self) -> cst.Module:
    """Docstring."""
    raise ValueError("Failed")


def test_ingest_code_adapter() -> None:
  """Docstring."""
  tracer: MagicMock = MagicMock()
  adapter: DummyAdapter = DummyAdapter(DummyParser)
  tree: cst.Module = ingest_code("dummy code", "dummy", "torch", adapter, tracer)
  assert isinstance(tree, cst.Module)
  tracer.start_phase.assert_called_with("Custom Ingest", "dummy Parser")


def test_ingest_code_adapter_fail() -> None:
  """Docstring."""
  tracer: MagicMock = MagicMock()
  adapter: DummyAdapter = DummyAdapter(DummyFailingParser)
  with pytest.raises(ValueError):
    ingest_code("dummy code", "dummy", "torch", adapter, tracer)
  tracer.end_phase.assert_called()


@patch("ml_switcheroo.core.ingestion.MlirParser")
@patch("ml_switcheroo.core.ingestion.MlirToPythonGenerator")
def test_ingest_code_mlir(mock_gen: MagicMock, mock_parser: MagicMock) -> None:
  """Docstring."""
  tracer: MagicMock = MagicMock()
  mock_parser_inst: MagicMock = MagicMock()
  mock_parser.return_value = mock_parser_inst
  mock_parser_inst.parse.return_value = "mlir_mod"

  mock_gen_inst: MagicMock = MagicMock()
  mock_gen.return_value = mock_gen_inst
  mock_gen_inst.generate.return_value = cst.parse_module("a = 2")

  tree: cst.Module = ingest_code("module {}", "mlir", "torch", None, tracer)
  assert isinstance(tree, cst.Module)
  tracer.start_phase.assert_called_with("MLIR Ingest", "MLIR Text -> Python CST")


@patch("ml_switcheroo.core.ingestion.StableHloParser")
@patch("ml_switcheroo.core.ingestion.MlirToPythonGenerator")
def test_ingest_code_stablehlo(mock_gen: MagicMock, mock_parser: MagicMock) -> None:
  """Docstring."""
  tracer: MagicMock = MagicMock()
  mock_parser_inst: MagicMock = MagicMock()
  mock_parser.return_value = mock_parser_inst
  mock_parser_inst.parse.return_value = "mlir_mod"

  mock_gen_inst: MagicMock = MagicMock()
  mock_gen.return_value = mock_gen_inst
  mock_gen_inst.generate.return_value = cst.parse_module("a = 3")

  tree: cst.Module = ingest_code("module {}", "stablehlo", "torch", None, tracer)
  assert isinstance(tree, cst.Module)
  tracer.start_phase.assert_called_with("STABLEHLO Ingest", "STABLEHLO Text -> Python CST")


@patch("ml_switcheroo.core.ingestion.MlirParser")
def test_ingest_code_mlir_fail(mock_parser: MagicMock) -> None:
  """Docstring."""
  tracer: MagicMock = MagicMock()
  mock_parser.side_effect = ValueError("fail")
  with pytest.raises(ValueError):
    ingest_code("module {}", "mlir", "torch", None, tracer)
  tracer.end_phase.assert_called()


@patch("ml_switcheroo.core.ingestion.TikzParser")
@patch("ml_switcheroo.core.tikz.parser._logical_from_tikz_graph")
@patch("ml_switcheroo.core.ingestion.PythonBackend")
def test_ingest_code_tikz(mock_backend: MagicMock, mock_logical: MagicMock, mock_parser: MagicMock) -> None:
  """Docstring."""
  tracer: MagicMock = MagicMock()

  mock_parser_inst: MagicMock = MagicMock()
  mock_parser.return_value = mock_parser_inst
  mock_parser_inst.parse.return_value = "tikz_graph"

  mock_logical.return_value = "logical_graph"

  mock_backend_inst: MagicMock = MagicMock()
  mock_backend.return_value = mock_backend_inst
  mock_backend_inst.generate.return_value = "a = 4"

  tree: cst.Module = ingest_code("\\node", "tikz", "jax", None, tracer)
  assert isinstance(tree, cst.Module)
  tracer.start_phase.assert_called_with("TikZ Ingest", "TikZ Text -> Logical Graph -> Python CST")
  mock_backend.assert_called_with(framework="jax")

  tree2: cst.Module = ingest_code("\\node", "tikz", "torch", None, tracer)
  assert isinstance(tree2, cst.Module)
  mock_backend.assert_called_with(framework="torch")


@patch("ml_switcheroo.core.ingestion.TikzParser")
def test_ingest_code_tikz_fail(mock_parser: MagicMock) -> None:
  """Docstring."""
  tracer: MagicMock = MagicMock()
  mock_parser.side_effect = ValueError("fail")
  with pytest.raises(ValueError):
    ingest_code("\\node", "tikz", "torch", None, tracer)
  tracer.end_phase.assert_called()


def test_ingest_code_python() -> None:
  """Docstring."""
  tracer: MagicMock = MagicMock()
  tree: cst.Module = ingest_code("a = 5", "torch", "torch", None, tracer)
  assert isinstance(tree, cst.Module)
  tracer.start_phase.assert_called_with("Preprocessing", "Parsing & Analysis")


def test_ingest_code_python_fail() -> None:
  """Docstring."""
  tracer: MagicMock = MagicMock()
  with pytest.raises(Exception):
    ingest_code("a = ", "torch", "torch", None, tracer)
  tracer.end_phase.assert_called()
