"""Unit tests for multi-format source ingestion (`ml_switcheroo.core.ingestion`)."""

import libcst as cst
import pytest

from ml_switcheroo.core.ingestion import ingest_code
from ml_switcheroo.core.tracer import TraceEventType, TraceLogger
from ml_switcheroo.exceptions import IngestionParseError, SyntaxParseError
from ml_switcheroo.frameworks.base import CustomParserProtocol

_BS = chr(92)
_NL = chr(10)


class _WorkingCustomParser:
  """Custom parser returning `a = 1`."""

  def __init__(self, code: str) -> None:
    """Store raw `code`.

    Args:
        code: Raw input source text.
    """
    self.code = code

  def parse(self) -> cst.Module:
    """Parse into `cst.Module('a = 1')`.

    Returns:
        Parsed `cst.Module`.
    """
    return cst.parse_module("a = 1")


class _FailingCustomParser:
  """Custom parser raising `SyntaxParseError`."""

  def __init__(self, code: str) -> None:
    """Store raw `code`.

    Args:
        code: Raw input source text.
    """
    self.code = code

  def parse(self) -> cst.Module:
    """Raise `SyntaxParseError` during parsing.

    Returns:
        Never returns; raises `SyntaxParseError`.

    Raises:
        SyntaxParseError: Always raised.
    """
    raise SyntaxParseError("custom parse failure")


class _CustomAdapter:
  """Typed adapter stub exposing `create_parser`."""

  def __init__(self, *, fail: bool = False) -> None:
    """Configure whether `create_parser` returns a failing parser.

    Args:
        fail: When `True`, returns `_FailingCustomParser`.
    """
    self.fail = fail

  def create_parser(self, code: str) -> CustomParserProtocol:
    """Instantiate custom parser for `code`.

    Args:
        code: Raw input text.

    Returns:
        Custom parser instance.
    """
    return _FailingCustomParser(code) if self.fail else _WorkingCustomParser(code)


def test_ingest_code_custom_adapter_and_python_paths() -> None:
  """Verify custom adapter ingestion, standard Python parsing, and `IngestionParseError` wrapping."""
  tracer = TraceLogger()
  mod = ingest_code("raw text", "custom_dsl", "torch", _CustomAdapter(fail=False), tracer)
  assert mod.code == "a = 1"
  assert any(e["description"] == "Custom Ingest" for e in tracer.export())

  err_tracer = TraceLogger()
  with pytest.raises(IngestionParseError, match="custom parse failure"):
    ingest_code("raw text", "custom_dsl", "torch", _CustomAdapter(fail=True), err_tracer)
  assert any(e["type"] == TraceEventType.PHASE_END for e in err_tracer.export())

  py_tracer = TraceLogger()
  py_mod = ingest_code("a = 5", "torch", "jax", None, py_tracer)
  assert py_mod.code == "a = 5"

  with pytest.raises(IngestionParseError):
    ingest_code("def broken(:", "torch", "jax", None, TraceLogger())

  from unittest.mock import MagicMock

  bad_parser = MagicMock(spec=CustomParserProtocol)
  bad_parser.parse.return_value = cst.Name("not_a_module")

  class _NonModuleAdapter:
    """Adapter returning a parser whose `parse()` yields a non-`cst.Module` node."""

    def create_parser(self, _code: str) -> CustomParserProtocol:
      """Return `bad_parser`.

      Args:
          _code: Source string.

      Returns:
          `CustomParserProtocol` mock returning `cst.Name`.
      """
      return bad_parser

  with pytest.raises(IngestionParseError, match=r"expected cst\.Module"):
    ingest_code("raw", "custom_dsl", "torch", _NonModuleAdapter(), TraceLogger())


def test_ingest_code_real_mlir_stablehlo_and_tikz_frontends() -> None:
  """Verify real MLIR, StableHLO, and TikZ ingestion to Python CST and syntax failure wrapping."""
  mlir_src = _NL.join(
    [
      'sw.func {sym_name = "f"} {',
      "  %0 = sw.constant {value = 42}",
      "  sw.return %0",
      "}",
    ]
  )
  mlir_mod = ingest_code(mlir_src, "mlir", "torch", None, TraceLogger())
  assert "def f():" in mlir_mod.code
  assert "return 42" in mlir_mod.code

  shlo_mod = ingest_code("module {}", "stablehlo", "jax", None, TraceLogger())
  assert isinstance(shlo_mod, cst.Module)

  with pytest.raises(IngestionParseError):
    ingest_code("%0 = ", "mlir", "jax", None, TraceLogger())

  tikz_src = (
    f"{_BS}begin{{tikzpicture}}"
    f"{_BS}node (in1) at (0,0) {{Input}};"
    f"{_BS}node (out1) at (0,-1) {{Output}};"
    f"{_BS}draw (in1) -- (out1);"
    f"{_BS}end{{tikzpicture}}"
  )
  tikz_jax = ingest_code(tikz_src, "tikz", "jax", None, TraceLogger())
  assert isinstance(tikz_jax, cst.Module)
  assert "class " in tikz_jax.code or "def " in tikz_jax.code

  tikz_torch = ingest_code(tikz_src, "tikz", "torch", None, TraceLogger())
  assert isinstance(tikz_torch, cst.Module)

  with pytest.raises(IngestionParseError):
    ingest_code(f"{_BS}node;", "tikz", "torch", None, TraceLogger())

  ir_json = '{"name": "Net", "nodes": {}, "edges": []}'
  ir_mod = ingest_code(ir_json, "ir", "jax", None, TraceLogger())
  assert isinstance(ir_mod, cst.Module)

  with pytest.raises(IngestionParseError):
    ingest_code("{bad_json", "ir", "jax", None, TraceLogger())
