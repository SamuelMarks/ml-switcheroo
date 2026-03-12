import pytest
import libcst as cst
from unittest.mock import MagicMock, patch
from ml_switcheroo.core.ingestion import ingest_code


def test_ingestion_adapter_exception():
  adapter = MagicMock()
  # Ensure it's treated as having create_parser
  adapter.create_parser.side_effect = Exception("Adapter Error")
  tracer = MagicMock()
  with pytest.raises(Exception, match="Adapter Error"):
    ingest_code("code", "custom", "jax", adapter, tracer)
  tracer.end_phase.assert_called()


def test_ingestion_mlir_exception():
  tracer = MagicMock()
  with patch("ml_switcheroo.core.ingestion.MlirParser") as MockMlir:
    MockMlir.side_effect = Exception("MLIR Error")
    with pytest.raises(Exception, match="MLIR Error"):
      ingest_code("invalid mlir", "mlir", "jax", None, tracer)
  tracer.end_phase.assert_called()


def test_ingestion_tikz_exception():
  tracer = MagicMock()
  with patch("ml_switcheroo.core.ingestion.TikzParser") as MockTikz:
    MockTikz.side_effect = Exception("TikZ Error")
    with pytest.raises(Exception, match="TikZ Error"):
      ingest_code("invalid tikz", "tikz", "jax", None, tracer)
  tracer.end_phase.assert_called()


def test_ingestion_tikz_success():
  tracer = MagicMock()
  with patch("ml_switcheroo.core.ingestion.TikzParser") as MockTikz:
    with patch("ml_switcheroo.core.ingestion.PythonBackend") as MockBackend:
      mock_parser = MagicMock()
      mock_parser.parse.return_value = "Graph"
      MockTikz.return_value = mock_parser

      mock_backend = MagicMock()
      mock_backend.generate.return_value = "def my_func(): pass"
      MockBackend.return_value = mock_backend

      tree = ingest_code("dummy code", "tikz", "jax", None, tracer)
      assert tree is not None

      # test torch target
      ingest_code("dummy code", "tikz", "keras", None, tracer)


def test_ingestion_python_exception():
  tracer = MagicMock()
  with pytest.raises(Exception):
    ingest_code("def foo(:", "torch", "jax", None, tracer)
  tracer.end_phase.assert_called()
