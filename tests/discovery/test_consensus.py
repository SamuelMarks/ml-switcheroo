"""Test suite for the Consensus module."""

from unittest.mock import patch, MagicMock
from ml_switcheroo.discovery.consensus import ConsensusEngine


def test_consensus_engine_init():
  """Verifies the behavior of consensus engine initialization."""
  engine = ConsensusEngine(["torch", "jax.numpy"])
  assert engine.frameworks == ["torch", "jax.numpy"]
  assert engine.vocabulary == {}


def test_consensus_engine_normalize():
  """Verifies the behavior of consensus engine normalize."""
  engine = ConsensusEngine([])
  assert engine.normalize("torch_add") == "add"
  assert engine.normalize("cross_entropy_loss") == "crossentropy"
  assert engine.normalize("jax_relu_fn") == "relu"
  assert engine.normalize("batch_norm") == "batchnorm"
  assert engine.normalize("TF_MSE_Loss") == "mse"


def test_consensus_engine_ingest_success():
  """Verifies the behavior of consensus engine ingest successfully."""
  engine = ConsensusEngine(["json"])
  engine.ingest()
  assert len(engine.vocabulary) > 0
  assert "load" in engine.vocabulary or "dump" in engine.vocabulary


def test_consensus_engine_ingest_import_error():
  """Verifies the behavior of consensus engine ingest import correctly handling an error."""
  engine = ConsensusEngine(["nonexistent_fw"])
  engine.ingest()
  assert len(engine.vocabulary) == 0


def test_consensus_engine_scan_module_recursion():
  """Verifies the behavior of consensus engine scan module recursion."""
  engine = ConsensusEngine([])

  class DummyModule:
    """Dummy."""

    __name__ = "dummy"

  engine._scan_module(DummyModule(), "dummy", depth=3)
  assert len(engine.vocabulary) == 0


def test_consensus_engine_scan_module_error():
  """Verifies the behavior of consensus engine scan module correctly handling an error."""
  engine = ConsensusEngine([])
  with patch("inspect.getmembers", side_effect=Exception("boom")):
    engine._scan_module(MagicMock(), "dummy")
  assert len(engine.vocabulary) == 0


def test_consensus_engine_scan_submodule():
  """Verifies the behavior of consensus engine scan submodule."""
  engine = ConsensusEngine([])

  class Sub:
    """Sub."""

    __name__ = "dummy.sub"

  class Dummy:
    """Dummy."""

    __name__ = "dummy"
    sub = Sub()

  with (
    patch("inspect.getmembers") as mock_gm,
    patch("inspect.ismodule") as mock_ism,
    patch("inspect.isfunction") as mock_isf,
  ):
    mock_gm.return_value = [("sub", Sub()), ("_priv", Sub()), ("myfunc", lambda: None)]

    def mock_is_m(obj):
      """Mock."""
      return isinstance(obj, Sub)

    mock_ism.side_effect = mock_is_m
    mock_isf.return_value = True
    engine._scan_module(Dummy(), "dummy")
  assert "myfunc" in engine.vocabulary


def test_consensus_engine_cluster():
  """Verifies the behavior of consensus engine cluster."""
  engine = ConsensusEngine([])
  engine.vocabulary = {
    "relu": ["torch.relu", "jax.nn.relu"],
    "relufn": ["other.relu_fn"],
    "add": ["torch.add"],
    "addition": ["other.addition"],
    "nonmatch": ["something"],
  }
  clusters = engine.cluster(threshold=0.8)
  assert "Relu" in clusters or "Relufn" in clusters
  assert "Add" in clusters


def test_consensus_engine_cluster_no_matches():
  """Verifies the behavior of consensus engine cluster no matches."""
  engine = ConsensusEngine([])
  engine.vocabulary = {"a": ["a"]}
  clusters = engine.cluster()
  assert "A" in clusters


def test_consensus_engine_cluster_difflib_no_matches():
  """Verifies the behavior of consensus engine cluster difflib no matches."""
  engine = ConsensusEngine([])
  engine.vocabulary = {"a": ["a"]}
  with patch("difflib.get_close_matches", return_value=[]):
    clusters = engine.cluster()
    assert len(clusters) == 0
