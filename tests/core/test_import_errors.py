"""Test suite for the Import Errors module."""

import contextlib
import importlib
import warnings
from unittest.mock import patch

with warnings.catch_warnings():
  warnings.simplefilter("ignore", DeprecationWarning)
  for _mod_name in ("flax", "jax", "tensorflow", "torch"):
    with contextlib.suppress(ImportError):
      importlib.import_module(_mod_name)


def test_import_errors_frameworks() -> None:
  """Verify `importlib.reload` handles import errors frameworks as expected."""
  with patch.dict(
    "sys.modules",
    {
      "jax": None,
      "jax.numpy": None,
      "flax": None,
      "flax.nnx": None,
      "mlx": None,
      "mlx.core": None,
      "mlx.nn": None,
      "paxml": None,
      "praxis": None,
      "tensorflow": None,
      "torch": None,
      "torch.nn": None,
      "optax": None,
      "torch.utils.data": None,
    },
  ):
    import ml_switcheroo.frameworks.flax_nnx

    importlib.reload(ml_switcheroo.frameworks.flax_nnx)
    import ml_switcheroo.frameworks.jax

    importlib.reload(ml_switcheroo.frameworks.jax)
    import ml_switcheroo.frameworks.mlx

    importlib.reload(ml_switcheroo.frameworks.mlx)
    import ml_switcheroo.frameworks.paxml

    importlib.reload(ml_switcheroo.frameworks.paxml)
    import ml_switcheroo.frameworks.tensorflow

    importlib.reload(ml_switcheroo.frameworks.tensorflow)
    import ml_switcheroo.frameworks.torch

    importlib.reload(ml_switcheroo.frameworks.torch)

    import ml_switcheroo.frameworks.jax

    a1: ml_switcheroo.frameworks.jax.JaxCoreAdapter = ml_switcheroo.frameworks.jax.JaxCoreAdapter()
    res = a1.convert([1, 2, 3])
  assert res == [1, 2, 3]


def test_import_error_fallbacks_in_file_loader_and_consensus() -> None:
  """Verify `ImportError` fallback helpers in `file_loader` and `consensus` when `ml_ecosystem_snapshots` is absent."""
  import ml_switcheroo.discovery.consensus as consensus_mod
  import ml_switcheroo.semantics.file_loader as file_loader_mod

  with patch.dict("sys.modules", {"ml_ecosystem_snapshots.utils": None}):
    reloaded_fl = importlib.reload(file_loader_mod)
    assert reloaded_fl.strip_sphinx_roles(None) is None
    assert reloaded_fl.strip_sphinx_roles(":class:`~torch.Tensor`") == "torch.Tensor"
    assert reloaded_fl.resolve_griffe_parser("google") is None
    assert reloaded_fl.parse_docstring_with_griffe("doc") == []
  importlib.reload(file_loader_mod)

  with patch.dict("sys.modules", {"ml_ecosystem_snapshots.grounding.engine": None}):
    reloaded_con = importlib.reload(consensus_mod)
    assert reloaded_con.compute_levenshtein("a", "ab") == 1
    assert reloaded_con.compute_levenshtein("abc", "") == 3
    assert reloaded_con.compute_levenshtein("abc", "abd") == 1
  importlib.reload(consensus_mod)
