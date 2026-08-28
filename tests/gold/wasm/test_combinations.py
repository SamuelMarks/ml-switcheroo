"""Test suite for the Combinations module."""

import pytest
import typing
import json
from ml_switcheroo.core.engine import ASTEngine, ConversionResult
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.sphinx_ext.registry import scan_registry


def get_examples() -> dict[str, typing.Any]:
  """Gets examples."""
  _, examples_json, _ = scan_registry()
  return typing.cast(dict[str, typing.Any], json.loads(examples_json))


EXAMPLES: dict[str, typing.Any] = get_examples()


@pytest.fixture(scope="module")
def semantics() -> SemanticsManager:
  """Helper to semantics."""
  return SemanticsManager()


@pytest.mark.parametrize("example_key", sorted(EXAMPLES.keys()))
def test_wasm_combination(example_key: str, semantics: SemanticsManager, snapshot: typing.Any) -> None:
  """Verifies the behavior of wasm combination."""
  ex: dict[str, typing.Any] = EXAMPLES[example_key]
  src_fw: str = ex["srcFw"]
  src_flavour: typing.Optional[str] = ex.get("srcFlavour")
  tgt_fw: str = ex["tgtFw"]
  tgt_flavour: typing.Optional[str] = ex.get("tgtFlavour")
  config_fwd = RuntimeConfig(
    source_framework=src_fw,
    source_flavour=src_flavour,
    target_framework=tgt_fw,
    target_flavour=tgt_flavour,
    strict_mode=False,
  )
  engine_fwd = ASTEngine(semantics, config_fwd)
  result_fwd: ConversionResult = engine_fwd.run(ex["code"])
  assert result_fwd.success, f"Forward transpilation failed: {result_fwd.errors}"
  config_bwd = RuntimeConfig(
    source_framework=tgt_fw,
    source_flavour=tgt_flavour,
    target_framework=src_fw,
    target_flavour=src_flavour,
    strict_mode=False,
  )
  engine_bwd = ASTEngine(semantics, config_bwd)
  result_bwd: ConversionResult = engine_bwd.run(result_fwd.code)
  assert result_bwd.success, f"Backward transpilation failed: {result_bwd.errors}"
  snapshot.assert_match(f"--- Forward ---\n{result_fwd.code}\n--- Backward ---\n{result_bwd.code}", extension="txt")
