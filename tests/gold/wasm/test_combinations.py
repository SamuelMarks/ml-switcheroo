import pytest
import json

from ml_switcheroo.core.engine import ASTEngine
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.sphinx_ext.registry import scan_registry


def get_examples():
  _, examples_json, _ = scan_registry()
  return json.loads(examples_json)


EXAMPLES = get_examples()


@pytest.fixture(scope="module")
def semantics():
  return SemanticsManager()


@pytest.mark.skip
@pytest.mark.parametrize("example_key", sorted(EXAMPLES.keys()))
def test_wasm_combination(example_key, semantics, snapshot):
  ex = EXAMPLES[example_key]
  src_fw = ex["srcFw"]
  src_flavour = ex["srcFlavour"]
  tgt_fw = ex["tgtFw"]
  tgt_flavour = ex["tgtFlavour"]

  # 1. Forward
  config_fwd = RuntimeConfig(
    source_framework=src_fw,
    source_flavour=src_flavour,
    target_framework=tgt_fw,
    target_flavour=tgt_flavour,
    strict_mode=False,
  )
  engine_fwd = ASTEngine(semantics, config_fwd)
  result_fwd = engine_fwd.run(ex["code"])
  assert result_fwd.success, f"Forward transpilation failed: {result_fwd.errors}"

  # 2. Backward
  config_bwd = RuntimeConfig(
    source_framework=tgt_fw,
    source_flavour=tgt_flavour,
    target_framework=src_fw,
    target_flavour=src_flavour,
    strict_mode=False,
  )
  engine_bwd = ASTEngine(semantics, config_bwd)
  result_bwd = engine_bwd.run(result_fwd.code)
  assert result_bwd.success, f"Backward transpilation failed: {result_bwd.errors}"

  # Snapshot the roundtrip
  snapshot.assert_match(f"--- Forward ---\n{result_fwd.code}\n--- Backward ---\n{result_bwd.code}", extension="txt")
