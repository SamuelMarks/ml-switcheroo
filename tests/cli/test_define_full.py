from ml_switcheroo.cli.handlers.define import handle_define
import pytest
import os
from unittest.mock import patch, MagicMock


@patch("ml_switcheroo.cli.handlers.define._inject_hub")
@patch("ml_switcheroo.cli.handlers.define._inject_spokes")
@patch("ml_switcheroo.cli.handlers.define._scaffold_plugins")
@patch("ml_switcheroo.cli.handlers.define.StandardsInjector")
@patch("ml_switcheroo.cli.handlers.define.SimulatedReflection")
def test_define_full(MockSim, MockStand, MockPlugin, MockFW, MockHub, tmp_path):
  f = tmp_path / "def.yaml"
  f.write_text(
    "name: Abs\noperation: abs\ndescription: ''\nvariants: {torch: {name: torch.abs, requires_plugin: 'AbsPlugin'}}\ntier: MATH\nstd_args: [x]"
  )

  MockHub.return_value = True
  assert handle_define(f, dry_run=False, no_test_gen=False) == 0

  assert MockHub.called
  assert MockFW.called
  assert MockPlugin.called
