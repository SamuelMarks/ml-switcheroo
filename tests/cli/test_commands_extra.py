import pytest
from unittest.mock import patch, MagicMock
from ml_switcheroo.cli.commands import handle_gen_weight_script
from pathlib import Path


def test_handle_gen_weight_script():
  with patch("ml_switcheroo.cli.commands.WeightScriptGenerator") as MockGen:
    with patch("ml_switcheroo.cli.commands.RuntimeConfig") as MockConfig:
      with patch("ml_switcheroo.cli.commands.SemanticsManager") as MockSemantics:
        MockGen.return_value.generate.return_value = True
        assert handle_gen_weight_script(Path("in.py"), Path("out.py"), "torch", "jax") == 0

        MockGen.return_value.generate.return_value = False
        assert handle_gen_weight_script(Path("in.py"), Path("out.py"), "torch", "jax") == 1
