from unittest.mock import patch
from ml_switcheroo.cli.commands import handle_gen_weight_script
from pathlib import Path


def test_handle_gen_weight_script():
  with patch("ml_switcheroo.cli.commands.WeightScriptGenerator") as MockGen:
    with patch("ml_switcheroo.cli.commands.RuntimeConfig"):
      with patch("ml_switcheroo.cli.commands.SemanticsManager"):
        MockGen.return_value.generate.return_value = True
        assert handle_gen_weight_script(Path("in.py"), Path("out.py"), "torch", "jax") == 0

        MockGen.return_value.generate.return_value = False
        assert handle_gen_weight_script(Path("in.py"), Path("out.py"), "torch", "jax") == 1
