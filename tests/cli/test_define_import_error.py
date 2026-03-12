import sys
from unittest.mock import patch
import importlib


def test_define_yaml_import_error():
  with patch.dict(sys.modules, {"yaml": None}):
    # Need to remove from sys.modules to reload it
    if "ml_switcheroo.cli.handlers.define" in sys.modules:
      del sys.modules["ml_switcheroo.cli.handlers.define"]
    import ml_switcheroo.cli.handlers.define

    assert ml_switcheroo.cli.handlers.define.yaml is None
