def test_injector_fw_missing():
  from ml_switcheroo.tools.injector_fw.core import FrameworkInjector
  from ml_switcheroo.core.dsl import FrameworkVariant

  variant = FrameworkVariant(api="foo")
  injector = FrameworkInjector("jax", "bar", variant)

  import json

  # Force idempotency match
  with __import__("unittest.mock").mock.patch.object(
    injector, "_load_current", return_value={"bar": variant.model_dump(exclude_none=True)}
  ):
    assert injector.inject(dry_run=False) is True

  # Force OSError on write
  with __import__("unittest.mock").mock.patch.object(injector, "_load_current", return_value={"other": {}}):
    with __import__("unittest.mock").mock.patch("builtins.open", side_effect=OSError("fail")):
      assert injector.inject(dry_run=False) is False

  # Force JSONDecodeError on read
  with __import__("unittest.mock").mock.patch(
    "builtins.open", __import__("unittest.mock").mock.mock_open(read_data="invalid json")
  ):
    assert injector._load_current() == {}


def test_injector_fw_updating():
  from ml_switcheroo.tools.injector_fw.core import FrameworkInjector
  from ml_switcheroo.core.dsl import FrameworkVariant

  variant = FrameworkVariant(api="foo")
  injector = FrameworkInjector("jax", "bar", variant)

  with __import__("unittest.mock").mock.patch.object(injector, "_load_current", return_value={"bar": {"api": "old_foo"}}):
    with __import__("unittest.mock").mock.patch("builtins.open", __import__("unittest.mock").mock.mock_open()):
      assert injector.inject(dry_run=False) is True
