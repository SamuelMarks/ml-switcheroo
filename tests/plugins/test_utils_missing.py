"""Test suite for the Utils Missing module."""


def test_utils_missing() -> None:
  """Verifies the behavior of utilities missing."""
  import libcst as cst
  from ml_switcheroo.plugins.utils import is_framework_module_node, _extract_root_name
  from ml_switcheroo.core.hooks import HookContext

  class DummyAlias:
    """Dummy Alias class for testing purposes."""

    def model_dump(self) -> dict:
      """Mock implementation of model dump.

      Returns:
          dict: Mock data.
      """
      return {"name": "pd"}

  class DummyConf:
    """Dummy Conf class for testing purposes."""

    alias: DummyAlias = DummyAlias()

  class DummyConfNoDump:
    """Dummy conf no dump."""

    alias: object = object()

  class DummySM:
    """Dummy S M class for testing purposes."""

    _source_registry: dict = {"torch.nn": {}}
    framework_configs: dict = {
      "pandas": DummyConf(),
      "other": DummyConfNoDump(),
      "direct_dict": {"alias": {"name": "dd"}},
      "direct_dict_no_name": {"alias": {}},
      "no_alias": {},
      "target_fw": {},
      "tf": {},
    }

  class DummyConfigObj:
    """Dummy Config Obj class for testing purposes."""

    source_framework: str = "s"
    target_framework: str = "target_fw"
    effective_source: str = "s"
    effective_target: str = "target_fw"

  ctx: HookContext = HookContext(DummySM(), DummyConfigObj())  # type: ignore
  assert is_framework_module_node(cst.Integer("1"), ctx) is False
  assert is_framework_module_node(cst.Name("pd"), ctx) is True
  assert is_framework_module_node(cst.Name("torch"), ctx) is True
  assert is_framework_module_node(cst.Name("target_fw"), ctx) is True
  assert is_framework_module_node(cst.Name("tf"), ctx) is True
  assert is_framework_module_node(cst.Name("dd"), ctx) is True

  assert _extract_root_name(cst.Integer("1")) is None

  # Also test complex extraction
  attr_node: cst.Attribute = cst.Attribute(value=cst.Name("tf"), attr=cst.Name("math"))
  assert _extract_root_name(attr_node) == "tf"

  # And unknown root
  assert is_framework_module_node(cst.Name("unknown"), ctx) is False
