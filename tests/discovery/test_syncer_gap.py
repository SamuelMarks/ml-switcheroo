def test_syncer_fallback_case_insensitive():
  from ml_switcheroo.discovery.syncer import FrameworkSyncer

  syncer = FrameworkSyncer()

  # Target object with weird casing
  import types

  MockTarget = types.ModuleType("mock_fw_case")
  MockTarget.wEirD_Op = lambda arg1, arg2: None

  import sys

  sys.modules["mock_fw_case"] = MockTarget

  spec = {"Weird_Op": {"std_args": ["arg1", "arg2"], "variants": {}}}

  # We mock _get_search_modules to return our mock
  with (
    __import__("unittest.mock").mock.patch("ml_switcheroo.discovery.syncer.get_adapter", return_value=None),
    __import__("unittest.mock").mock.patch("importlib.import_module", return_value=sys.modules["mock_fw_case"]),
  ):
    syncer.sync(spec, "mock_fw_case")

  assert "mock_fw_case" in spec["Weird_Op"]["variants"]
  assert spec["Weird_Op"]["variants"]["mock_fw_case"]["api"] == "mock_fw_case.wEirD_Op"


def test_syncer_compatibility_check_fallback():
  from ml_switcheroo.discovery.syncer import FrameworkSyncer

  syncer = FrameworkSyncer()

  # line 151: if not sig, return False
  # We can pass something that throws ValueError for inspect.signature (e.g. str)
  assert syncer._is_compatible(str, ["arg1"]) is True


def test_syncer_extract_names_tuple():
  from ml_switcheroo.discovery.syncer import FrameworkSyncer

  syncer = FrameworkSyncer()

  # 132-136: tuples
  class MockClass:
    def forward(self, arg1):
      pass

  assert syncer._is_compatible(MockClass, ["arg1"]) is True

  class MockClass2:
    def call(self, arg1):
      pass

  assert syncer._is_compatible(MockClass2, ["arg1"]) is True

  class MockClass3:
    pass

  assert syncer._is_compatible(MockClass3, ["arg1"]) is True


def test_syncer_names_tuple_and_self():
  from ml_switcheroo.discovery.syncer import FrameworkSyncer

  syncer = FrameworkSyncer()

  # 115: tuple in args
  assert syncer._extract_names([["a", "b"], "c"]) == ["a", "c"]

  # 152: self parameter is popped
  # MockClass2.call has `self` and `arg1`. It was tested but maybe not hit?
  # Actually wait, `is_class_obj` is True for class instances or classes?
  # `isinstance(obj, type)` -> obj is a class!
  # If obj is a class, it does `params = list(sig.parameters.values())`.
  # Let's verify `self` is popped by making it fail if it doesn't pop.

  class MockClassSelf:
    def __call__(self, arg_x):
      pass

  assert syncer._is_compatible(MockClassSelf, ["arg_x"]) is True

  # 103: no variants dict
  # the sync test `Weird_Op` had `"variants": {}`.
  spec = {"Weird_Op2": {"std_args": ["arg1"]}}

  class MockTarget:
    __name__ = "mock_fw_no_var"

    def Weird_Op2(self, arg1):
      pass

  import sys

  sys.modules["mock_fw_no_var"] = MockTarget()

  with (
    __import__("unittest.mock").mock.patch("ml_switcheroo.discovery.syncer.get_adapter", return_value=None),
    __import__("unittest.mock").mock.patch("importlib.import_module", return_value=sys.modules["mock_fw_no_var"]),
  ):
    syncer.sync(spec, "mock_fw_no_var")

  assert "mock_fw_no_var" in spec["Weird_Op2"]["variants"]
