"""Module tests."""

import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo_ir.schema.ghost import SemanticTier


class TestSemanticsManagerExtra(unittest.TestCase):
  """Test class."""

  @patch("ml_switcheroo.semantics.manager.KnowledgeBaseLoader")
  @patch("ml_switcheroo.semantics.manager.RegistryLoader")
  def setUp(self, MockRegistryLoader, MockKnowledgeBaseLoader):
    """Test method."""
    self.manager = SemanticsManager()
    self.manager.data = {}
    self.manager.framework_configs = {}
    self.manager.test_templates = {}
    self.manager._providers = {}
    self.manager._source_registry = {}
    self.manager._reverse_index = {}
    self.manager._key_origins = {}
    self.manager._validation_status = {}
    self.manager._known_rng_methods = set()
    self.manager.patterns = []

  def test_build_index_empty_impl(self):
    """Test method."""
    self.manager.data = {"abs1": {"variants": {"fw1": None}}}
    self.manager._build_index()
    self.assertEqual(self.manager._reverse_index, {})

  def test_build_index_priority(self):
    """Test method."""
    self.manager.framework_configs = {"fw1": {"alias": {"module": "mod1", "name": "api_mod"}}}
    self.manager.data = {
      "concat": {"variants": {"fw1": {"api": "api_mod.concat"}}},
      "cat": {"variants": {"fw1": {"api": "api_mod.concat"}}},
      "Append": {"variants": {"fw1": {"api": "api_mod.concat"}}},
      "Mean": {"variants": {"fw1": {"api": "mean"}}},
      "Average": {"variants": {"fw1": {"api": "mean"}}},
      "mean": {"variants": {"fw1": {"api": "mean"}}},
      "relu": {"variants": {"fw1": {"api": "relu"}}},
      "ReLU": {"variants": {"fw1": {"api": "relu"}}},
      "MultiHeadAttention": {"variants": {"fw1": {"api": "mha"}}},
      "AttentionLayer": {"variants": {"fw1": {"api": "mha"}}},
      "Dropout": {"variants": {"fw1": {"api": "dropout"}}},
      "Dropout_": {"variants": {"fw1": {"api": "dropout"}}},
      "arr_op": {"variants": {"fw1": {"api": "arr"}}},
      "nn_op": {"variants": {"fw1": {"api": "nn"}}},
      "ext_op": {"variants": {"fw1": {"api": "ext"}}},
    }
    self.manager._key_origins = {
      "arr_op": SemanticTier.ARRAY_API.value,
      "nn_op": SemanticTier.NEURAL.value,
      "ext_op": SemanticTier.EXTRAS.value,
    }
    self.manager._build_index()
    self.assertIn("api_mod.concat", self.manager._reverse_index)
    self.assertEqual(self.manager._reverse_index["api_mod.concat"][0], "cat")
    self.assertIn("mod1.concat", self.manager._reverse_index)

  def test_get_import_map(self):
    """Test method."""
    self.manager._providers = {
      "target_fw": {SemanticTier.ARRAY_API: {"root": "target_root", "sub": "sub1", "alias": "tgt_alias"}},
      "parent_fw": {SemanticTier.NEURAL: {"root": "parent_root", "sub": "sub2", "alias": "prt_alias"}},
    }
    self.manager.framework_configs = {"target_fw": {"extends": "parent_fw"}}
    self.manager._source_registry = {
      "src/path1": ("some_fw", SemanticTier.ARRAY_API),
      "src/path2": ("some_fw", SemanticTier.NEURAL),
      "src/path3": ("some_fw", SemanticTier.EXTRAS),
    }

    result = self.manager.get_import_map("target_fw")
    self.assertEqual(result.get("src/path1"), ("target_root", "sub1", "tgt_alias"))
    self.assertEqual(result.get("src/path2"), ("parent_root", "sub2", "prt_alias"))
    self.assertNotIn("src/path3", result)

  @patch("ml_switcheroo.semantics.manager.get_adapter")
  def test_resolve_inheritance(self, mock_get_adapter):
    """Test method."""
    self.manager.framework_configs = {"fw1": {"extends": "parent1"}}
    self.assertEqual(self.manager._resolve_inheritance("fw1"), "parent1")

    mock_adapter = MagicMock()
    mock_adapter.inherits_from = "parent2"
    mock_get_adapter.return_value = mock_adapter
    self.assertEqual(self.manager._resolve_inheritance("fw2"), "parent2")

    mock_adapter2 = MagicMock()
    del mock_adapter2.inherits_from
    mock_get_adapter.return_value = mock_adapter2
    self.assertIsNone(self.manager._resolve_inheritance("fw3"))

    mock_get_adapter.return_value = None
    self.assertIsNone(self.manager._resolve_inheritance("fw4"))

  def test_resolve_variant(self):
    """Test method."""
    self.manager.data = {
      "op1": {"variants": {"target_fw": {"impl": 1}, "parent_fw": {"impl": 2}, "grandparent_fw": {"impl": 3}}},
      "op2": {"variants": {"parent_fw": {"impl": 2}}},
    }
    self.manager.framework_configs = {
      "target_fw": {"extends": "parent_fw"},
      "parent_fw": {"extends": "grandparent_fw"},
      "grandparent_fw": {"extends": "greatgrandparent_fw"},
      "greatgrandparent_fw": {"extends": "another_fw"},
      "another_fw": {"extends": "fw6"},
      "fw6": {"extends": "fw7"},
      "fw_limit_0": {"extends": "fw_limit_1"},
      "fw_limit_1": {"extends": "fw_limit_2"},
      "fw_limit_2": {"extends": "fw_limit_3"},
      "fw_limit_3": {"extends": "fw_limit_4"},
      "fw_limit_4": {"extends": "fw_limit_5"},
      "fw_limit_5": {"extends": "fw_limit_6"},
    }

    self.assertEqual(self.manager.resolve_variant("op1", "target_fw"), {"impl": 1})
    self.assertEqual(self.manager.resolve_variant("op2", "target_fw"), {"impl": 2})
    self.assertIsNone(self.manager.resolve_variant("op3", "target_fw"))
    self.assertIsNone(self.manager.resolve_variant("op2", "another_fw"))
    self.assertIsNone(self.manager.resolve_variant("op2", "unknown_fw"))
    self.assertIsNone(self.manager.resolve_variant("op2", "fw_limit_0"))

  def test_is_verified(self):
    """Test method."""
    self.manager._validation_status = {"abs1": False}
    self.assertFalse(self.manager.is_verified("abs1"))
    self.assertTrue(self.manager.is_verified("abs2"))

  def test_get_definition_by_id(self):
    """Test method."""
    self.manager.data = {"abs1": {"def": 1}}
    self.assertEqual(self.manager.get_definition_by_id("abs1"), {"def": 1})
    self.assertIsNone(self.manager.get_definition_by_id("abs2"))

  def test_get_definition(self):
    """Test method."""
    self.manager._reverse_index = {"api1": ("abs1", {"def": 1})}
    self.manager.data = {"abs2": {"def": 2}}
    self.assertEqual(self.manager.get_definition("api1"), ("abs1", {"def": 1}))
    self.assertEqual(self.manager.get_definition("abs2"), ("abs2", {"def": 2}))
    self.assertIsNone(self.manager.get_definition("unknown"))

  def test_get_known_apis(self):
    """Test method."""
    self.manager.data = {"abs1": {"def": 1}}
    self.assertEqual(self.manager.get_known_apis(), {"abs1": {"def": 1}})

  def test_get_framework_config(self):
    """Test method."""
    self.manager.framework_configs = {"fw1": {"config": 1}}
    self.assertEqual(self.manager.get_framework_config("fw1"), {"config": 1})
    self.assertEqual(self.manager.get_framework_config("fw2"), {})

  def test_get_test_template(self):
    """Test method."""
    self.manager.test_templates = {"fw1": {"template": "a"}}
    self.assertEqual(self.manager.get_test_template("fw1"), {"template": "a"})
    self.assertIsNone(self.manager.get_test_template("fw2"))

  def test_get_framework_aliases(self):
    """Test method."""
    self.manager.framework_configs = {
      "fw1": {"alias": {"module": "mod1", "name": "name1"}},
      "fw2": {"alias": "invalid_type"},
      "fw3": {"alias": {"module": "mod3"}},
      "fw4": {},
    }
    expected = {"fw1": ("mod1", "name1")}
    self.assertEqual(self.manager.get_framework_aliases(), expected)

  def test_get_all_rng_methods(self):
    """Test method."""
    self.manager._known_rng_methods = {"rng1"}
    self.assertEqual(self.manager.get_all_rng_methods(), {"rng1"})

  def test_get_patterns(self):
    """Test method."""
    self.manager.patterns = ["pat1"]
    self.assertEqual(self.manager.get_patterns(), ["pat1"])

  @patch("pathlib.Path.exists")
  @patch("builtins.open", new_callable=unittest.mock.mock_open, read_data='{"op1": false}')
  def test_load_validation_report(self, mock_open, mock_exists):
    """Test method."""
    mock_exists.return_value = False
    self.manager.load_validation_report(Path("dummy.json"))
    self.assertEqual(self.manager._validation_status, {})

    mock_exists.return_value = True
    self.manager.load_validation_report(Path("dummy.json"))
    self.assertEqual(self.manager._validation_status, {"op1": False})

    mock_open.side_effect = Exception("error")
    self.manager.load_validation_report(Path("dummy.json"))

  @patch("pathlib.Path.mkdir")
  @patch("builtins.open", new_callable=unittest.mock.mock_open)
  @patch("yaml.dump")
  def test_update_definition(self, mock_yaml_dump, mock_open, mock_mkdir):
    """Test method."""
    self.manager.update_definition("op1", {"description": "desc", "variants": {"fw1": {"api": "op1.api"}}})
    self.assertIn("op1", self.manager.data)
    self.assertEqual(self.manager.data["op1"]["description"], "desc")
    self.assertEqual(self.manager._reverse_index["op1.api"][0], "op1")
    mock_open.assert_called()
    mock_yaml_dump.assert_called()

    mock_open.side_effect = Exception("error")
    self.manager.update_definition("op2", {})
    self.assertIn("op2", self.manager.data)

    self.manager.update_definition("op3", {"variants": []})
    self.assertNotIn("op3", self.manager.data)
