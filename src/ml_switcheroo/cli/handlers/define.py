"""
Handler for the 'define' command.

This module orchestrates the "Code Injection" workflow.

It imports YAML, validates schema, and uses:
1. JSON Injector to inject specifications into `semantics/*.json` (The Hub).
2. JSON Injector to update framework definition files in `snapshots/` (The Spokes).
3. Code Generators to scaffold plugins and tests.
"""

from pathlib import Path
from typing import Optional

try:
  import yaml
except ImportError:  # pragma: no cover
  yaml = None  # pragma: no cover

from ml_switcheroo.core.dsl import OperationDef
from ml_switcheroo.core.discovery import SimulatedReflection
from ml_switcheroo.tools.injector_spec import StandardsInjector
from ml_switcheroo.tools.injector_fw.core import FrameworkInjector
from ml_switcheroo.tools.injector_plugin import PluginGenerator
from ml_switcheroo.utils.console import log_success, log_error, log_info, log_warning
from ml_switcheroo.frameworks import get_adapter
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.generated_tests.generator import TestCaseGenerator
import ml_switcheroo.plugins


def handle_define(yaml_file: Path, dry_run: bool = False, no_test_gen: bool = False) -> int:
  """
  Main entry point for defining new operations.

  Args:
      yaml_file: Path to the input YAML definition.
      dry_run: If True, simulate changes without writing to disk.
      no_test_gen: If True, skip generation of test files.

  Returns:
      int: Exit Code (0 for success, 1 for failure).
  """
  if yaml is None:
    log_error("PyYAML is not installed.")  # pragma: no cover
    return 1  # pragma: no cover

  if not yaml_file.exists() and not str(yaml_file) == "-":
    log_error(f"File not found: {yaml_file}")  # pragma: no cover
    return 1  # pragma: no cover

  try:
    if str(yaml_file) == "-":
      import sys  # pragma: no cover

      content = sys.stdin.read()  # pragma: no cover
    else:
      content = yaml_file.read_text(encoding="utf-8")

    ops = []
    for doc in yaml.safe_load_all(content):
      if not doc:
        continue  # pragma: no cover

      if isinstance(doc, list):
        ops.extend([OperationDef(**d) for d in doc])  # pragma: no cover
      else:
        ops.append(OperationDef(**doc))

  except Exception as e:  # pragma: no cover
    log_error(f"Failed to validate ODL YAML: {e}")  # pragma: no cover
    return 1  # pragma: no cover

  semantics_mgr = SemanticsManager()

  for op_def in ops:
    log_info(f"Processing '{op_def.operation}'...")

    # 1. Auto-Inference
    _resolve_inferred_apis(op_def)

    # 2. Hub Injection (JSON)
    if not _inject_hub(op_def, dry_run=dry_run):
      return 1  # pragma: no cover

    # 3. Spoke Injection (JSON)
    _inject_spokes(op_def, dry_run=dry_run)

    # 4. Plugin Scaffolding
    _scaffold_plugins(op_def, dry_run=dry_run)

    # 5. Test Generation
    if not no_test_gen:
      _generate_test_file(op_def, semantics_mgr, dry_run=dry_run)

  return 0


def _resolve_inferred_apis(op_def: OperationDef) -> None:
  """
  Updates variant APIs in `op_def` if they are set to "infer".

  Args:
      op_def: The definition object to mutate.
  """
  for fw_key, variant in op_def.variants.items():
    if variant and variant.api and variant.api.lower() == "infer":
      reflector = SimulatedReflection(fw_key)
      discovered = reflector.discover(op_def.operation)
      if discovered:
        variant.api = discovered
        log_success(f"  Inferred {fw_key}: {discovered}")
      else:
        log_warning(f"  Could not infer API for {op_def.operation} in {fw_key}")  # pragma: no cover


def _inject_hub(op_def: OperationDef, dry_run: bool = False) -> bool:
  """
  Injects the definition into the Semantic Knowledge Base (JSON).

  Args:
      op_def: The operation definition.
      dry_run: Simulation mode flag.

  Returns:
      bool: True on success.
  """
  try:
    injector = StandardsInjector(op_def)
    return injector.inject(dry_run=dry_run)
  except Exception as e:  # pragma: no cover
    log_error(f"Hub Injection failed for operation '{op_def.operation}': {e}")  # pragma: no cover
    return False  # pragma: no cover


def _inject_spokes(op_def: OperationDef, dry_run: bool = False) -> None:
  """
  Injects definitions into framework JSON files.

  Args:
      op_def: The operation definition.
      dry_run: Simulation mode flag.
  """
  for fw_key, variant in op_def.variants.items():
    if not variant or (variant.api and variant.api.lower() == "infer"):
      continue  # pragma: no cover

    # Check if framework is valid
    adapter = get_adapter(fw_key)
    if not adapter:
      log_warning(f"  Skipping {fw_key}: No adapter registered.")  # pragma: no cover
      continue  # pragma: no cover

    try:
      injector = FrameworkInjector(target_fw=fw_key, op_name=op_def.operation, variant=variant)
      injector.inject(dry_run=dry_run)

    except Exception as e:  # pragma: no cover
      log_warning(f"  Failed to update {fw_key}: {e}")  # pragma: no cover


def _scaffold_plugins(op_def: OperationDef, dry_run: bool = False) -> None:
  """
  Scaffolds new plugin files defined in the ODL.

  Args:
      op_def: The operation definition.
      dry_run: Simulation mode flag.
  """
  if not op_def.scaffold_plugins:
    return
  try:
    # Determine plugins directory
    if ml_switcheroo.plugins.__file__:
      plugins_pkg_dir = Path(ml_switcheroo.plugins.__file__).parent
    else:
      # Fallback relative calculation
      plugins_pkg_dir = Path(__file__).resolve().parents[2] / "plugins"  # pragma: no cover

    generator = PluginGenerator(plugins_pkg_dir)
    for plug_def in op_def.scaffold_plugins:
      if dry_run:
        log_info(f"  [Dry Run] Would generate plugin file: {plug_def.name}")  # pragma: no cover
      else:
        generator.generate(plug_def)
        log_success(f"  Generated Plugin: {plug_def.name}")
  except Exception as e:  # pragma: no cover
    log_error(f"Plugin scaffolding error: {e}")  # pragma: no cover


def _generate_test_file(op_def: OperationDef, mgr: SemanticsManager, dry_run: bool = False) -> None:
  """
  Generates a physical test file for verification.

  Args:
      op_def: The operation definition.
      mgr: The Semantics Manager.
      dry_run: Simulation mode flag.
  """
  if dry_run:
    log_info(f"  [Dry Run] Would generate test file for {op_def.operation}")  # pragma: no cover
    return  # pragma: no cover

  try:
    root_dir = Path.cwd()
    test_dir = root_dir / "tests" / "generated"
    safe_name = op_def.operation.lower().replace(" ", "_").replace("-", "_")
    test_file = test_dir / f"test_odl_{safe_name}.py"

    gen = TestCaseGenerator(semantics_mgr=mgr)
    gen.generate({op_def.operation: op_def.model_dump(exclude_unset=True)}, test_file)
    # We don't log success here as TestCaseGenerator manages file IO quietly usually,
    # but it's good feedback
  except Exception as e:  # pragma: no cover
    log_warning(f"Test generation failed: {e}")  # pragma: no cover
