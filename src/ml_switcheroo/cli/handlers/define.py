"""
Handler for the 'define' command.

This module orchestrates the "Code Injection" workflow.

It imports YAML, validates schema, and uses LibCST injectors to modify source files.
Robust checks added for syntax errors during injection.
"""

import difflib
import fileinput
import inspect
from pathlib import Path
from typing import Optional, Any, List

import libcst as cst

try:
  import yaml
except ImportError:
  yaml = None

from ml_switcheroo.core.dsl import OperationDef
from ml_switcheroo.core.discovery import SimulatedReflection
from ml_switcheroo.tools.injector_spec import StandardsInjector
from ml_switcheroo.tools.injector_fw import FrameworkInjector
from ml_switcheroo.tools.injector_plugin import PluginGenerator
from ml_switcheroo.utils.console import log_success, log_error, log_info, log_warning
from ml_switcheroo.frameworks import get_adapter
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.generated_tests.generator import TestGenerator
import ml_switcheroo.semantics.standards_internal as internal_standards_module
import ml_switcheroo.plugins


def handle_define(yaml_file: Path, dry_run: bool = False) -> int:
  if yaml is None:
    log_error("PyYAML is not installed.")
    return 1

  if not yaml_file.exists() and not yaml_file.name == "-":
    log_error(f"File not found: {yaml_file}")
    return 1

  try:
    with fileinput.FileInput(str(yaml_file), mode="r", encoding="utf-8") as content:
      data = yaml.safe_load("".join(content))

    if isinstance(data, list):
      ops = [OperationDef(**d) for d in data]
    else:
      ops = [OperationDef(**data)]

  except Exception as e:
    log_error(f"Failed to valid ODL YAML: {e}")
    return 1

  semantics_mgr = SemanticsManager()

  for op_def in ops:
    # 1. Auto-Inference
    _resolve_inferred_apis(op_def)

    # 2. Hub Injection
    if not _inject_hub(op_def, dry_run=dry_run):
      # If hub fails (syntax error), stop
      return 1

    # 3. Spoke Injection
    _inject_spokes(op_def, dry_run=dry_run)

    # 4. Plugin Scaffolding
    _scaffold_plugins(op_def, dry_run=dry_run)

    # 5. Test Generation
    _generate_test_file(op_def, semantics_mgr, dry_run=dry_run)

    if not dry_run:
      log_success(f"Operation '{op_def.operation}' defined & tested.")

  return 0


def _resolve_inferred_apis(op_def: OperationDef) -> None:
  for fw_key, variant in op_def.variants.items():
    if variant and variant.api and variant.api.lower() == "infer":
      reflector = SimulatedReflection(fw_key)
      discovered = reflector.discover(op_def.operation)
      if discovered:
        variant.api = discovered
        log_success(f"  Inferred {fw_key}: {discovered}")
      else:
        log_warning(f"  Could not infer API for {op_def.operation} in {fw_key}")


def _inject_hub(op_def: OperationDef, dry_run: bool = False) -> bool:
  try:
    spec_file = Path(inspect.getfile(internal_standards_module))
    source_code = spec_file.read_text("utf-8")
    tree = cst.parse_module(source_code)

    transformer = StandardsInjector(op_def)
    new_tree = tree.visit(transformer)

    if not transformer.found:
      log_warning("INTERNAL_OPS dictionary not found.")
      return False

    generated_code = new_tree.code

    # Syntax check before writing
    try:
      compile(generated_code, "<string>", "exec")
    except SyntaxError as e:
      log_error(f"Generated Invalid Python (Hub): {e}")
      return False

    if generated_code != source_code:
      if dry_run:
        _print_diff(source_code, generated_code, str(spec_file))
      else:
        spec_file.write_text(generated_code, "utf-8")
        log_success(f"Updated Hub: {spec_file.name}")
    else:
      log_info(f"Hub unchanged (Key exists): {spec_file.name}")
    return True
  except Exception as e:
    log_error(f"Hub Injection failed: {e}")
    return False


def _inject_spokes(op_def: OperationDef, dry_run: bool = False) -> None:
  for fw_key, variant in op_def.variants.items():
    if not variant or (variant.api and variant.api.lower() == "infer"):
      continue

    adapter = get_adapter(fw_key)
    if not adapter:
      continue

    try:
      adapter_cls = type(adapter)
      adapter_file = Path(inspect.getfile(adapter_cls))
      source_code = adapter_file.read_text("utf-8")
      tree = cst.parse_module(source_code)

      injector = FrameworkInjector(target_fw=fw_key, op_name=op_def.operation, variant=variant)
      new_tree = tree.visit(injector)

      generated_code = new_tree.code
      try:
        compile(generated_code, "<string>", "exec")
      except SyntaxError as e:
        log_error(f"Generated Invalid Python (Spoke {fw_key}): {e}")
        continue

      if not injector.found:
        log_warning(f"Could not inject into {fw_key} (definitions property missing?)")
        continue

      if generated_code != source_code:
        if dry_run:
          _print_diff(source_code, generated_code, str(adapter_file))
        else:
          adapter_file.write_text(generated_code, "utf-8")
          log_success(f"Updated Spoke ({fw_key})")
      else:
        log_info(f"Spoke unchanged ({fw_key})")
    except Exception as e:
      log_warning(f"Failed to update {fw_key}: {e}")


def _scaffold_plugins(op_def: OperationDef, dry_run: bool = False) -> None:
  if not op_def.scaffold_plugins:
    return
  try:
    if ml_switcheroo.plugins.__file__:
      plugins_pkg_dir = Path(ml_switcheroo.plugins.__file__).parent
    else:
      plugins_pkg_dir = Path(__file__).resolve().parents[2] / "plugins"

    generator = PluginGenerator(plugins_pkg_dir)
    for plug_def in op_def.scaffold_plugins:
      if dry_run:
        log_info(f"[Dry Run] Generate plugin: {plug_def.name}")
      else:
        generator.generate(plug_def)
        log_success(f"Generated Plugin: {plug_def.name}")
  except Exception as e:
    log_error(f"Plugin scaffolding error: {e}")


def _generate_test_file(op_def: OperationDef, mgr: SemanticsManager, dry_run: bool = False) -> None:
  if dry_run:
    return

  try:
    root_dir = Path.cwd()
    test_dir = root_dir / "tests" / "generated"
    safe_name = op_def.operation.lower().replace(" ", "_").replace("-", "_")
    test_file = test_dir / f"test_odl_{safe_name}.py"

    gen = TestGenerator(semantics_mgr=mgr)
    gen.generate({op_def.operation: op_def.model_dump(exclude_unset=True)}, test_file)
  except Exception as e:
    log_warning(f"Test match generation failed: {e}")


def _print_diff(old_code: str, new_code: str, filename: str) -> None:
  diff = difflib.unified_diff(
    old_code.splitlines(keepends=True),
    new_code.splitlines(keepends=True),
    fromfile=filename,
    tofile=f"{filename} (modified)",
  )
  print("".join(diff))
