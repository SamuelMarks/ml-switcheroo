Ideas
=====

## Delivered Ideas (Now in Core)

* **Dynamic Shape Tracing & On-Device Execution**: Implemented via `src/ml_switcheroo/core/runtime_tracer.py`, `src/ml_switcheroo/core/ir_bridge.py`, and `RuntimeAnnotationPass`, using the portable `zero-*` and `ml-switcheroo-compiler` ecosystem to trace shapes, resolve lazy constructors, and compute reverse-mode VJP gradients without requiring `torch.fx` or heavy host binaries.
* **External Snapshot Registry**: Implemented via `ml-ecosystem-snapshots` (with SQLite FTS5 symbol indexing and versioned wheel releases).
* **Type Inference Engine**: Implemented via `SymbolTable` (`src/ml_switcheroo/analysis/symbol_table.py`) and `HookContext.resolve_type()`, enabling type-aware `method_property` and plugin transformations.

## Future Ideas & Explorations

### Benchmarking

Compare same architecture, precision, hyperparameters across different datasets and accelerators. Show performance from
each ML framework, in order to confidently—and with evidence—make haughty claims like:
> "ml-switcheroo speeds up ML by up to XXX%"

## Novel source/target frameworks

### Math for theorem provers

Translate neural network definitions and tensor equations to/from formal verification languages such as Lean 4 or Coq to formally prove shape invariants and numerical bounds.

## Mobile friendly doc

Show always-on-top selection of target, source framework
Show always-on-top selection of example
Show two tabs, source target
Activate target tab when "Translate" clicked
