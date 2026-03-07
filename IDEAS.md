Ideas
=====

## Advanced Features

### Dynamic Shape Tracing

Integrate with `torch.fx` or `jax.make_jaxpr` to extract graph information for cases where static AST analysis fails (
e.g. data-dependent control flow that resolves to static graph).

### Remote Snapshot Registry

Currently, `snapshots/` are stored in the repo. Move to a remote registry (like `conda-forge` or a dedicated repo)
allowing users to download mappings for specific framework versions (e.g. `ml_switcheroo sync --remote torch==1.13.0`).

### Type Inference Engine

Enhance the `PurityScanner` to perform basic type inference, allowing for more robust "Method vs Property" resolution
without relying strictly on API naming conventions.

### Benchmarking

Compare same architecture, precision, hyperparameters across different datasets and accelerators. Show performance from
each ML framework, in order to confidently—and with evidence—make haughty claims like:
> "ml-switcheroo speeds up ML by up to XXX%"

## Novel source/target frameworks

### Math for theorem provers

To/fro Lean or Coq. Maybe u

## Mobile friendly doc

Show always-on-top selection of target, source framework
Show always-on-top selection of example
Show two tabs, source target
Activate target tab when "Translate" clicked
