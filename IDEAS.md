Ideas
=====

## Advanced Features

### Dynamic Shape Tracing
Integrate with `torch.fx` or `jax.make_jaxpr` to extract graph information for cases where static AST analysis fails (e.g. data-dependent control flow that resolves to static graph). 

### Remote Snapshot Registry
Currently, `snapshots/` are stored in the repo. Move to a remote registry (like `conda-forge` or a dedicated repo) allowing users to download mappings for specific framework versions (e.g. `ml_switcheroo sync --remote torch==1.13.0`).

### Type Inference Engine
Enhance the `PurityScanner` to perform basic type inference, allowing for more robust "Method vs Property" resolution without relying strictly on API naming conventions.
