ml-switcheroo ideas
===================

## Website

Hosted through GitHub Actions or ReadTheDocs

### API docs

Using sphinx or mkdocs

### Tutorials

From markdown showing how to use the tech and how to extend it

### Rendered markdown

Of the files:

- "ARCHITECTURE.md"
- "EXTENDING.md"
- "MAINTENANCE.md"

### Interactive Python

WebAssembly (WASM) with this project preinstalled.

Frontend can look something like this, mayhaps written in Angular:
```
_______________________________________
| From      [button <->]    TO        |
| dropdown       |          dropdown  |
|____________    |          __________|
              EXAMPLES
              dropdown
|RUN BUTTON|
_______________________________________
| editable code here                  |
|                                     |
|                                     |
---------------------------------------
            [button <->]
_______________________________________
| output code here                    |
|                                     |
|                                     |
---------------------------------------

---------------------
| Console log       |
---------------------
---------------------
| HTML table report |
---------------------
```

## Advanced Features

### Reverse Class Warping
Currently `torch.nn.Module` -> `flax.nnx.Module` works. 
Reverse (`NNX` -> `Torch`) is complex due to `__init__` vs `__call__` state management. 
Idea: Use the `SemanticsManager` state traits to reverse-engineer `self.param` assignments.

### Dynamic Shape Tracing
Integrate with `torch.fx` or `jax.make_jaxpr` to extract graph information for cases where static AST analysis fails (e.g. data-dependent control flow that resolves to static graph).

### Documentation Generator
Generate "Migration Guides" automatically by diffing the semantics JSONs.
Example output: "In JAX, `torch.foo` is equivalent to `jax.bar`, but argument `dim` becomes `axis`. Warning: `keepdims` default differs."
