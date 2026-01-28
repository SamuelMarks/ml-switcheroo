Definitions
===========

Hint: you can generate the .yml files like so:

```sh
ml_switcheroo suggest 'mlx.core.*' \
  --out-dir tmp/mlx_ops \
  --batch-size 30
```

Then run:

```sh
n='001'
printf '%s\n---\n%s\n---\n%s\n' \
  "$(cat -- 'tmp/mlx_ops/suggest_mlx_core_'"$n"'.md')" \
  "$(code2prompt -e '*.lock,*.json,*.yml,*.yaml,test_*,sphinx*' --token-map .)" \
  'I expect YAML for 30 operations for: PyTorch, MLX, Keras, TensorFlow (if different from Keras), Flax NNX (if different from JAX), and Pax (if different from JAX).' \
```

…and put that prompt into your favourite LLM. Paste here the output, e.g.,

- `src/ml_switcheroo/frameworks/definitions/suggest_mlx_core_001.yml`

Check if it worked by hand, and initially at least a basic count equality:

```sh
$ grep -F 'Name:' 'tmp/mlx_ops/suggest_mlx_core_'"$n"'.md' | sort -u | wc -l
30
$ grep -F 'operation:' 'src/ml_switcheroo/frameworks/definitions/suggest_mlx_core_'"${n?}"'.yml' | wc -l
30
```

If not, redo the whole prompt—context and all—or just tell the LLM:
> "I expected x you gave me y"

Rinse and repeat until all prompt Markdown files are read in and all YAML files are produced.

For a completely scripted approach to above see [`scripts/suggest_gen_llm_loop.sh`](/scripts/suggest_gen_llm_loop.sh)

To analyse what you have, use commands like:

```sh
# Find total number of unique operations mapped out of `mlx.core`
$ yq -r '.operation' $(fd -F 'suggest_mlx_core' -e yml) | sort -u | wc -l                
     252
# Find number of `mlx.core` operations that map to NumPy
$ yq -r 'select(.variants.numpy.api) | .operation' $(fd -F 'suggest_mlx_core' -e yml) | sort -u | wc -l
     164
# Find number of `mlx.core` operations that map to each ${ml framework}
$ for variant in jax flax_nnx paxml keras mlx torch tensorflow numpy; do \
  printf '%-11s%3d\n' "$variant" "$(yq -r 'select(.variants.'"$variant"'.api) | .operation' $(fd -F 'suggest_mlx' -e yml) | sort -u | wc -l)" ; done
jax        226
flax_nnx   160
paxml       90
keras      184
mlx        326
torch      271
tensorflow 179
numpy      161
```
