#!/bin/sh

set -e

if ! command -v code2prompt >/dev/null 2>&1 ; then
  printf 'Install code2prompt then try again\n'
  exit 2
elif ! command -v pbcopy >/dev/null 2>&1 ; then
  # shellcheck disable=SC2016
  printf 'Set `pbcopy` `alias` then try again\n'
  exit 2
fi

exec_on_num() {
  n="$(printf "%03d" "$1")"
  f="${BASE_DIR}"'/'"${BASE_FILE}${n}""${BASE_EXT}"
  if [ ! -f "$f" ]; then
    2>&1 printf 'File nonexistent %s\n\n' "$f"
  else
    printf '%s\n---\n%s\n---\n%s\n' \
      "$(cat -- "$f")" \
      "$(code2prompt -e '*.lock,*.json,*.yml,*.yaml,test_*,sphinx*' --token-map .)" \
      'I expect YAML for 30 operations for: PyTorch, MLX, Keras, TensorFlow (if different from Keras), Flax NNX (if different from JAX), and Pax (if different from JAX).' \
      | pbcopy

    out='src/ml_switcheroo/frameworks/definitions/'"${BASE_FILE}${n}"'.yml'
    printf 'f=%s\n' "$f"
    printf 'out=%s\n' "$out"
    printf 'Everything is in your clipboard, provide it to your LLM. Then provide the file, e.g.:\n%s%s%s\n\n' \
           "pbpaste > '" \
           "$out" \
           "'"
    printf 'Press ENTER once you'"'"'ve pasted for a sanity check\n\t'
    read -r _

    printf 'Requested: %d\nGenerated: %d\n\n' \
            "$(grep -F 'Name:' -- "$f" | sort -u | wc -l)" \
            "$(grep -Fc 'operation:' -- "$out")"
  fi
}

suggest_gen_llm_loop() {
    BASE_DIR="${BASE_DIR:-tmp/keras}"
    BASE_FILE="${BASE_FILE:-suggest_keras_layers_}"
    BASE_EXT='.md'
    n=0
    while true; do
        # shellcheck disable=SC2016
        printf 'Enter r to redo sanity check, n for next number, or a digit
next NUMBER for "%s/%s${NUMBER}%s":\n\t' "$BASE_DIR" "$BASE_FILE" "$BASE_EXT"
        read -r user_input
        if [ -n "$user_input" ]; then
          case "$user_input" in
              'r')
                printf 'Requested: %d\nGenerated: %d\n\n' \
                        "$(grep -F 'Name:' -- "$f" | sort -u | wc -l)" \
                        "$(grep -Fc 'operation:' -- "$out")"
                ;;
              'n')
                # shellcheck disable=SC2003
                n="$(expr "$n" + 1)"
                exec_on_num "$n"
                ;;
              'q'*|'x'*|'e'*)
                break
                ;;
              ''|*[!0-9]*)  echo "Invalid input. Please enter a valid number." ;;
              *)
                n="$user_input"
                exec_on_num "$n"
                ;;
          esac
        else
            # Input is not a valid number
            echo "Invalid input. Please enter a valid number."
        fi
    done
}

suggest_gen_llm_loop
