#!/usr/bin/env bash
# Lanza varias semillas de uno o más experimentos, en serie.
# Es reanudable: omite los runs que ya tienen test.json y reinicia los incompletos,
# así que tras una interrupción basta con volver a lanzar el mismo comando.
#   scripts/run_seeds.sh "0 1 2" configs/cifar10_dot.yaml configs/cifar10_ham_concat.yaml [-- clave=valor ...]
set -euo pipefail
cd "$(dirname "$0")/.."

seeds=$1; shift
configs=()
while [[ $# -gt 0 && $1 != "--" ]]; do configs+=("$1"); shift; done
[[ ${1:-} == "--" ]] && shift
overrides=("$@")

for cfg in "${configs[@]}"; do
  for s in $seeds; do
    echo ">>> $cfg  seed=$s"
    uv run python -m ham.train --skip-done --config "$cfg" --seed "$s" "${overrides[@]}"
  done
done
