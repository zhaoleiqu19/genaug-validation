#!/usr/bin/env bash
# Full CDFSOD baseline sweep: 6 domains x {1,5,10} shots x N seeds.
# Usage: run_matrix.sh <gpu_id> [seeds="42 43 44"]
set -euo pipefail

GPU="${1:?usage: run_matrix.sh <gpu_id> [seeds]}"
SEEDS="${2:-42 43 44}"
PORT_BASE=9999

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOMAINS="ArTaxOr clipart1k DIOR FISH NEU-DET UODD"
SHOTS="1 5 10"

i=0
for domain in $DOMAINS; do
  for shot in $SHOTS; do
    for seed in $SEEDS; do
      port=$((PORT_BASE + i))
      i=$((i + 1))
      "${SCRIPT_DIR}/run_one.sh" "${domain}" "${shot}" "${seed}" "${GPU}" "${port}"
    done
  done
done
