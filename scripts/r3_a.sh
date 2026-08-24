#!/bin/bash

set -euo pipefail

mkdir -p scripts_out/R3/a

MODELOS=(
    "gemma3:1b-it-fp16"
    "gemma3:1b-it-q8_0"
    "gemma3:1b-it-q4_K_M"
)

for MODELO in "${MODELOS[@]}"; do

    NOMBRE=$(echo "$MODELO" | tr ':' '_' | tr '/' '_')

    julia translate_pipeline.jl "$MODELO" English \
        > "scripts_out/R3/a/${NOMBRE}.txt" 2>&1

done