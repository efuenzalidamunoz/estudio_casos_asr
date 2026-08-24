#!/bin/bash

set -euo pipefail


QUANTIZE="whisper.cpp/build/bin/whisper-quantize"
MODEL_DIR="whisper.cpp/models"
INPUT_MODEL="$MODEL_DIR/ggml-base.bin"
OUTPUT_DIR="scripts_out/R1/a"
mkdir -p "$OUTPUT_DIR"

if [ ! -x "$QUANTIZE" ]; then
    echo "ERROR: no se encontró whisper-quantize:"
    echo "$QUANTIZE"
    exit 1
fi

if [ ! -f "$INPUT_MODEL" ]; then
    echo "ERROR: no se encontró el modelo base:"
    echo "$INPUT_MODEL"
    exit 1
fi

FORMATOS=(
    "q8_0"
    "q5_1"
    "q4_0"
)


for FORMATO in "${FORMATOS[@]}"; do

    OUTPUT_MODEL="$MODEL_DIR/ggml-base-${FORMATO}.bin"
    OUTPUT_LOG="$OUTPUT_DIR/cuantizacion_${FORMATO}.txt"

    echo "Cuantizando $FORMATO..."

    "$QUANTIZE" \
        "$INPUT_MODEL" \
        "$OUTPUT_MODEL" \
        "$FORMATO" \
        > "$OUTPUT_LOG" 2>&1

done

echo "Cuantizacion terminada."