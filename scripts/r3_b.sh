#!/bin/bash

set -euo pipefail

LLM_MODEL="gemma3:1b-it-q4_K_M"
TARGET_LANG="English"

WHISPER_BASE="whisper.cpp/models/ggml-base.bin"
WHISPER_Q4="whisper.cpp/models/ggml-base-q4_0.bin"

OUT_DIR="scripts_out/R3/b/"

REF_OUT="$OUT_DIR/referencia_llm.txt"
BASE_OUT="$OUT_DIR/asr_alta_precision_llm.txt"
Q4_OUT="$OUT_DIR/asr_cuantizado_llm.txt"

mkdir -p "$OUT_DIR"

if ! command -v julia >/dev/null 2>&1; then
    echo "ERROR: Julia no está disponible en PATH."
    exit 1
fi

if [ ! -f "translate_reference.jl" ]; then
    echo "ERROR: falta translate_reference.jl"
    exit 1
fi

if [ ! -f "asr_pipeline.jl" ]; then
    echo "ERROR: falta asr_pipeline.jl"
    exit 1
fi

if [ ! -f "translate_pipeline.jl" ]; then
    echo "ERROR: falta translate_pipeline.jl"
    exit 1
fi

if [ ! -f "$WHISPER_BASE" ]; then
    echo "ERROR: falta $WHISPER_BASE"
    exit 1
fi

if [ ! -f "$WHISPER_Q4" ]; then
    echo "ERROR: falta $WHISPER_Q4"
    exit 1
fi
echo "1. Referencia -> LLM"

julia translate_reference.jl \
    "$LLM_MODEL" \
    "$TARGET_LANG" \
    > "$REF_OUT" 2>&1

echo "Guardado en:"
echo "  $REF_OUT"

echo "2. ASR alta precisión -> LLM"

# Primero genera las transcripciones ASR con Base
julia asr_pipeline.jl \
    "$WHISPER_BASE" \
    > "$BASE_OUT" 2>&1

# Luego traduce esas salidas con el mismo LLM
julia translate_pipeline.jl \
    "$LLM_MODEL" \
    "$TARGET_LANG" \
    >> "$BASE_OUT" 2>&1

echo "Guardado en:"
echo "  $BASE_OUT"

echo "3. ASR cuantizado -> LLM"

# Genera transcripciones con Q4_0
julia asr_pipeline.jl \
    "$WHISPER_Q4" \
    > "$Q4_OUT" 2>&1

# Traduce esas transcripciones
julia translate_pipeline.jl \
    "$LLM_MODEL" \
    "$TARGET_LANG" \
    >> "$Q4_OUT" 2>&1

echo "Guardado en:"
echo "  $Q4_OUT"

echo "Listo"
echo
