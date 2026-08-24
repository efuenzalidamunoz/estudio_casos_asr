#!/bin/bash

set -euo pipefail
OUTPUT_DIR="scripts_out/R3/b"
# Mantener SIEMPRE el mismo LLM en las tres condiciones
LLM_MODEL="gemma3:1b-it-q4_K_M"
LANGUAGE="English"
# Modelos Whisper
ASR_ALTA="whisper.cpp/models/ggml-base.bin"
ASR_CUANTIZADO="whisper.cpp/models/ggml-base-q4_0.bin"

mkdir -p "$OUTPUT_DIR"


OUT_REF="$OUTPUT_DIR/01_referencia_llm.txt"

{
    echo "========================================"
    echo "CONDICIÓN 1: REFERENCIA -> LLM"
    echo "========================================"
    echo
    echo "Modelo LLM: $LLM_MODEL"
    echo "Error de entrada (WER): 0"
    echo

    julia translate_reference.jl \
        "$LLM_MODEL" \
        "$LANGUAGE"

} > "$OUT_REF" 2>&1

OUT_BASE="$OUTPUT_DIR/02_asr_alta_precision_llm.txt"

{
    echo "========================================"
    echo "CONDICIÓN 2: ASR ALTA PRECISIÓN -> LLM"
    echo "========================================"
    echo
    echo "Modelo ASR: $ASR_ALTA"
    echo "Modelo LLM: $LLM_MODEL"
    echo

    echo "----------------------------------------"
    echo "SALIDA ASR"
    echo "----------------------------------------"
    echo

    julia asr_pipeline.jl "$ASR_ALTA"

    echo
    echo "----------------------------------------"
    echo "SALIDA TRADUCCIÓN"
    echo "----------------------------------------"
    echo

    julia translate_pipeline.jl \
        "$LLM_MODEL" \
        "$LANGUAGE"

} > "$OUT_BASE" 2>&1


OUT_Q4="$OUTPUT_DIR/03_asr_cuantizado_llm.txt"

{
    echo "========================================"
    echo "CONDICIÓN 3: ASR CUANTIZADO -> LLM"
    echo "========================================"
    echo
    echo "Modelo ASR: $ASR_CUANTIZADO"
    echo "Modelo LLM: $LLM_MODEL"
    echo

    echo "----------------------------------------"
    echo "SALIDA ASR"
    echo "----------------------------------------"
    echo

    julia asr_pipeline.jl "$ASR_CUANTIZADO"

    echo
    echo "----------------------------------------"
    echo "SALIDA TRADUCCIÓN"
    echo "----------------------------------------"
    echo

    julia translate_pipeline.jl \
        "$LLM_MODEL" \
        "$LANGUAGE"

} > "$OUT_Q4" 2>&1