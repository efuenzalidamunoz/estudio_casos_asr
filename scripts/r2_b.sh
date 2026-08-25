#!/bin/bash

mkdir -p scripts_out/R2/b/

LLM_MODEL="gemma3:1b-it-q4_K_M"

modelos=(
  "ggml-base.bin"
  "ggml-tiny.bin"
  "ggml-base-q8_0.bin"
  "ggml-base-q5_1.bin"
  "ggml-base-q4_0.bin"
)

for modelo in "${modelos[@]}"; do
  echo "Procesando con $modelo..."

  OUT_FILE="scripts_out/R2/b/salida_${modelo}.txt"

  # Ejecutar ASR
  julia asr_pipeline_igpu.jl "whisper.cpp/models/${modelo}" \
    > "$OUT_FILE" 2>&1

  # Ejecutar traduccion
  julia translate_pipeline.jl "$LLM_MODEL" English \
    >> "$OUT_FILE" 2>&1

  echo "Listo"
done