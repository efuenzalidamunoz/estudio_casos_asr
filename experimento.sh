#!/bin/bash

mkdir -p scripts_out/R2/
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

  OUT_FILE="scripts_out/R2/salida_${modelo}.txt"
  TEMP_TIME="scripts_out/R2/temp_time.log"

  # Ejecutar ASR y medir el pico de RAM de whisper
  /usr/bin/time -v -o "$TEMP_TIME" \
    julia asr_pipeline.jl "whisper.cpp/models/${modelo}" \
    > "$OUT_FILE" 2>&1

  # Obtener Maximum Resident Set Size (pico de RAM)
  KBYTES_ASR=$(grep "Maximum resident set size" "$TEMP_TIME" | awk '{print $NF}')

  # Convertir KiB a MiB
  if [ -n "$KBYTES_ASR" ]; then
    MB_ASR=$(awk -v k="$KBYTES_ASR" 'BEGIN {printf "%.2f", k / 1024}')
  else
    MB_ASR="0.00"
    echo "No se pudo leer la memoria del ASR."
  fi

  julia translate_pipeline.jl "$LLM_MODEL" English >> "$OUT_FILE" 2>&1

  echo "" >> "$OUT_FILE"
  echo "RAM pico Whisper: $MB_ASR MB" >> "$OUT_FILE"

  rm -f "$TEMP_TIME"

  echo "Listo"
done
