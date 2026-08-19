#!/bin/bash

cd .. && cd ..
mkdir -p raw_output
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

  OUT_FILE="scripts_out/salida_${modelo}.txt"
  TEMP_TIME="scripts_out/temp_time.log"

  # Ejecutar ASR (sobrescribe el archivo si existe)
  /usr/bin/time -v -o "$TEMP_TIME" julia asr_pipeline.jl "whisper.cpp/models/${modelo}" > "$OUT_FILE" 2>&1
  KBYTES_ASR=$(grep "Maximum resident set size" "$TEMP_TIME" | tr -dc '0-9')

  # Ejecutar traduccion (se anexa al mismo archivo)
  /usr/bin/time -v -o "$TEMP_TIME" julia translate_pipeline.jl "$LLM_MODEL" English >> "$OUT_FILE" 2>&1
  KBYTES_TRANS=$(grep "Maximum resident set size" "$TEMP_TIME" | tr -dc '0-9')

  # Sumar la memoria de ambas partes y convertir a MB
  if [ -n "$KBYTES_ASR" ] && [ -n "$KBYTES_TRANS" ]; then
    MB_TOTAL=$(awk -v k1="$KBYTES_ASR" -v k2="$KBYTES_TRANS" 'BEGIN {printf "%.2f", (k1 + k2) / 1024}')
  else
    MB_TOTAL="0.00"
    echo "No se pudo leer la memoria."
  fi

  echo "" >> "$OUT_FILE"
  echo "RAM usada: $MB_TOTAL MB" >> "$OUT_FILE"

  rm -f "$TEMP_TIME"
  echo "Listo"
done
