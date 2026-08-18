#!/bin/bash

mkdir -p raw_output

modelos=(
  "ggml-base.bin"
  "ggml-tiny.bin"
  "ggml-base-q8_0.bin"
  "ggml-base-q5_1.bin"
  "ggml-base-q4_0.bin"
)

for modelo in "${modelos[@]}"; do
  echo "Procesando con $modelo..."

  OUT_FILE="raw_output/salida_${modelo}.txt"
  TEMP_TIME="raw_output/temp_time.log"

  # Ejecutar comando de Julia y desviar el reporte de tiempo al archivo temporal
  /usr/bin/time -v -o "$TEMP_TIME" julia asr_pipeline.jl "whisper.cpp/models/${modelo}" > "$OUT_FILE" 2>&1

  # Extraer solo los numeros de la linea de memoria
  KBYTES=$(grep "Maximum resident set size" "$TEMP_TIME" | tr -dc '0-9')

  # Convertir a MB verificando que KBYTES no este vacio
  if [ -n "$KBYTES" ]; then
    MB=$(awk -v kb="$KBYTES" 'BEGIN {printf "%.2f", kb / 1024}')
  else
    MB="0.00"
    echo "No se pudo leer la memoria."
  fi

  echo "" >> "$OUT_FILE"
  echo "RAM usada: $MB MB" >> "$OUT_FILE"

  # Eliminar el archivo temporal
  rm "$TEMP_TIME"
  echo "Listo"
done
