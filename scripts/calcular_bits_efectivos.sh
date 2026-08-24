#!/bin/bash

set -euo pipefail
OUTPUT_DIR="scripts_out/R1/b"
OUTPUT_CSV="$OUTPUT_DIR/bits_efectivos.csv"
mkdir -p "$OUTPUT_DIR"
echo "formato,pesos_por_bloque,bits_pesos,bits_metadata,bits_totales_bloque,bits_efectivos_por_peso" \
    > "$OUTPUT_CSV"

# BASE
# Se reporta como 32 bits/peso
BLOQUE=1
BITS_PESOS=32
BITS_METADATA=0
BITS_TOTAL=$((BITS_PESOS + BITS_METADATA))
BITS_EFECTIVOS=$(awk -v total="$BITS_TOTAL" -v bloque="$BLOQUE" \
    'BEGIN { printf "%.2f", total / bloque }')
echo "base,$BLOQUE,$BITS_PESOS,$BITS_METADATA,$BITS_TOTAL,$BITS_EFECTIVOS" \
    >> "$OUTPUT_CSV"

# TINY
# Se reporta como 16 bits/peso
BLOQUE=1
BITS_PESOS=16
BITS_METADATA=0
BITS_TOTAL=$((BITS_PESOS + BITS_METADATA))
BITS_EFECTIVOS=$(awk -v total="$BITS_TOTAL" -v bloque="$BLOQUE" \
    'BEGIN { printf "%.2f", total / bloque }')
echo "tiny,$BLOQUE,$BITS_PESOS,$BITS_METADATA,$BITS_TOTAL,$BITS_EFECTIVOS" \
    >> "$OUTPUT_CSV"

# Q8_0
# 32 pesos x 8 bits + d FP16
BLOQUE=32
BITS_PESOS=$((32 * 8))
BITS_METADATA=16
BITS_TOTAL=$((BITS_PESOS + BITS_METADATA))
BITS_EFECTIVOS=$(awk -v total="$BITS_TOTAL" -v bloque="$BLOQUE" \
    'BEGIN { printf "%.2f", total / bloque }')
echo "q8_0,$BLOQUE,$BITS_PESOS,$BITS_METADATA,$BITS_TOTAL,$BITS_EFECTIVOS" \
    >> "$OUTPUT_CSV"
    
# Q5_1
# 32 pesos x 5 bits + d FP16 + m FP16
BLOQUE=32
BITS_PESOS=$((32 * 5))
BITS_METADATA=$((16 + 16))
BITS_TOTAL=$((BITS_PESOS + BITS_METADATA))

BITS_EFECTIVOS=$(awk -v total="$BITS_TOTAL" -v bloque="$BLOQUE" \
    'BEGIN { printf "%.2f", total / bloque }')

echo "q5_1,$BLOQUE,$BITS_PESOS,$BITS_METADATA,$BITS_TOTAL,$BITS_EFECTIVOS" \
    >> "$OUTPUT_CSV"

# Q4_0
# 32 pesos x 4 bits + d FP16
BLOQUE=32
BITS_PESOS=$((32 * 4))
BITS_METADATA=16
BITS_TOTAL=$((BITS_PESOS + BITS_METADATA))

BITS_EFECTIVOS=$(awk -v total="$BITS_TOTAL" -v bloque="$BLOQUE" \
    'BEGIN { printf "%.2f", total / bloque }')

echo "q4_0,$BLOQUE,$BITS_PESOS,$BITS_METADATA,$BITS_TOTAL,$BITS_EFECTIVOS" \
    >> "$OUTPUT_CSV"