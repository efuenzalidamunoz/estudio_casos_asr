#!/usr/bin/env bash

set -euo pipefail

MODELS_DIR="whisper.cpp/models"
WHISPER_CLI="whisper.cpp/build/bin/whisper-cli"

# ============================================================
# MODELOS A COMPARAR
# ============================================================

F32_MODEL="$MODELS_DIR/ggml-base.bin"
F16_MODEL="$MODELS_DIR/ggml-tiny.bin"

Q8_MODEL="$MODELS_DIR/ggml-base-q8_0.bin"
Q5_MODEL="$MODELS_DIR/ggml-base-q5_1.bin"
Q4_MODEL="$MODELS_DIR/ggml-base-q4_0.bin"

# ============================================================
# FUNCIONES
# ============================================================

bytes_to_mb() {
    awk -v b="$1" 'BEGIN {
        printf "%.2f", b / 1000000
    }'
}

file_size() {
    stat -c%s "$1"
}

# Bits efectivos por peso considerando el costo
# del factor de escala por bloque de 32 pesos.

bits_por_peso() {

    case "$1" in

        F32)
            echo "32.00"
            ;;

        F16)
            echo "16.00"
            ;;

        Q8_0)
            # 32 pesos x 8 bits + 16 bits del factor d
            awk 'BEGIN {
                printf "%.2f", (32*8 + 16) / 32
            }'
            ;;

        Q5_1)
            # 32 pesos x 5 bits
            # + escala FP16
            # + mínimo FP16
            awk 'BEGIN {
                printf "%.2f", (32*5 + 32) / 32
            }'
            ;;

        Q4_0)
            # 32 pesos x 4 bits + escala FP16
            awk 'BEGIN {
                printf "%.2f", (32*4 + 16) / 32
            }'
            ;;

        *)
            echo "N/D"
            ;;

    esac
}

# ============================================================
# MEDICION DE RAM
# ============================================================

medir_ram() {
    local modelo="$1"
    local audio="$2"
    local tmpfile

    tmpfile=$(mktemp)

    /usr/bin/time -f "%M" \
        "$WHISPER_CLI" \
        -m "$modelo" \
        -f "$audio" \
        >/dev/null \
        2>"$tmpfile" || true

    # Buscar la ultima linea que contenga solamente un numero
    local ram_kb
    ram_kb=$(grep -E '^[0-9]+$' "$tmpfile" | tail -n 1)

    rm -f "$tmpfile"

    if [[ -n "$ram_kb" ]]; then
        awk -v kb="$ram_kb" 'BEGIN {
            printf "%.2f", kb / 1024
        }'
    else
        echo "N/D"
    fi
}

# ============================================================
# REPORTE
# ============================================================

analizar() {

    local formato="$1"
    local modelo="$2"
    local parametros="$3"
    local d="$4"

    if [[ ! -f "$modelo" ]]; then
        echo "ADVERTENCIA: no existe $modelo"
        return
    fi

    if [[ ! -f "audio/test1.wav" ]]; then
        echo "ERROR: no existe audio/test1.wav"
        return
    fi

    local bytes
    local mb
    local ram
    local bits

    bytes=$(file_size "$modelo")
    mb=$(bytes_to_mb "$bytes")

    ram=$(medir_ram "$modelo" "audio/test1.wav")

    bits=$(bits_por_peso "$formato")

    printf "%-8s %-24s %-10s %-12s %-12s %-12s %-12s\n" \
        "$formato" \
        "$(basename "$modelo")" \
        "$parametros" \
        "$mb" \
        "$ram" \
        "$bits" \
        "$d"
}

# ============================================================
# ENCABEZADO
# ============================================================

echo
echo "=========================================================================="
echo " ANALISIS DE REPRESENTACION NUMERICA - WHISPER.CPP"
echo "=========================================================================="
echo

printf "%-8s %-24s %-10s %-12s %-12s %-12s %-12s\n" \
    "Formato" \
    "Modelo" \
    "Params" \
    "Disco MB" \
    "RAM MB" \
    "Bits/peso" \
    "d"

echo "--------------------------------------------------------------------------"

# F32 BASE: 74M parámetros
analizar "F32" "$F32_MODEL" "74M" "N/A"

# F16 TINY: 39M parámetros
analizar "F16" "$F16_MODEL" "39M" "N/A"

# Versiones cuantizadas del BASE
analizar "Q8_0" "$Q8_MODEL" "74M" "por bloque"
analizar "Q5_1" "$Q5_MODEL" "74M" "por bloque"
analizar "Q4_0" "$Q4_MODEL" "74M" "por bloque"

echo "--------------------------------------------------------------------------"

echo
echo "NOTAS:"
echo "1. F32 corresponde al modelo BASE (~74M parámetros)."
echo "2. F16 corresponde al modelo TINY (~39M parámetros)."
echo "3. Q8_0, Q5_1 y Q4_0 corresponden al modelo BASE."
echo "4. Los bits/peso incluyen el costo amortizado del factor de escala."
echo "5. d es un factor de escala por bloque; no necesariamente existe un"
echo "   único valor d para todo el modelo."
echo
