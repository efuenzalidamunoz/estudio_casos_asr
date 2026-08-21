#!/usr/bin/env bash

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WHISPER_DIR="$ROOT/whisper.cpp"
WHISPER="$WHISPER_DIR/build/bin/whisper-cli"
MODEL_DIR="$WHISPER_DIR/models"
AUDIO_DIR="$ROOT/audio"
OUTPUT_DIR="$ROOT/scripts_out/R2/ram"
MEMORY_CSV="$OUTPUT_DIR/memory_all_models.csv"
SUMMARY_CSV="$OUTPUT_DIR/ram_promedios.csv"
LANGUAGE="es"
mkdir -p "$OUTPUT_DIR"

MODELS=(
    "base|ggml-base.bin"
    "tiny|ggml-tiny.bin"
    "q8_0|ggml-base-q8_0.bin"
    "q5_1|ggml-base-q5_1.bin"
    "q4_0|ggml-base-q4_0.bin"
)

if [ ! -x "$WHISPER" ]; then
    echo "ERROR: no se encontró whisper-cli:"
    echo "$WHISPER"
    exit 1
fi

if [ ! -x /usr/bin/time ]; then
    echo "ERROR: falta GNU time."
    echo "Instálalo con:"
    echo "sudo dnf install time"
    exit 1
fi

for i in $(seq 1 10); do
    if [ ! -f "$AUDIO_DIR/test${i}.wav" ]; then
        echo "ERROR: no existe:"
        echo "$AUDIO_DIR/test${i}.wav"
        exit 1
    fi
done

echo "formato,audio,rss_kib,rss_mib" > "$MEMORY_CSV"

echo "Procesando..."

for ENTRY in "${MODELS[@]}"; do

    IFS='|' read -r FORMAT MODEL <<< "$ENTRY"

    MODEL_PATH="$MODEL_DIR/$MODEL"

    if [ ! -f "$MODEL_PATH" ]; then
        echo "ERROR: no existe el modelo:"
        echo "$MODEL_PATH"
        exit 1
    fi

    for i in $(seq 1 10); do

        AUDIO_NAME="test${i}.wav"
        AUDIO="$AUDIO_DIR/$AUDIO_NAME"

        TMP="$(mktemp)"

        # whisper necesita un prefijo para -of.
        # Se usa uno temporal y luego se elimina.
        OUT_PREFIX="$(mktemp -u "/tmp/whisper_ram_${FORMAT}_${i}_XXXX")"

        /usr/bin/time -v \
            "$WHISPER" \
            -m "$MODEL_PATH" \
            -f "$AUDIO" \
            -l "$LANGUAGE" \
            -nt \
            -np \
            -otxt \
            -of "$OUT_PREFIX" \
            -t 4 \
            --temperature 0.0 \
            -ng \
            >/dev/null \
            2>"$TMP"

        # GNU time entrega RSS máximo en KiB
        RSS_KIB=$(
            grep "Maximum resident set size" "$TMP" |
            awk -F ':' '{
                gsub(/^[ \t]+/, "", $2)
                gsub(/[ \t]+$/, "", $2)
                print $2
            }'
        )

        rm -f "$TMP"
        rm -f "${OUT_PREFIX}.txt"

        if [ -z "$RSS_KIB" ]; then
            echo "ERROR: no se pudo obtener RSS para $FORMAT / $AUDIO_NAME"
            exit 1
        fi

        RSS_MIB=$(
            awk -v rss="$RSS_KIB" \
                'BEGIN { printf "%.2f", rss / 1024 }'
        )

        echo "$FORMAT,$AUDIO_NAME,$RSS_KIB,$RSS_MIB" \
            >> "$MEMORY_CSV"
    done

done

TMP_SUMMARY="$(mktemp)"

awk -F',' '
NR > 1 {
    formato = $1
    ram = $4 + 0

    sum[formato] += ram
    count[formato]++

    if (!(formato in min) || ram < min[formato]) {
        min[formato] = ram
    }

    if (!(formato in max) || ram > max[formato]) {
        max[formato] = ram
    }
}

END {
    for (formato in sum) {
        promedio = sum[formato] / count[formato]

        printf "%s,%.2f,%.2f,%.2f\n",
            formato,
            promedio,
            min[formato],
            max[formato]
    }
}
' "$MEMORY_CSV" > "$TMP_SUMMARY"

echo "formato,ram_promedio_mib,ram_min_mib,ram_max_mib" \
    > "$SUMMARY_CSV"

sort -t',' -k2,2n "$TMP_SUMMARY" >> "$SUMMARY_CSV"

rm -f "$TMP_SUMMARY"

echo "Medición terminada."
echo "$MEMORY_CSV"
echo "$SUMMARY_CSV"
