#!/usr/bin/env bash

set -euo pipefail
export LC_ALL=C

# ============================================================
# CONFIGURACIÓN
# ============================================================

WHISPER_DIR="whisper.cpp"
WHISPER="$WHISPER_DIR/build-vulkan/bin/whisper-cli"

MODEL_DIR="$WHISPER_DIR/models"
AUDIO_DIR="audio"

OUTPUT_DIR="scripts_out/R2/ram"
MEMORY_CSV="$OUTPUT_DIR/memory_all_models.csv"
SUMMARY_CSV="$OUTPUT_DIR/ram_promedios.csv"

LANGUAGE="es"

# Intel UHD Graphics (Comet Lake-H GT2)
# 8086 = Intel
# 9bc4 = dispositivo
# ! = ocultar los demás dispositivos Vulkan
export MESA_VK_DEVICE_SELECT="8086:9bc4!"

# Cada audio se ejecuta 3 veces.
# 10 audios x 3 = 30 mediciones por modelo.
REPEATS=3

# Baseline:
# 20 muestras x 0.05 s = ~1 segundo.
BASELINE_SAMPLES=20

# Durante Whisper se promedian grupos de 5 muestras.
# 5 x 0.05 s = ventana de ~250 ms.
SMOOTH_WINDOW=5

SAMPLE_INTERVAL=0.05

# Tiempo de estabilización entre ejecuciones.
COOLDOWN_SECONDS=1

mkdir -p "$OUTPUT_DIR"

MODELS=(
    "base|ggml-base.bin"
    "tiny|ggml-tiny.bin"
    "q8_0|ggml-base-q8_0.bin"
    "q5_1|ggml-base-q5_1.bin"
    "q4_0|ggml-base-q4_0.bin"
)

# ============================================================
# FUNCIONES
# ============================================================

# Devuelve MemAvailable en KiB.
mem_available_kib() {
    awk '/^MemAvailable:/ {print $2; exit}' /proc/meminfo
}

# ------------------------------------------------------------
# Baseline estable:
# mediana de BASELINE_SAMPLES lecturas
# ------------------------------------------------------------

baseline_mem_available_kib() {

    local tmp
    local result

    tmp="$(mktemp)"

    for ((sample=1; sample<=BASELINE_SAMPLES; sample++)); do
        mem_available_kib >> "$tmp"
        sleep "$SAMPLE_INTERVAL"
    done

    result="$(
        sort -n "$tmp" |
        awk '
        {
            values[NR] = $1
        }
        END {
            if (NR == 0) exit 1
            if (NR % 2 == 1) {
                print values[(NR + 1) / 2]
            } else {
                print int((values[NR / 2] + values[NR / 2 + 1]) / 2)
            }
        }'
    )"

    rm -f "$tmp"

    printf "%s\n" "$result"
}

# ------------------------------------------------------------
# Monitoreo de MemAvailable
#
# En lugar de usar una lectura instantánea, toma grupos de
# SMOOTH_WINDOW muestras y calcula su promedio.
# ------------------------------------------------------------

monitor_memory() {

    local baseline="$1"
    local stop_file="$2"
    local result_file="$3"

    local minimum="$baseline"

    while [ ! -f "$stop_file" ]; do

        local sum=0
        local samples=0

        for ((sample=1; sample<=SMOOTH_WINDOW; sample++)); do

            local current

            current="$(mem_available_kib)"

            sum=$((sum + current))
            samples=$((samples + 1))

            sleep "$SAMPLE_INTERVAL"

            if [ -f "$stop_file" ]; then
                break
            fi

        done

        if [ "$samples" -gt 0 ]; then

            local average

            average=$((sum / samples))

            if [ "$average" -lt "$minimum" ]; then
                minimum="$average"
            fi

        fi

    done

    printf "%s\n" "$minimum" > "$result_file"
}

# ------------------------------------------------------------
# Mediana de un archivo que contiene un número por línea
# ------------------------------------------------------------

median_from_file() {

    local file="$1"

    sort -n "$file" |
    awk '
    {
        values[NR] = $1
    }
    END {
        if (NR == 0) exit 1

        if (NR % 2 == 1) {
            median = values[(NR + 1) / 2]
        } else {
            median = (values[NR / 2] + values[NR / 2 + 1]) / 2
        }

        printf "%.2f", median
    }'
}

# ============================================================
# COMPROBACIONES
# ============================================================

if [ ! -x "$WHISPER" ]; then
    echo "ERROR: no se encontró whisper-cli Vulkan:"
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

    AUDIO="$AUDIO_DIR/test${i}.wav"

    if [ ! -f "$AUDIO" ]; then
        echo "ERROR: no existe:"
        echo "$AUDIO"
        exit 1
    fi

done

for ENTRY in "${MODELS[@]}"; do

    IFS='|' read -r FORMAT MODEL <<< "$ENTRY"

    MODEL_PATH="$MODEL_DIR/$MODEL"

    if [ ! -f "$MODEL_PATH" ]; then
        echo "ERROR: no existe:"
        echo "$MODEL_PATH"
        exit 1
    fi

done

# Avisar si Ollama sigue ejecutándose.
if systemctl is-active --quiet ollama 2>/dev/null; then

    echo
    echo "============================================================"
    echo "ADVERTENCIA"
    echo "============================================================"
    echo
    echo "Ollama está ejecutándose."
    echo "Puede introducir ruido en la medición."
    echo
    echo "Se recomienda ejecutar:"
    echo
    echo "sudo systemctl stop ollama"
    echo

fi

# ============================================================
# CSV DE MEDICIONES INDIVIDUALES
# ============================================================

echo "formato,audio,repeticion,baseline_kib,min_available_kib,delta_ram_kib,delta_ram_mib,rss_kib,rss_mib" \
    > "$MEMORY_CSV"

echo
echo "============================================================"
echo " MEDICIÓN DE MEMORIA WHISPER.CPP + INTEL iGPU"
echo "============================================================"
echo
echo "Backend:                  Vulkan"
echo "GPU:                      Intel UHD Graphics (CML GT2)"
echo "Vulkan device:            $MESA_VK_DEVICE_SELECT"
echo "Threads:                  4"
echo "Temperature:              0.0"
echo "Repeticiones por audio:   $REPEATS"
echo "Muestras baseline:        $BASELINE_SAMPLES"
echo "Ventana de suavizado:     $SMOOTH_WINDOW"
echo "Intervalo entre muestras: $SAMPLE_INTERVAL s"
echo

# ============================================================
# EJECUCIÓN
# ============================================================

for ENTRY in "${MODELS[@]}"; do

    IFS='|' read -r FORMAT MODEL <<< "$ENTRY"

    MODEL_PATH="$MODEL_DIR/$MODEL"

    echo
    echo "============================================================"
    echo " MODELO: $FORMAT"
    echo "============================================================"
    echo

    # ========================================================
    # CALENTAMIENTO
    #
    # La primera ejecución puede contener inicialización
    # adicional de Vulkan, shaders, buffers, etc.
    # Esta corrida NO se mide.
    # ========================================================

    echo "Calentamiento..."

    "$WHISPER" \
        -m "$MODEL_PATH" \
        -f "$AUDIO_DIR/test1.wav" \
        -l "$LANGUAGE" \
        -nt \
        -np \
        -t 4 \
        --temperature 0.0 \
        >/dev/null \
        2>/dev/null

    sleep 2

    echo "Calentamiento terminado."
    echo

    # ========================================================
    # REPETICIONES
    # ========================================================

    for REP in $(seq 1 "$REPEATS"); do

        echo "------------------------------------------------------------"
        echo "Repetición $REP / $REPEATS"
        echo "------------------------------------------------------------"

        for i in $(seq 1 10); do

            AUDIO_NAME="test${i}.wav"
            AUDIO="$AUDIO_DIR/$AUDIO_NAME"

            TMP_TIME="$(mktemp)"
            STOP_FILE="$(mktemp)"
            RESULT_FILE="$(mktemp)"

            # Borramos STOP_FILE porque su existencia
            # funciona como señal para terminar el monitor.
            rm -f "$STOP_FILE"

            OUT_PREFIX="$(
                mktemp "/tmp/whisper_ram_${FORMAT}_${i}_${REP}_XXXXXX"
            )"

            # Solo necesitamos el nombre como prefijo.
            rm -f "$OUT_PREFIX"

            # ------------------------------------------------
            # ESTABILIZACIÓN
            # ------------------------------------------------

            sleep "$COOLDOWN_SECONDS"

            # ------------------------------------------------
            # BASELINE
            # ------------------------------------------------

            BASELINE_KIB="$(baseline_mem_available_kib)"

            # ------------------------------------------------
            # INICIAR MONITOR DE RAM
            # ------------------------------------------------

            monitor_memory \
                "$BASELINE_KIB" \
                "$STOP_FILE" \
                "$RESULT_FILE" &

            MONITOR_PID=$!

            # ------------------------------------------------
            # EJECUTAR WHISPER
            # ------------------------------------------------

            set +e

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
                >/dev/null \
                2>"$TMP_TIME"

            STATUS=$?

            set -e

            # ------------------------------------------------
            # DETENER MONITOR
            # ------------------------------------------------

            touch "$STOP_FILE"

            wait "$MONITOR_PID"

            # ------------------------------------------------
            # VERIFICAR WHISPER
            # ------------------------------------------------

            if [ "$STATUS" -ne 0 ]; then

                echo
                echo "ERROR ejecutando:"
                echo "$FORMAT / $AUDIO_NAME / repetición $REP"
                echo
                cat "$TMP_TIME"

                rm -f "$TMP_TIME"
                rm -f "$STOP_FILE"
                rm -f "$RESULT_FILE"
                rm -f "${OUT_PREFIX}.txt"

                exit "$STATUS"
            fi

            # ------------------------------------------------
            # RESULTADO DEL MONITOR
            # ------------------------------------------------

            MIN_AVAILABLE_KIB="$(cat "$RESULT_FILE")"

            # ------------------------------------------------
            # DELTA DE RAM DEL SISTEMA
            # ------------------------------------------------

            DELTA_KIB=$((BASELINE_KIB - MIN_AVAILABLE_KIB))

            # MemAvailable puede aumentar durante la ejecución
            # si el kernel libera caché.
            if [ "$DELTA_KIB" -lt 0 ]; then
                DELTA_KIB=0
            fi

            DELTA_MIB="$(
                awk -v ram="$DELTA_KIB" \
                    'BEGIN {printf "%.2f", ram / 1024}'
            )"

            # ------------------------------------------------
            # RSS DEL PROCESO
            # ------------------------------------------------

            RSS_KIB="$(
                grep "Maximum resident set size" "$TMP_TIME" |
                awk -F ':' '{gsub(/^[ \t]+/, "", $2); gsub(/[ \t]+$/, "", $2); print $2}'
            )"

            if [ -z "$RSS_KIB" ]; then

                echo
                echo "ERROR: no se pudo obtener RSS:"
                echo "$FORMAT / $AUDIO_NAME / repetición $REP"

                rm -f "$TMP_TIME"
                rm -f "$STOP_FILE"
                rm -f "$RESULT_FILE"
                rm -f "${OUT_PREFIX}.txt"

                exit 1
            fi

            RSS_MIB="$(
                awk -v rss="$RSS_KIB" \
                    'BEGIN {printf "%.2f", rss / 1024}'
            )"

            # ------------------------------------------------
            # GUARDAR
            # ------------------------------------------------

            echo "$FORMAT,$AUDIO_NAME,$REP,$BASELINE_KIB,$MIN_AVAILABLE_KIB,$DELTA_KIB,$DELTA_MIB,$RSS_KIB,$RSS_MIB" \
                >> "$MEMORY_CSV"

            printf \
                "  %-10s ΔRAM: %7s MiB   RSS: %7s MiB\n" \
                "$AUDIO_NAME" \
                "$DELTA_MIB" \
                "$RSS_MIB"

            # ------------------------------------------------
            # LIMPIEZA
            # ------------------------------------------------

            rm -f "$TMP_TIME"
            rm -f "$STOP_FILE"
            rm -f "$RESULT_FILE"
            rm -f "${OUT_PREFIX}.txt"

        done

        echo

    done

done

# ============================================================
# RESUMEN ESTADÍSTICO
# ============================================================

echo
echo "============================================================"
echo " CALCULANDO ESTADÍSTICAS"
echo "============================================================"
echo

echo "formato,n,ram_media_mib,ram_mediana_mib,ram_desv_std_mib,ram_min_mib,ram_max_mib,rss_promedio_mib" \
    > "$SUMMARY_CSV"

for FORMAT in base tiny q8_0 q5_1 q4_0; do

    TMP_VALUES="$(mktemp)"

    # Columna 7 = delta RAM MiB
    awk -F',' -v formato="$FORMAT" \
        'NR > 1 && $1 == formato {print $7}' \
        "$MEMORY_CSV" \
        > "$TMP_VALUES"

    # --------------------------------------------------------
    # MEDIA, DESVIACIÓN ESTÁNDAR, MÍNIMO Y MÁXIMO
    # --------------------------------------------------------

    STATS="$(
        awk '
        {
            x = $1 + 0
            n++
            sum += x
            sumsq += x * x

            if (n == 1 || x < minimum) minimum = x
            if (n == 1 || x > maximum) maximum = x
        }

        END {
            if (n == 0) exit 1

            mean = sum / n

            if (n > 1) {
                variance = (sumsq - (sum * sum / n)) / (n - 1)
                if (variance < 0) variance = 0
                stddev = sqrt(variance)
            } else {
                stddev = 0
            }

            printf "%d %.2f %.2f %.2f %.2f\n", n, mean, stddev, minimum, maximum
        }' "$TMP_VALUES"
    )"

    read -r N MEAN STDDEV MINIMUM MAXIMUM <<< "$STATS"

    # --------------------------------------------------------
    # MEDIANA
    # --------------------------------------------------------

    MEDIAN="$(median_from_file "$TMP_VALUES")"

    # --------------------------------------------------------
    # RSS PROMEDIO
    # --------------------------------------------------------

    RSS_MEAN="$(
        awk -F',' -v formato="$FORMAT" '
        NR > 1 && $1 == formato {
            sum += $9
            n++
        }
        END {
            if (n > 0) {
                printf "%.2f", sum / n
            } else {
                printf "0.00"
            }
        }' "$MEMORY_CSV"
    )"

    # --------------------------------------------------------
    # GUARDAR RESUMEN
    # --------------------------------------------------------

    echo "$FORMAT,$N,$MEAN,$MEDIAN,$STDDEV,$MINIMUM,$MAXIMUM,$RSS_MEAN" \
        >> "$SUMMARY_CSV"

    rm -f "$TMP_VALUES"

done

# ============================================================
# RESULTADO
# ============================================================

echo
echo "============================================================"
echo " MEDICIÓN TERMINADA"
echo "============================================================"
echo

echo "Mediciones individuales:"
echo "$MEMORY_CSV"
echo

echo "Resumen:"
echo "$SUMMARY_CSV"
echo

if command -v column >/dev/null 2>&1; then
    column -s',' -t "$SUMMARY_CSV"
else
    cat "$SUMMARY_CSV"
fi

echo
echo "Para el informe:"
echo
echo "  - Usa principalmente ram_mediana_mib"
echo "  - Usa ram_desv_std_mib para evaluar estabilidad"
echo "  - rss_promedio_mib queda como medición auxiliar"
echo