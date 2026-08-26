#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# R3 - Propagación del error ASR -> LLM
#
# Utiliza directamente:
#   - asr_pipeline.jl
#   - translate_reference.jl
#
# Condiciones:
#   1. Referencia -> LLM
#   2. Base -> LLM
#   3. Base-Q4_0 -> LLM
#
# LLM:
#   - Gemma3:1B FP16
#   - Gemma3:1B Q8_0
#   - Gemma3:1B Q4_K_M
# ============================================================


# ------------------------------------------------------------
# Archivos Julia existentes
# ------------------------------------------------------------

ASR_SCRIPT="asr_pipeline_igpu.jl"
TRANSLATE_SCRIPT="translate_reference.jl"

GROUNDTRUTH="groundtruth.txt"
GROUNDTRUTH_EN="groundtruth_en.txt"

OUT_DIR="out"


# ------------------------------------------------------------
# Modelos Whisper
# ------------------------------------------------------------

WHISPER_BASE="whisper.cpp/models/ggml-base.bin"
WHISPER_Q4="whisper.cpp/models/ggml-base-q4_0.bin"


# ------------------------------------------------------------
# Modelos Ollama
#
# Si "ollama list" muestra nombres diferentes,
# cambia solamente estas tres líneas.
# ------------------------------------------------------------

LLM_FP16="gemma3:1b-it-fp16"
LLM_Q8="gemma3:1b-it-q8_0"
LLM_Q4="gemma3:1b-it-q4_K_M"

TARGET_LANG="English"


# ------------------------------------------------------------
# Salidas
# ------------------------------------------------------------

RESULT_DIR="scripts_out/R3/propagacion"

ASR_RESULT_DIR="$RESULT_DIR/asr"
TRANSCRIPT_DIR="$RESULT_DIR/transcripciones"
LLM_RESULT_DIR="$RESULT_DIR/ollama"

mkdir -p \
    "$ASR_RESULT_DIR" \
    "$TRANSCRIPT_DIR/base" \
    "$TRANSCRIPT_DIR/base_q4_0" \
    "$LLM_RESULT_DIR"


# ------------------------------------------------------------
# Colores
# ------------------------------------------------------------

GREEN="\033[0;32m"
CYAN="\033[0;36m"
YELLOW="\033[1;33m"
RED="\033[0;31m"
NC="\033[0m"


# ------------------------------------------------------------
# Comprobaciones
# ------------------------------------------------------------

echo -e "${CYAN}==============================================${NC}"
echo -e "${CYAN} R3 - Propagación del error ASR -> LLM${NC}"
echo -e "${CYAN}==============================================${NC}"
echo

for archivo in \
    "$ASR_SCRIPT" \
    "$TRANSLATE_SCRIPT" \
    "$GROUNDTRUTH" \
    "$GROUNDTRUTH_EN" \
    "$WHISPER_BASE" \
    "$WHISPER_Q4"
do
    if [[ ! -f "$archivo" ]]; then
        echo -e "${RED}ERROR: no existe:${NC}"
        echo "$archivo"
        exit 1
    fi
done


if ! command -v julia >/dev/null 2>&1; then
    echo -e "${RED}ERROR: Julia no está instalado o no está en PATH.${NC}"
    exit 1
fi


if ! command -v ollama >/dev/null 2>&1; then
    echo -e "${RED}ERROR: Ollama no está instalado o no está en PATH.${NC}"
    exit 1
fi


echo -e "${GREEN}Archivos requeridos encontrados.${NC}"
echo


# ------------------------------------------------------------
# Mostrar modelos Ollama disponibles
# ------------------------------------------------------------

echo -e "${CYAN}Modelos Ollama disponibles:${NC}"
ollama list
echo


# ------------------------------------------------------------
# Backup seguro de groundtruth.txt
#
# translate_reference.jl tiene la ruta hardcodeada.
# Por eso se reemplaza TEMPORALMENTE durante las
# condiciones Base y Base-Q4_0.
# ------------------------------------------------------------

GROUNDTRUTH_BACKUP="$(mktemp)"

cp "$GROUNDTRUTH" "$GROUNDTRUTH_BACKUP"


restaurar_groundtruth() {

    if [[ -f "$GROUNDTRUTH_BACKUP" ]]; then
        cp "$GROUNDTRUTH_BACKUP" "$GROUNDTRUTH"
        rm -f "$GROUNDTRUTH_BACKUP"
    fi
}


# Se restaura incluso si presionas Ctrl+C
# o algún comando falla.
trap restaurar_groundtruth EXIT INT TERM


# ============================================================
# 1. ASR BASE
# ============================================================

echo
echo -e "${CYAN}==============================================${NC}"
echo -e "${CYAN} Generando ASR Base${NC}"
echo -e "${CYAN}==============================================${NC}"
echo


rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"


julia "$ASR_SCRIPT" "$WHISPER_BASE" \
    2>&1 | tee "$ASR_RESULT_DIR/base.txt"


# Guardar transcripciones antes de que otra corrida
# sobrescriba out/
cp "$OUT_DIR"/*.txt "$TRANSCRIPT_DIR/base/"


echo
echo -e "${GREEN}Transcripciones Base guardadas en:${NC}"
echo "$TRANSCRIPT_DIR/base"


# ============================================================
# 2. ASR BASE Q4_0
# ============================================================

echo
echo -e "${CYAN}==============================================${NC}"
echo -e "${CYAN} Generando ASR Base-Q4_0${NC}"
echo -e "${CYAN}==============================================${NC}"
echo


rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"


julia "$ASR_SCRIPT" "$WHISPER_Q4" \
    2>&1 | tee "$ASR_RESULT_DIR/base_q4_0.txt"


cp "$OUT_DIR"/*.txt "$TRANSCRIPT_DIR/base_q4_0/"


echo
echo -e "${GREEN}Transcripciones Base-Q4_0 guardadas en:${NC}"
echo "$TRANSCRIPT_DIR/base_q4_0"


# ============================================================
# Función:
# construir groundtruth.txt temporal desde una carpeta
# de transcripciones Whisper.
#
# Resultado:
#
# test1.wav,"texto generado por whisper"
# test2.wav,"texto generado por whisper"
# ...
# ============================================================

crear_groundtruth_desde_asr() {

    local carpeta="$1"

    : > "$GROUNDTRUTH"

    for txt in "$carpeta"/*.txt
    do

        nombre="$(basename "$txt" .txt)"
        wav="${nombre}.wav"

        # Whisper genera normalmente una sola línea,
        # pero eliminamos saltos de línea por seguridad.
        texto="$(
            tr '\n' ' ' < "$txt" |
            sed 's/[[:space:]]\+/ /g' |
            sed 's/^ //; s/ $//'
        )"

        # Escapar comillas para mantener el formato CSV.
        texto="${texto//\"/\\\"}"

        printf '%s,"%s"\n' \
            "$wav" \
            "$texto" \
            >> "$GROUNDTRUTH"

    done
}


# ============================================================
# Función:
# ejecutar una condición para los tres LLM
# ============================================================

ejecutar_llms() {

    local condicion="$1"
    local nombre_archivo="$2"

    echo
    echo -e "${YELLOW}----------------------------------------------${NC}"
    echo -e "${YELLOW} Condición: $condicion${NC}"
    echo -e "${YELLOW}----------------------------------------------${NC}"


    # --------------------------------------------------------
    # FP16
    # --------------------------------------------------------

    echo
    echo -e "${CYAN}Gemma3:1B FP16${NC}"

    julia "$TRANSLATE_SCRIPT" \
        "$LLM_FP16" \
        "$TARGET_LANG" \
        2>&1 | tee \
        "$LLM_RESULT_DIR/fp16_${nombre_archivo}.txt"


    # --------------------------------------------------------
    # Q8_0
    # --------------------------------------------------------

    echo
    echo -e "${CYAN}Gemma3:1B Q8_0${NC}"

    julia "$TRANSLATE_SCRIPT" \
        "$LLM_Q8" \
        "$TARGET_LANG" \
        2>&1 | tee \
        "$LLM_RESULT_DIR/q8_0_${nombre_archivo}.txt"


    # --------------------------------------------------------
    # Q4_K_M
    # --------------------------------------------------------

    echo
    echo -e "${CYAN}Gemma3:1B Q4_K_M${NC}"

    julia "$TRANSLATE_SCRIPT" \
        "$LLM_Q4" \
        "$TARGET_LANG" \
        2>&1 | tee \
        "$LLM_RESULT_DIR/q4_k_m_${nombre_archivo}.txt"
}


# ============================================================
# 3. REFERENCIA -> LLM
#
# Aquí se utiliza groundtruth.txt ORIGINAL.
# WER de entrada = 0.
# ============================================================

echo
echo -e "${CYAN}==============================================${NC}"
echo -e "${CYAN} Referencia -> LLM${NC}"
echo -e "${CYAN}==============================================${NC}"


cp "$GROUNDTRUTH_BACKUP" "$GROUNDTRUTH"


ejecutar_llms \
    "Referencia -> LLM" \
    "referencia"


# ============================================================
# 4. BASE -> LLM
#
# Se reemplaza temporalmente groundtruth.txt con las
# transcripciones generadas por Base.
# translate_reference.jl las traducirá y comparará
# contra groundtruth_en.txt.
# ============================================================

echo
echo -e "${CYAN}==============================================${NC}"
echo -e "${CYAN} Base -> LLM${NC}"
echo -e "${CYAN}==============================================${NC}"


crear_groundtruth_desde_asr \
    "$TRANSCRIPT_DIR/base"


ejecutar_llms \
    "Base -> LLM" \
    "base"


# ============================================================
# 5. BASE-Q4_0 -> LLM
# ============================================================

echo
echo -e "${CYAN}==============================================${NC}"
echo -e "${CYAN} Base-Q4_0 -> LLM${NC}"
echo -e "${CYAN}==============================================${NC}"


crear_groundtruth_desde_asr \
    "$TRANSCRIPT_DIR/base_q4_0"


ejecutar_llms \
    "Base-Q4_0 -> LLM" \
    "base_q4_0"


# ============================================================
# Restaurar groundtruth original
# ============================================================

restaurar_groundtruth

# Evitar que trap intente hacerlo nuevamente.
trap - EXIT INT TERM


# ============================================================
# 6. RESUMEN DE RESULTADOS
#
# No recalcula absolutamente nada.
# Solo extrae las líneas "promedio" que imprimieron
# tus propios códigos Julia.
# ============================================================

RESUMEN="$RESULT_DIR/resumen.txt"


{
    echo "============================================================"
    echo "R3 - PROPAGACIÓN DEL ERROR ASR -> LLM"
    echo "============================================================"
    echo

    echo "ERROR DE ENTRADA - ASR"
    echo "------------------------------------------------------------"

    echo "Base:"
    grep -E \
        "WER promedio|CER promedio|RTF promedio" \
        "$ASR_RESULT_DIR/base.txt" || true

    echo

    echo "Base-Q4_0:"
    grep -E \
        "WER promedio|CER promedio|RTF promedio" \
        "$ASR_RESULT_DIR/base_q4_0.txt" || true


    echo
    echo
    echo "ERROR DE SALIDA - chrF"
    echo "============================================================"


    echo
    echo "Gemma3:1B FP16"
    echo "------------------------------------------------------------"

    echo -n "Referencia -> LLM: "
    grep "chrF promedio" \
        "$LLM_RESULT_DIR/fp16_referencia.txt" || true

    echo -n "Base -> LLM: "
    grep "chrF promedio" \
        "$LLM_RESULT_DIR/fp16_base.txt" || true

    echo -n "Base-Q4_0 -> LLM: "
    grep "chrF promedio" \
        "$LLM_RESULT_DIR/fp16_base_q4_0.txt" || true


    echo
    echo "Gemma3:1B Q8_0"
    echo "------------------------------------------------------------"

    echo -n "Referencia -> LLM: "
    grep "chrF promedio" \
        "$LLM_RESULT_DIR/q8_0_referencia.txt" || true

    echo -n "Base -> LLM: "
    grep "chrF promedio" \
        "$LLM_RESULT_DIR/q8_0_base.txt" || true

    echo -n "Base-Q4_0 -> LLM: "
    grep "chrF promedio" \
        "$LLM_RESULT_DIR/q8_0_base_q4_0.txt" || true


    echo
    echo "Gemma3:1B Q4_K_M"
    echo "------------------------------------------------------------"

    echo -n "Referencia -> LLM: "
    grep "chrF promedio" \
        "$LLM_RESULT_DIR/q4_k_m_referencia.txt" || true

    echo -n "Base -> LLM: "
    grep "chrF promedio" \
        "$LLM_RESULT_DIR/q4_k_m_base.txt" || true

    echo -n "Base-Q4_0 -> LLM: "
    grep "chrF promedio" \
        "$LLM_RESULT_DIR/q4_k_m_base_q4_0.txt" || true

} | tee "$RESUMEN"


echo
echo -e "${GREEN}==============================================${NC}"
echo -e "${GREEN} Experimento finalizado${NC}"
echo -e "${GREEN}==============================================${NC}"
echo

echo "Resultados:"
echo "  $RESULT_DIR"
echo
echo "Resumen:"
echo "  $RESUMEN"
echo