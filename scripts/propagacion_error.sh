#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# R3 - Propagación del error ASR -> LLM
#
# Referencia:
#   translate_reference.jl
#
# Salidas ASR:
#   translate_pipeline.jl
#
# ASR:
#   - Referencia
#   - Base
#   - Tiny
#   - Base-Q8_0
#   - Base-Q5_1
#   - Base-Q4_0
#
# LLM:
#   - Gemma3:1B FP16
#   - Gemma3:1B Q8_0
#   - Gemma3:1B Q4_K_M
#
# Total:
#   6 condiciones x 3 LLM = 18 combinaciones
# ============================================================


# ------------------------------------------------------------
# Scripts Julia
# ------------------------------------------------------------

ASR_SCRIPT="asr_pipeline_igpu.jl"

TRANSLATE_REFERENCE_SCRIPT="translate_reference.jl"
TRANSLATE_PIPELINE_SCRIPT="translate_pipeline.jl"

GROUNDTRUTH="groundtruth.txt"
GROUNDTRUTH_EN="groundtruth_en.txt"

OUT_DIR="out"


# ------------------------------------------------------------
# Modelos Whisper
# ------------------------------------------------------------

ASR_KEYS=(
    "base"
    "tiny"
    "base_q8_0"
    "base_q5_1"
    "base_q4_0"
)

ASR_LABELS=(
    "Base"
    "Tiny"
    "Base-Q8_0"
    "Base-Q5_1"
    "Base-Q4_0"
)

ASR_MODELS=(
    "whisper.cpp/models/ggml-base.bin"
    "whisper.cpp/models/ggml-tiny.bin"
    "whisper.cpp/models/ggml-base-q8_0.bin"
    "whisper.cpp/models/ggml-base-q5_1.bin"
    "whisper.cpp/models/ggml-base-q4_0.bin"
)


# ------------------------------------------------------------
# Modelos Ollama
# ------------------------------------------------------------

LLM_KEYS=(
    "fp16"
    "q8_0"
    "q4_k_m"
)

LLM_LABELS=(
    "Gemma3:1B FP16"
    "Gemma3:1B Q8_0"
    "Gemma3:1B Q4_K_M"
)

LLM_MODELS=(
    "gemma3:1b-it-fp16"
    "gemma3:1b-it-q8_0"
    "gemma3:1b-it-q4_K_M"
)

TARGET_LANG="English"


# ------------------------------------------------------------
# Directorios
# ------------------------------------------------------------

RESULT_DIR="scripts_out/R3/propagacion"

ASR_RESULT_DIR="$RESULT_DIR/asr"
TRANSCRIPT_DIR="$RESULT_DIR/transcripciones"
LLM_RESULT_DIR="$RESULT_DIR/ollama"

RESUMEN="$RESULT_DIR/resumen.txt"
TABLA_CSV="$RESULT_DIR/tabla_completa.csv"

mkdir -p \
    "$ASR_RESULT_DIR" \
    "$TRANSCRIPT_DIR" \
    "$LLM_RESULT_DIR"


# ------------------------------------------------------------
# Colores
# ------------------------------------------------------------

GREEN="\033[0;32m"
CYAN="\033[0;36m"
YELLOW="\033[1;33m"
RED="\033[0;31m"
NC="\033[0m"


# ============================================================
# COMPROBACIONES
# ============================================================

echo -e "${CYAN}============================================================${NC}"
echo -e "${CYAN} R3 - Matriz completa ASR -> LLM${NC}"
echo -e "${CYAN}============================================================${NC}"
echo


for archivo in \
    "$ASR_SCRIPT" \
    "$TRANSLATE_REFERENCE_SCRIPT" \
    "$TRANSLATE_PIPELINE_SCRIPT" \
    "$GROUNDTRUTH" \
    "$GROUNDTRUTH_EN"
do
    if [[ ! -f "$archivo" ]]; then
        echo -e "${RED}ERROR: no existe:${NC}"
        echo "$archivo"
        exit 1
    fi
done


for modelo in "${ASR_MODELS[@]}"
do
    if [[ ! -f "$modelo" ]]; then
        echo -e "${RED}ERROR: no existe el modelo Whisper:${NC}"
        echo "$modelo"
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

echo -e "${CYAN}Modelos Ollama disponibles:${NC}"
ollama list
echo


# ============================================================
# FUNCIÓN: GENERAR ASR
# ============================================================

generar_asr() {

    local key="$1"
    local label="$2"
    local model="$3"

    local transcript_output="$TRANSCRIPT_DIR/$key"
    local result_file="$ASR_RESULT_DIR/${key}.txt"

    echo
    echo -e "${CYAN}============================================================${NC}"
    echo -e "${CYAN} Generando ASR: $label${NC}"
    echo -e "${CYAN}============================================================${NC}"
    echo

    rm -rf "$OUT_DIR"
    mkdir -p "$OUT_DIR"

    rm -rf "$transcript_output"
    mkdir -p "$transcript_output"


    julia "$ASR_SCRIPT" "$model" \
        2>&1 | tee "$result_file"


    # --------------------------------------------------------
    # Comprobar que se generaron transcripciones
    # --------------------------------------------------------

    shopt -s nullglob
    archivos_txt=("$OUT_DIR"/*.txt)
    shopt -u nullglob

    if [[ ${#archivos_txt[@]} -eq 0 ]]; then
        echo -e "${RED}ERROR: no se generaron transcripciones para $label.${NC}"
        exit 1
    fi


    # --------------------------------------------------------
    # Guardarlas antes de que la siguiente corrida
    # sobrescriba out/
    # --------------------------------------------------------

    cp "${archivos_txt[@]}" "$transcript_output/"


    echo
    echo -e "${GREEN}Transcripciones guardadas:${NC}"
    echo "$transcript_output"
}


# ============================================================
# 1. GENERAR TODAS LAS SALIDAS ASR
# ============================================================

for i in "${!ASR_KEYS[@]}"
do
    generar_asr \
        "${ASR_KEYS[$i]}" \
        "${ASR_LABELS[$i]}" \
        "${ASR_MODELS[$i]}"
done


# ============================================================
# FUNCIÓN:
# Cargar las transcripciones de una condición nuevamente
# en out/, porque translate_pipeline.jl lee directamente out/
# ============================================================

cargar_transcripciones_en_out() {

    local carpeta="$1"

    rm -rf "$OUT_DIR"
    mkdir -p "$OUT_DIR"

    shopt -s nullglob
    archivos_txt=("$carpeta"/*.txt)
    shopt -u nullglob

    if [[ ${#archivos_txt[@]} -eq 0 ]]; then
        echo -e "${RED}ERROR: no existen transcripciones en:${NC}"
        echo "$carpeta"
        exit 1
    fi

    cp "${archivos_txt[@]}" "$OUT_DIR/"
}


# ============================================================
# FUNCIÓN:
# REFERENCIA -> todos los LLM
#
# Usa translate_reference.jl
#
# groundtruth.txt
#       |
#       v
#      LLM
#       |
#       v
# traducción
#       |
#       v
# groundtruth_en.txt
#
# Mide el error/calidad propia del LLM.
# ============================================================

ejecutar_referencia_llms() {

    echo
    echo -e "${YELLOW}------------------------------------------------------------${NC}"
    echo -e "${YELLOW} Referencia -> LLM${NC}"
    echo -e "${YELLOW} usando translate_reference.jl${NC}"
    echo -e "${YELLOW}------------------------------------------------------------${NC}"


    for i in "${!LLM_KEYS[@]}"
    do
        local llm_key="${LLM_KEYS[$i]}"
        local llm_label="${LLM_LABELS[$i]}"
        local llm_model="${LLM_MODELS[$i]}"

        echo
        echo -e "${CYAN}$llm_label${NC}"

        julia "$TRANSLATE_REFERENCE_SCRIPT" \
            "$llm_model" \
            "$TARGET_LANG" \
            2>&1 | tee \
            "$LLM_RESULT_DIR/${llm_key}_referencia.txt"
    done
}


# ============================================================
# FUNCIÓN:
# ASR -> todos los LLM
#
# Usa translate_pipeline.jl
#
# translate_pipeline.jl lee:
#
#   groundtruth.txt
#   out/*.txt
#
# y traduce ambos textos con el mismo LLM.
# ============================================================

ejecutar_asr_llms() {

    local condicion="$1"
    local condicion_key="$2"

    echo
    echo -e "${YELLOW}------------------------------------------------------------${NC}"
    echo -e "${YELLOW} $condicion -> LLM${NC}"
    echo -e "${YELLOW} usando translate_pipeline.jl${NC}"
    echo -e "${YELLOW}------------------------------------------------------------${NC}"


    for i in "${!LLM_KEYS[@]}"
    do
        local llm_key="${LLM_KEYS[$i]}"
        local llm_label="${LLM_LABELS[$i]}"
        local llm_model="${LLM_MODELS[$i]}"

        echo
        echo -e "${CYAN}$llm_label${NC}"

        julia "$TRANSLATE_PIPELINE_SCRIPT" \
            "$llm_model" \
            "$TARGET_LANG" \
            2>&1 | tee \
            "$LLM_RESULT_DIR/${llm_key}_${condicion_key}.txt"
    done
}


# ============================================================
# 2. REFERENCIA -> TODOS LOS LLM
#
# SOLO translate_reference.jl
# ============================================================

echo
echo -e "${CYAN}============================================================${NC}"
echo -e "${CYAN} Referencia -> LLM${NC}"
echo -e "${CYAN}============================================================${NC}"

ejecutar_referencia_llms


# ============================================================
# 3. TODOS LOS ASR -> TODOS LOS LLM
#
# SOLO translate_pipeline.jl
# ============================================================

for i in "${!ASR_KEYS[@]}"
do
    asr_key="${ASR_KEYS[$i]}"
    asr_label="${ASR_LABELS[$i]}"

    echo
    echo -e "${CYAN}============================================================${NC}"
    echo -e "${CYAN} $asr_label -> LLM${NC}"
    echo -e "${CYAN}============================================================${NC}"


    # --------------------------------------------------------
    # translate_pipeline.jl lee siempre desde out/
    # --------------------------------------------------------

    cargar_transcripciones_en_out \
        "$TRANSCRIPT_DIR/$asr_key"


    # --------------------------------------------------------
    # Ejecutar los tres LLM con translate_pipeline.jl
    # --------------------------------------------------------

    ejecutar_asr_llms \
        "$asr_label" \
        "$asr_key"
done


# ============================================================
# FUNCIONES PARA EXTRAER RESULTADOS
# ============================================================

extraer_ultimo_numero() {

    local linea="$1"

    printf '%s\n' "$linea" |
        grep -oE '[-+]?[0-9]+([.,][0-9]+)?([eE][-+]?[0-9]+)?' |
        tail -n 1 |
        tr ',' '.'
}


extraer_chrf() {

    local archivo="$1"
    local linea

    linea="$(
        grep "chrF promedio" "$archivo" |
        tail -n 1 || true
    )"

    if [[ -z "$linea" ]]; then
        echo "N/A"
        return
    fi

    extraer_ultimo_numero "$linea"
}


extraer_wer_pct() {

    local archivo="$1"
    local linea
    local valor

    linea="$(
        grep -E "WER promedio" "$archivo" |
        tail -n 1 || true
    )"

    if [[ -z "$linea" ]]; then
        echo "N/A"
        return
    fi


    # --------------------------------------------------------
    # Si ya viene expresado en porcentaje
    # --------------------------------------------------------

    if [[ "$linea" == *"%"* ]]; then

        valor="$(
            printf '%s\n' "$linea" |
            grep -oE '[0-9]+([.,][0-9]+)?[[:space:]]*%' |
            tail -n 1 |
            tr -d ' %' |
            tr ',' '.'
        )"

        echo "$valor"
        return
    fi


    # --------------------------------------------------------
    # Si viene como fracción: 0.10 -> 10 %
    # --------------------------------------------------------

    valor="$(extraer_ultimo_numero "$linea")"

    awk -v x="$valor" '
        BEGIN {
            if (x >= 0 && x <= 1)
                printf "%.6f", x * 100
            else
                printf "%.6f", x
        }
    '
}


# ============================================================
# 4. RESUMEN
# ============================================================

{
    echo "============================================================"
    echo "R3 - MATRIZ COMPLETA ASR -> LLM"
    echo "============================================================"
    echo


    # --------------------------------------------------------
    # ASR
    # --------------------------------------------------------

    echo "ERROR DE ENTRADA - ASR"
    echo "------------------------------------------------------------"
    echo

    for i in "${!ASR_KEYS[@]}"
    do
        asr_key="${ASR_KEYS[$i]}"
        asr_label="${ASR_LABELS[$i]}"

        echo "$asr_label:"

        grep -E \
            "WER promedio|CER promedio|RTF promedio" \
            "$ASR_RESULT_DIR/${asr_key}.txt" || true

        echo
    done


    # --------------------------------------------------------
    # Traducción
    # --------------------------------------------------------

    echo
    echo "SALIDA - chrF"
    echo "============================================================"


    for i in "${!LLM_KEYS[@]}"
    do
        llm_key="${LLM_KEYS[$i]}"
        llm_label="${LLM_LABELS[$i]}"

        echo
        echo "$llm_label"
        echo "------------------------------------------------------------"


        # Referencia: translate_reference.jl

        echo -n "Referencia -> LLM: "

        grep "chrF promedio" \
            "$LLM_RESULT_DIR/${llm_key}_referencia.txt" || true


        # ASR: translate_pipeline.jl

        for j in "${!ASR_KEYS[@]}"
        do
            asr_key="${ASR_KEYS[$j]}"
            asr_label="${ASR_LABELS[$j]}"

            echo -n "$asr_label -> LLM: "

            grep "chrF promedio" \
                "$LLM_RESULT_DIR/${llm_key}_${asr_key}.txt" || true
        done
    done

} | tee "$RESUMEN"


# ============================================================
# 5. CSV COMPLETO
# ============================================================

echo \
"llm,condicion,wer_pct,chrf" \
> "$TABLA_CSV"


# ------------------------------------------------------------
# Referencia -> LLM
# translate_reference.jl
# ------------------------------------------------------------

for i in "${!LLM_KEYS[@]}"
do
    llm_key="${LLM_KEYS[$i]}"
    llm_label="${LLM_LABELS[$i]}"

    chrf="$(
        extraer_chrf \
            "$LLM_RESULT_DIR/${llm_key}_referencia.txt"
    )"

    printf '"%s","%s",%s,%s\n' \
        "$llm_label" \
        "Referencia -> LLM" \
        "0.0" \
        "$chrf" \
        >> "$TABLA_CSV"
done


# ------------------------------------------------------------
# Todos los ASR -> todos los LLM
# translate_pipeline.jl
# ------------------------------------------------------------

for i in "${!LLM_KEYS[@]}"
do
    llm_key="${LLM_KEYS[$i]}"
    llm_label="${LLM_LABELS[$i]}"

    for j in "${!ASR_KEYS[@]}"
    do
        asr_key="${ASR_KEYS[$j]}"
        asr_label="${ASR_LABELS[$j]}"

        wer="$(
            extraer_wer_pct \
                "$ASR_RESULT_DIR/${asr_key}.txt"
        )"

        chrf="$(
            extraer_chrf \
                "$LLM_RESULT_DIR/${llm_key}_${asr_key}.txt"
        )"

        printf '"%s","%s",%s,%s\n' \
            "$llm_label" \
            "$asr_label -> LLM" \
            "$wer" \
            "$chrf" \
            >> "$TABLA_CSV"
    done
done


# ============================================================
# FINAL
# ============================================================

echo
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN} Experimento finalizado${NC}"
echo -e "${GREEN}============================================================${NC}"
echo

echo "Resultados:"
echo "  $RESULT_DIR"

echo
echo "Resumen:"
echo "  $RESUMEN"

echo
echo "Tabla completa:"
echo "  $TABLA_CSV"

echo