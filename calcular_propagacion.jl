#!/usr/bin/env julia

using Statistics
using Printf


# ============================================================
# RUTAS
# ============================================================

ROOT = normpath(joinpath(@__DIR__))

RESULT_DIR = joinpath(
    ROOT,
    "scripts_out",
    "R3",
    "propagacion"
)

ASR_DIR = joinpath(
    RESULT_DIR,
    "asr"
)

LLM_DIR = joinpath(
    RESULT_DIR,
    "ollama"
)

OUTPUT_TXT = joinpath(
    RESULT_DIR,
    "propagacion_gauss.txt"
)

OUTPUT_CSV = joinpath(
    RESULT_DIR,
    "propagacion_gauss.csv"
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

# false = desviación estándar poblacional
# sobre los 10 audios del corpus.
const CORRECTED = false


# ============================================================
# CONDICIONES ASR
# ============================================================

condiciones_asr = [
    (
        key = "base",
        nombre = "Base",
        archivo = "base.txt"
    ),
    (
        key = "tiny",
        nombre = "Tiny",
        archivo = "tiny.txt"
    ),
    (
        key = "base_q8_0",
        nombre = "Base-Q8_0",
        archivo = "base_q8_0.txt"
    ),
    (
        key = "base_q5_1",
        nombre = "Base-Q5_1",
        archivo = "base_q5_1.txt"
    ),
    (
        key = "base_q4_0",
        nombre = "Base-Q4_0",
        archivo = "base_q4_0.txt"
    )
]


# ============================================================
# MODELOS OLLAMA
# ============================================================

modelos_llm = [
    (
        nombre = "Gemma3:1B FP16",
        prefijo = "fp16"
    ),
    (
        nombre = "Gemma3:1B Q8_0",
        prefijo = "q8_0"
    ),
    (
        nombre = "Gemma3:1B Q4_K_M",
        prefijo = "q4_k_m"
    )
]


# ============================================================
# LEER WER DESDE asr_pipeline.jl
# ============================================================

function leer_wer(path::String)

    isfile(path) || error(
        "No existe el archivo ASR: $path"
    )

    valores = Float64[]

    for linea in eachline(path)

        m = match(
            r"^(\S+\.wav)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)",
            strip(linea)
        )

        m === nothing && continue

        wer_fraccion = parse(
            Float64,
            m.captures[2]
        )

        # WER como fracción -> puntos porcentuales
        push!(
            valores,
            wer_fraccion * 100
        )
    end

    isempty(valores) && error(
        "No se encontraron valores WER en: $path"
    )

    return valores
end


# ============================================================
# LEER chrF DESDE LOS RESULTADOS DE TRADUCCIÓN
# ============================================================

function leer_chrf(path::String)

    isfile(path) || error(
        "No existe el archivo chrF: $path"
    )

    valores = Float64[]

    for linea in eachline(path)

        # Acepta testN.wav y testN.txt.
        m = match(
            r"^(\S+\.(?:wav|txt))\s+([0-9.eE+-]+)\s*$"i,
            strip(linea)
        )

        m === nothing && continue

        score = parse(
            Float64,
            m.captures[2]
        )

        push!(valores, score)
    end

    isempty(valores) && error(
        "No se encontraron valores chrF en: $path"
    )

    return valores
end

# ============================================================
# DESVIACIÓN ESTÁNDAR
# ============================================================

function sigma(x)

    return std(
        x;
        corrected=CORRECTED
    )
end


# ============================================================
# REGRESIÓN LINEAL
#
# chrF = beta0 + beta1 * WER
#
# alpha = |beta1|
# ============================================================

function calcular_alpha(
    wer::Vector{Float64},
    chrf::Vector{Float64}
)

    length(wer) == length(chrf) ||
        error("WER y chrF deben tener igual tamaño")

    x̄ = mean(wer)
    ȳ = mean(chrf)

    numerador = sum(
        (wer .- x̄) .* (chrf .- ȳ)
    )

    denominador = sum(
        (wer .- x̄).^2
    )

    denominador == 0 &&
        error("No es posible calcular la pendiente")

    pendiente = numerador / denominador
    alpha = abs(pendiente)

    return pendiente, alpha
end


# ============================================================
# PROPAGACIÓN DE GAUSS
# ============================================================

function sigma_sistema(
    alpha,
    sigma_wer,
    sigma_chrf
)

    return sqrt(
        (alpha * sigma_wer)^2
        +
        sigma_chrf^2
    )
end


# ============================================================
# MAIN
# ============================================================

function main()

    println()
    println("="^80)
    println("PROPAGACIÓN DEL ERROR - MATRIZ COMPLETA ASR -> LLM")
    println("="^80)
    println()


    # ========================================================
    # 1. LEER WER DE TODOS LOS MODELOS ASR
    # ========================================================

    datos_asr = Dict{String, Any}()

    println("WER POR MODELO")
    println("-"^80)

    for condicion in condiciones_asr

        wer = leer_wer(
            joinpath(
                ASR_DIR,
                condicion.archivo
            )
        )

        media_wer = mean(wer)
        sigma_wer = sigma(wer)

        datos_asr[condicion.key] = (
            nombre = condicion.nombre,
            wer = wer,
            media_wer = media_wer,
            sigma_wer = sigma_wer
        )

        @printf(
            "%-18s media = %8.3f %%   sigma = %8.3f pp   n = %d\n",
            condicion.nombre,
            media_wer,
            sigma_wer,
            length(wer)
        )
    end

    println()


    # ========================================================
    # 2. RESULTADOS POR LLM
    # ========================================================

    resultados = []


    for modelo in modelos_llm

        # ----------------------------------------------------
        # REFERENCIA -> LLM
        # ----------------------------------------------------

        chrf_ref = leer_chrf(
            joinpath(
                LLM_DIR,
                "$(modelo.prefijo)_referencia.txt"
            )
        )

        media_chrf_ref = mean(chrf_ref)

        # Variabilidad intrínseca del LLM:
        # Referencia -> LLM no contiene error ASR.
        sigma_chrf_intrinseco = sigma(chrf_ref)


        # ----------------------------------------------------
        # LEER chrF DE TODAS LAS CONDICIONES ASR
        # ----------------------------------------------------

        datos_chrf = Dict{String, Any}()

        for condicion in condiciones_asr

            chrf = leer_chrf(
                joinpath(
                    LLM_DIR,
                    "$(modelo.prefijo)_$(condicion.key).txt"
                )
            )

            datos_chrf[condicion.key] = (
                valores = chrf,
                media = mean(chrf),
                sigma_observado = sigma(chrf)
            )
        end


        # ----------------------------------------------------
        # 3. CALCULAR ALPHA MEDIANTE REGRESIÓN
        #
        # Puntos utilizados:
        #
        # Referencia      WER = 0
        # Base
        # Tiny
        # Base-Q8_0
        # Base-Q5_1
        # Base-Q4_0
        # ----------------------------------------------------

        x = Float64[0.0]
        y = Float64[media_chrf_ref]

        for condicion in condiciones_asr

            push!(
                x,
                datos_asr[condicion.key].media_wer
            )

            push!(
                y,
                datos_chrf[condicion.key].media
            )
        end


        pendiente, alpha = calcular_alpha(
            x,
            y
        )


        # ----------------------------------------------------
        # REFERENCIA
        # ----------------------------------------------------

        sigma_ref = sigma_sistema(
            alpha,
            0.0,
            sigma_chrf_intrinseco
        )

        push!(
            resultados,
            (
                modelo = modelo.nombre,
                condicion = "Referencia",
                pendiente = pendiente,
                alpha = alpha,

                media_wer = 0.0,
                sigma_wer = 0.0,

                media_chrf = media_chrf_ref,
                sigma_chrf = sigma_chrf_intrinseco,
                sigma_chrf_observado = sigma_chrf_intrinseco,

                sigma_sistema = sigma_ref
            )
        )


        # ----------------------------------------------------
        # TODAS LAS CONDICIONES ASR
        # ----------------------------------------------------

        for condicion in condiciones_asr

            asr = datos_asr[condicion.key]
            chrf = datos_chrf[condicion.key]

            sigma_total = sigma_sistema(
                alpha,
                asr.sigma_wer,
                sigma_chrf_intrinseco
            )

            push!(
                resultados,
                (
                    modelo = modelo.nombre,
                    condicion = condicion.nombre,
                    pendiente = pendiente,
                    alpha = alpha,

                    media_wer = asr.media_wer,
                    sigma_wer = asr.sigma_wer,

                    media_chrf = chrf.media,
                    sigma_chrf = sigma_chrf_intrinseco,
                    sigma_chrf_observado = chrf.sigma_observado,

                    sigma_sistema = sigma_total
                )
            )
        end
    end


    # ========================================================
    # 4. MOSTRAR RESULTADOS
    # ========================================================

    println()
    println("="^80)
    println("RESULTADOS DE PROPAGACIÓN")
    println("="^80)


    for modelo in modelos_llm

        filas = filter(
            r -> r.modelo == modelo.nombre,
            resultados
        )

        primera = first(filas)

        println()
        println(modelo.nombre)
        println("-"^80)

        @printf(
            "Pendiente regresión:      %8.4f\n",
            primera.pendiente
        )

        @printf(
            "alpha = |pendiente|:      %8.4f chrF/pp WER\n",
            primera.alpha
        )

        @printf(
            "sigma_chrF intrínseco:    %8.4f puntos chrF\n",
            primera.sigma_chrf
        )

        println()

        @printf(
            "%-18s %10s %10s %10s %12s\n",
            "Condición",
            "WER %",
            "σ_WER",
            "chrF",
            "σ_sistema"
        )

        println("-"^80)

        for r in filas

            @printf(
                "%-18s %10.3f %10.3f %10.3f %12.3f\n",
                r.condicion,
                r.media_wer,
                r.sigma_wer,
                r.media_chrf,
                r.sigma_sistema
            )
        end
    end


    # ========================================================
    # 5. GUARDAR TXT
    # ========================================================

    open(
        OUTPUT_TXT,
        "w"
    ) do io

        println(
            io,
            "PROPAGACIÓN DEL ERROR - MATRIZ COMPLETA ASR -> LLM"
        )

        println(
            io,
            "="^80
        )

        println(io)


        println(
            io,
            "WER POR MODELO ASR"
        )

        println(
            io,
            "-"^80
        )

        for condicion in condiciones_asr

            asr = datos_asr[condicion.key]

            @printf(
                io,
                "%-18s media = %.4f %% | sigma = %.4f pp\n",
                condicion.nombre,
                asr.media_wer,
                asr.sigma_wer
            )
        end


        for modelo in modelos_llm

            filas = filter(
                r -> r.modelo == modelo.nombre,
                resultados
            )

            primera = first(filas)

            println(io)
            println(io)

            println(
                io,
                modelo.nombre
            )

            println(
                io,
                "-"^80
            )

            @printf(
                io,
                "pendiente: %.6f\n",
                primera.pendiente
            )

            @printf(
                io,
                "alpha: %.6f\n",
                primera.alpha
            )

            @printf(
                io,
                "sigma_chrF intrinseco: %.6f\n",
                primera.sigma_chrf
            )

            println(io)

            for r in filas

                @printf(
                    io,
                    "%-18s | WER = %8.4f %% | sigma_WER = %8.4f pp | chrF = %8.4f | sigma_sistema = %8.4f\n",
                    r.condicion,
                    r.media_wer,
                    r.sigma_wer,
                    r.media_chrf,
                    r.sigma_sistema
                )
            end
        end
    end


    # ========================================================
    # 6. GUARDAR CSV
    # ========================================================

    open(
        OUTPUT_CSV,
        "w"
    ) do io

        println(
            io,
            "llm,condicion,media_wer,sigma_wer,media_chrf,alpha,sigma_chrf,sigma_chrf_observado,sigma_sistema"
        )


        for r in resultados

            @printf(
                io,
                "%s,%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                r.modelo,
                r.condicion,
                r.media_wer,
                r.sigma_wer,
                r.media_chrf,
                r.alpha,
                r.sigma_chrf,
                r.sigma_chrf_observado,
                r.sigma_sistema
            )
        end
    end


    println()
    println()
    println("="^80)
    println("Archivos generados:")
    println()
    println("  $OUTPUT_TXT")
    println("  $OUTPUT_CSV")
    println("="^80)
end


main()
