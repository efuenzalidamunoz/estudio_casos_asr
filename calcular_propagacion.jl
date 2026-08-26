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
#
# Si el profesor pide explícitamente desviación
# estándar muestral, cambiar a true.
const CORRECTED = false


# ============================================================
# LEER WER DESDE LAS SALIDAS DE asr_pipeline.jl
# ============================================================

function leer_wer(path::String)

    isfile(path) || error(
        "No existe el archivo ASR: $path"
    )

    valores = Float64[]

    for linea in eachline(path)

        # Ejemplo esperado:
        #
        # test1.wav   0.100   0.023   0.246

        m = match(
            r"^(\S+\.wav)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)",
            strip(linea)
        )

        m === nothing && continue

        wer_fraccion = parse(
            Float64,
            m.captures[2]
        )

        # asr_pipeline.jl entrega WER como fracción:
        #
        # 0.10 -> 10 %
        #
        # Para la propagación trabajaremos en
        # puntos porcentuales.
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
# LEER chrF DESDE translate_reference.jl
# ============================================================

function leer_chrf(path::String)

    isfile(path) || error(
        "No existe el archivo chrF: $path"
    )

    valores = Float64[]

    for linea in eachline(path)

        # Ejemplo:
        #
        # test1.wav   82.43

        m = match(
            r"^(\S+\.wav)\s+([0-9.eE+-]+)\s*$",
            strip(linea)
        )

        m === nothing && continue

        score = parse(
            Float64,
            m.captures[2]
        )

        push!(
            valores,
            score
        )
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
    println("="^72)
    println("PROPAGACIÓN DEL ERROR - ASR -> LLM")
    println("="^72)
    println()


    # --------------------------------------------------------
    # WER POR AUDIO
    # --------------------------------------------------------

    wer_base = leer_wer(
        joinpath(
            ASR_DIR,
            "base.txt"
        )
    )

    wer_q4 = leer_wer(
        joinpath(
            ASR_DIR,
            "base_q4_0.txt"
        )
    )


    n_base = length(wer_base)
    n_q4   = length(wer_q4)

    println("Audios Base encontrados:       $n_base")
    println("Audios Base-Q4_0 encontrados:  $n_q4")
    println()


    # --------------------------------------------------------
    # ESTADÍSTICAS WER
    # --------------------------------------------------------

    media_wer_base = mean(
        wer_base
    )

    media_wer_q4 = mean(
        wer_q4
    )

    sigma_wer_base = sigma(
        wer_base
    )

    sigma_wer_q4 = sigma(
        wer_q4
    )


    println("WER")
    println("-"^72)

    @printf(
        "%-18s media = %8.3f %%   sigma = %8.3f pp\n",
        "Base",
        media_wer_base,
        sigma_wer_base
    )

    @printf(
        "%-18s media = %8.3f %%   sigma = %8.3f pp\n",
        "Base-Q4_0",
        media_wer_q4,
        sigma_wer_q4
    )

    println()


    # --------------------------------------------------------
    # MODELOS OLLAMA
    # --------------------------------------------------------

    modelos = [
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


    resultados = []


    # --------------------------------------------------------
    # CADA LLM
    # --------------------------------------------------------

    for modelo in modelos

        # ----------------------------------------------------
        # chrF POR CONDICIÓN
        # ----------------------------------------------------

        chrf_ref = leer_chrf(
            joinpath(
                LLM_DIR,
                "$(modelo.prefijo)_referencia.txt"
            )
        )

        chrf_base = leer_chrf(
            joinpath(
                LLM_DIR,
                "$(modelo.prefijo)_base.txt"
            )
        )

        chrf_q4 = leer_chrf(
            joinpath(
                LLM_DIR,
                "$(modelo.prefijo)_base_q4_0.txt"
            )
        )


        # ----------------------------------------------------
        # PROMEDIOS
        # ----------------------------------------------------

        media_chrf_ref = mean(
            chrf_ref
        )

        media_chrf_base = mean(
            chrf_base
        )

        media_chrf_q4 = mean(
            chrf_q4
        )


        # ----------------------------------------------------
        # sigma_chrF INTRÍNSECO DEL LLM
        #
        # Usamos Referencia -> LLM porque esta condición
        # no contiene error del ASR.
        # ----------------------------------------------------

        sigma_chrf_base = sigma(
            chrf_ref
        )


        # También calculamos los sigmas observados de las
        # demás condiciones solo como información adicional.

        sigma_chrf_base_asr = sigma(
            chrf_base
        )

        sigma_chrf_q4_asr = sigma(
            chrf_q4
        )


        # ----------------------------------------------------
        # ALPHA
        #
        # Tres puntos experimentales:
        #
        # WER:
        #   0
        #   Base
        #   Base-Q4_0
        #
        # chrF:
        #   Referencia
        #   Base
        #   Base-Q4_0
        # ----------------------------------------------------

        x = Float64[
            0.0,
            media_wer_base,
            media_wer_q4
        ]

        y = Float64[
            media_chrf_ref,
            media_chrf_base,
            media_chrf_q4
        ]


        pendiente, alpha = calcular_alpha(
            x,
            y
        )


        # ----------------------------------------------------
        # PROPAGACIÓN
        # ----------------------------------------------------

        sigma_ref = sigma_sistema(
            alpha,
            0.0,
            sigma_chrf_base
        )

        sigma_sistema_base = sigma_sistema(
            alpha,
            sigma_wer_base,
            sigma_chrf_base
        )

        sigma_sistema_q4 = sigma_sistema(
            alpha,
            sigma_wer_q4,
            sigma_chrf_base
        )


        # ----------------------------------------------------
        # GUARDAR
        # ----------------------------------------------------

        push!(
            resultados,
            (
                modelo=modelo.nombre,

                pendiente=pendiente,
                alpha=alpha,

                media_chrf_ref=media_chrf_ref,
                media_chrf_base=media_chrf_base,
                media_chrf_q4=media_chrf_q4,

                sigma_chrf=sigma_chrf_base,

                sigma_chrf_obs_ref=sigma_chrf_base,
                sigma_chrf_obs_base=sigma_chrf_base_asr,
                sigma_chrf_obs_q4=sigma_chrf_q4_asr,

                sigma_sistema_ref=sigma_ref,
                sigma_sistema_base=sigma_sistema_base,
                sigma_sistema_q4=sigma_sistema_q4
            )
        )
    end


    # ========================================================
    # MOSTRAR RESULTADOS
    # ========================================================

    println("="^72)
    println("RESULTADOS DE PROPAGACIÓN")
    println("="^72)

    for r in resultados

        println()
        println(r.modelo)
        println("-"^72)

        @printf(
            "Pendiente regresión:       %8.4f\n",
            r.pendiente
        )

        @printf(
            "alpha = |pendiente|:       %8.4f\n",
            r.alpha
        )

        @printf(
            "sigma_chrF (LLM aislado):  %8.4f\n",
            r.sigma_chrf
        )

        println()

        @printf(
            "sigma_sistema Referencia:  %8.4f\n",
            r.sigma_sistema_ref
        )

        @printf(
            "sigma_sistema Base:        %8.4f\n",
            r.sigma_sistema_base
        )

        @printf(
            "sigma_sistema Base-Q4_0:   %8.4f\n",
            r.sigma_sistema_q4
        )

        println()
        println(
            "sigma chrF observado por condición:"
        )

        @printf(
            "  Referencia -> LLM:       %8.4f\n",
            r.sigma_chrf_obs_ref
        )

        @printf(
            "  Base -> LLM:             %8.4f\n",
            r.sigma_chrf_obs_base
        )

        @printf(
            "  Base-Q4_0 -> LLM:        %8.4f\n",
            r.sigma_chrf_obs_q4
        )
    end


    # ========================================================
    # GUARDAR TXT
    # ========================================================

    open(
        OUTPUT_TXT,
        "w"
    ) do io

        println(
            io,
            "PROPAGACIÓN DEL ERROR ASR -> LLM"
        )

        println(
            io,
            "="^72
        )

        println(io)

        @printf(
            io,
            "WER Base:      media = %.4f %% | sigma = %.4f pp\n",
            media_wer_base,
            sigma_wer_base
        )

        @printf(
            io,
            "WER Base-Q4_0: media = %.4f %% | sigma = %.4f pp\n",
            media_wer_q4,
            sigma_wer_q4
        )

        println(io)


        for r in resultados

            println(
                io,
                r.modelo
            )

            println(
                io,
                "-"^72
            )

            @printf(
                io,
                "alpha: %.6f\n",
                r.alpha
            )

            @printf(
                io,
                "sigma_chrF: %.6f\n",
                r.sigma_chrf
            )

            @printf(
                io,
                "sigma_sistema Referencia: %.6f\n",
                r.sigma_sistema_ref
            )

            @printf(
                io,
                "sigma_sistema Base: %.6f\n",
                r.sigma_sistema_base
            )

            @printf(
                io,
                "sigma_sistema Base-Q4_0: %.6f\n",
                r.sigma_sistema_q4
            )

            println(io)
        end
    end


    # ========================================================
    # GUARDAR CSV
    # ========================================================

    open(
        OUTPUT_CSV,
        "w"
    ) do io

        println(
            io,
            "llm,condicion,alpha,sigma_wer,sigma_chrf,sigma_sistema"
        )


        for r in resultados

            @printf(
                io,
                "%s,Referencia,%.6f,%.6f,%.6f,%.6f\n",
                r.modelo,
                r.alpha,
                0.0,
                r.sigma_chrf,
                r.sigma_sistema_ref
            )

            @printf(
                io,
                "%s,Base,%.6f,%.6f,%.6f,%.6f\n",
                r.modelo,
                r.alpha,
                sigma_wer_base,
                r.sigma_chrf,
                r.sigma_sistema_base
            )

            @printf(
                io,
                "%s,Base-Q4_0,%.6f,%.6f,%.6f,%.6f\n",
                r.modelo,
                r.alpha,
                sigma_wer_q4,
                r.sigma_chrf,
                r.sigma_sistema_q4
            )
        end
    end


    println()
    println("="^72)
    println("Archivos generados:")
    println()
    println("  $OUTPUT_TXT")
    println("  $OUTPUT_CSV")
    println("="^72)
end


main()