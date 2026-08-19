#!/usr/bin/env julia
#
const R2_DIR = joinpath(@__DIR__, "scripts_out", "R2")
const OUTPUT_DIR = joinpath(@__DIR__, "scripts_out", "R3")
const OUTPUT_FILE = joinpath(OUTPUT_DIR, "error_cascada.txt")
const BASE_FILE = joinpath(R2_DIR,"salida_ggml-base.bin.txt")
const Q4_FILE = joinpath(R2_DIR, "salida_ggml-base-q4_0.bin.txt")


# Extre el valor numerico de una linea
function extraer_valor(path::String, etiqueta::String)
    isfile(path) ||
        error("No existe el archivo: $path")
    for line in eachline(path)
        if occursin(etiqueta, line)
            partes = split(line, ":")
            length(partes) < 2 && continue
            valor_texto = strip(partes[end])
            valor_texto =
                replace(valor_texto, "," => ".")
            return parse(Float64, valor_texto)
        end
    end
    error("No se encontró '$etiqueta' en $path")
end

# Lee los resultados del modelo
function leer_modelo(path::String)
    cer = extraer_valor(path,"CER promedio")
    chrf = extraer_valor(path,"chrF promedio")
    return cer, chrf
end


function error_desde_chrf(chrf::Float64)
    return 1.0 - chrf / 100.0
end


function kappa_empirico(error_entrada::Float64, error_salida::Float64)
    if error_entrada == 0.0
        return NaN
    end
    return error_salida / error_entrada
end

function main()
    # Modelo de ASR base
    cer_base, chrf_base = leer_modelo(BASE_FILE)

    # Modelo de ASR base con cuantizacion (Q4)
    cer_q4, chrf_q4 =
        leer_modelo(Q4_FILE)

    error_salida_base = error_desde_chrf(chrf_base)
    error_salida_q4 = error_desde_chrf(chrf_q4)

    kappa_base = kappa_empirico(cer_base, error_salida_base)
    kappa_q4 = kappa_empirico(cer_q4, error_salida_q4)

    mkpath(OUTPUT_DIR)
    open(OUTPUT_FILE, "w") do io
        println(io, "Condicion experimental,Error entrada,Error salida,κ empirico")
        println(io, "Referencia -> LLM,0.0000,0.0000,N/A")
        println(io,"ASR alta precisión -> LLM,", round(cer_base; digits=4),",", round(error_salida_base; digits=4),",",round(kappa_base; digits=4))
        println( io,"ASR cuantizado -> LLM,",round(cer_q4; digits=4),",",round(error_salida_q4; digits=4),",",round(kappa_q4; digits=4))
        println(io)
        println(io, "Datos utilizados")
        println(io,"Modelo,CER promedio,chrF promedio,Error salida,Kappa")
        println(io,"ggml-base.bin,",round(cer_base; digits=4),",",round(chrf_base; digits=2),",",round(error_salida_base; digits=4),",",round(kappa_base; digits=4))
        println(io,"ggml-base-q4_0.bin,",round(cer_q4; digits=4),",",round(chrf_q4; digits=2),",",round(error_salida_q4; digits=4),",",round(kappa_q4; digits=4))
    end
    println()
    println("Resultado guardado en:")
    println(OUTPUT_FILE)
    println()
end

main()
