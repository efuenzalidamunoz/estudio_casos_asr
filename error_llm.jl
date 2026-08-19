using Printf

modelos = [
    "gemma3:1b-it-fp16",
    "gemma3:1b-it-q8_0",
    "gemma3:1b-it-q4_K_M"
]

# Funcion para extraer el chrF de la salida
function extraer_chrf(salida_texto)
    lineas = filter(x -> !isempty(strip(x)), split(salida_texto, "\n"))
    for i in length(lineas):-1:1
        linea = lineas[i]
        if occursin(r"\d", linea)
            matchs = collect(eachmatch(r"\d+\.\d+|\d+", linea))
            if !isempty(matchs)
                val = parse(Float64, matchs[1].match)
                if 0.0 <= val <= 100.0
                    return val
                end
            end
        end
    end
    return 0.0
end

archivo_salida = "scripts_out/R3/resultados_error_llm.txt"
mkpath(dirname(archivo_salida))

open(archivo_salida, "w") do f
    write(f, "Modelo,chrF,Error_Traduccion\n")
    for modelo in modelos
        cmd = `julia translate_pipeline.jl $modelo English`
        try
            salida = read(cmd, String)
            chrf = extraer_chrf(salida)
            error_llm = 100.0 - chrf
            @printf(f, "%s,%.2f,%.2f\n", modelo, chrf, error_llm)
        catch e
            # Permite que se detenga el script con ctrl+c
            if isa(e, InterruptException)
                rethrow(e)
            end
            @printf(f, "%s,ERROR,ERROR\n", modelo)
            @warn "Falló la ejecución para el modelo: $modelo" exception=e
        end
    end
end
println("Proceso finalizado.")
