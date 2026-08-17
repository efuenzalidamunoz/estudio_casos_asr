#!/usr/bin/env julia
# Traduce con un LLM servido en Ollama las transcripciones de out/ y evalúa la
# calidad con chrF, comparando contra la traducción del texto de referencia
# (groundtruth.txt) generada por el mismo modelo. Esto aísla el error introducido
# por el ASR: ambas traducciones pasan por el mismo LLM, la única diferencia es
# si el texto de entrada es la transcripción de referencia o la salida de whisper-cli.

const OUT_DIR     = joinpath(@__DIR__, "out")
const GROUNDTRUTH = joinpath(@__DIR__, "groundtruth.txt")
const RESULTS_DIR = joinpath(@__DIR__, "translations")
const OLLAMA_URL  = "http://localhost:11434/api/generate"

function usage_and_exit()
    println(stderr, "uso: julia translate_pipeline.jl <modelo_ollama> <idioma_objetivo>")
    println(stderr, "ejemplo: julia translate_pipeline.jl gemma4:12b-mlx English")
    exit(1)
end

# --- Lectura de referencia (groundtruth.txt: filename,"transcripcion") ----

function read_groundtruth(path::String)
    refs = Dict{String,String}()
    for line in eachline(path)
        isempty(strip(line)) && continue
        m = match(r"^([^,]+),\"(.*)\"$", line)
        m === nothing && continue
        refs[String(m.captures[1])] = String(m.captures[2])
    end
    return refs
end

# --- JSON mínimo (sin dependencias externas) --------------------------

function json_escape(s::AbstractString)
    io = IOBuffer()
    for c in s
        if c == '"'
            print(io, "\\\"")
        elseif c == '\\'
            print(io, "\\\\")
        elseif c == '\n'
            print(io, "\\n")
        elseif c == '\r'
            print(io, "\\r")
        elseif c == '\t'
            print(io, "\\t")
        else
            print(io, c)
        end
    end
    return String(take!(io))
end

const JSON_ESCAPES = Dict('n' => '\n', 't' => '\t', 'r' => '\r',
                          '"' => '"', '\\' => '\\', '/' => '/')

function json_extract_string(json::AbstractString, key::String)
    marker = "\"$key\":\""
    range = findfirst(marker, json)
    range === nothing && error("Campo '$key' no encontrado en la respuesta de Ollama: $json")
    i = nextind(json, last(range))
    io = IOBuffer()
    while true
        c = json[i]
        if c == '"'
            break
        elseif c == '\\'
            i = nextind(json, i)
            e = json[i]
            if e == 'u'
                hex = json[nextind(json, i):nextind(json, i, 4)]
                print(io, Char(parse(UInt32, hex; base=16)))
                i = nextind(json, i, 4)
            else
                print(io, get(JSON_ESCAPES, e, e))
            end
        else
            print(io, c)
        end
        i = nextind(json, i)
    end
    return String(take!(io))
end

# --- Traducción vía la API REST de Ollama (/api/generate) -----------------

function translate(text::AbstractString, model::String, target_lang::String)
    prompt = "Translate the following text into $target_lang. Output ONLY " *
              "the translated text, with no explanations, quotation marks, " *
              "or additional commentary.\n\nText: $text"

    payload = """{"model":"$(json_escape(model))","prompt":"$(json_escape(prompt))",""" *
              """"stream":false,"think":false,"options":{"temperature":0}}"""

    cmd = Cmd(["curl", "-s", "-X", "POST", OLLAMA_URL,
               "-H", "Content-Type: application/json",
               "--data-binary", "@-"])

    out = IOBuffer()
    run(pipeline(cmd; stdin=IOBuffer(payload), stdout=out))
    response = String(take!(out))
    return strip(json_extract_string(response, "response"))
end

# --- Métrica chrF (Popović, 2015): F-score de n-gramas de caracteres ------
# Se usa chrF en lugar de BLEU porque es más estable en oraciones cortas y
# aisladas (nuestro corpus tiene una frase por archivo); BLEU sentence-level
# sin suavizado tiende a colapsar a 0 con muestras tan pequeñas.

function char_ngrams(s::AbstractString, n::Int)
    chars = collect(s)
    counts = Dict{String,Int}()
    for i in 1:(length(chars) - n + 1)
        g = String(chars[i:i+n-1])
        counts[g] = get(counts, g, 0) + 1
    end
    return counts
end

function chrf(reference::AbstractString, hypothesis::AbstractString;
              max_n::Int=6, beta::Float64=2.0)
    precisions = Float64[]
    recalls = Float64[]
    for n in 1:max_n
        ref_ng = char_ngrams(reference, n)
        hyp_ng = char_ngrams(hypothesis, n)
        ref_total = sum(values(ref_ng); init=0)
        hyp_total = sum(values(hyp_ng); init=0)
        (ref_total == 0 || hyp_total == 0) && continue

        matched = sum(min(c, get(ref_ng, g, 0)) for (g, c) in hyp_ng; init=0)
        push!(precisions, matched / hyp_total)
        push!(recalls, matched / ref_total)
    end
    isempty(precisions) && return 0.0

    p = sum(precisions) / length(precisions)
    r = sum(recalls) / length(recalls)
    (p + r) == 0 && return 0.0
    return 100 * (1 + beta^2) * p * r / (beta^2 * p + r)
end

# --- Main --------------------------------------------------------------

function main()
    length(ARGS) == 2 || usage_and_exit()
    model, target_lang = ARGS[1], ARGS[2]

    isdir(RESULTS_DIR) || mkdir(RESULTS_DIR)
    refs = read_groundtruth(GROUNDTRUTH)
    txt_files = sort(filter(f -> endswith(f, ".txt"), readdir(OUT_DIR)))

    println(rpad("archivo", 12), rpad("chrF", 10))
    println("-"^22)

    scores = Float64[]

    for txt_file in txt_files
        base = splitext(txt_file)[1]
        wav_name = base * ".wav"
        reference_es = get(refs, wav_name, nothing)
        if reference_es === nothing
            @warn "Sin referencia en groundtruth.txt" wav_name
            continue
        end
        hypothesis_es = strip(read(joinpath(OUT_DIR, txt_file), String))

        ref_translation = translate(reference_es, model, target_lang)
        hyp_translation = translate(hypothesis_es, model, target_lang)

        write(joinpath(RESULTS_DIR, base * "_ref.txt"), ref_translation)
        write(joinpath(RESULTS_DIR, base * "_hyp.txt"), hyp_translation)

        score = chrf(ref_translation, hyp_translation)
        push!(scores, score)

        println(rpad(txt_file, 12), rpad(round(score; digits=2), 10))
        println("  ref (es):          ", reference_es)
        println("  hyp (es, ASR):     ", hypothesis_es)
        println("  ref -> $target_lang: ", ref_translation)
        println("  hyp -> $target_lang: ", hyp_translation)
        println()
    end

    println("-"^22)
    println("chrF promedio: ", round(sum(scores) / length(scores); digits=2))
end

main()
