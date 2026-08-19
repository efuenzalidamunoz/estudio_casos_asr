#!/usr/bin/env julia

# Traduce con un LLM servido en Ollama el texto de referencia en español
# (groundtruth.txt) y evalúa la calidad con chrF contra la referencia
# humana en inglés (groundtruth_en.txt).
# Esto permite medir el error del LLM aislado, sin intervención del ASR.

const GROUNDTRUTH_ES = joinpath(@__DIR__, "groundtruth.txt")
const GROUNDTRUTH_EN = joinpath(@__DIR__, "groundtruth_en.txt")
const OLLAMA_URL = "http://localhost:11434/api/generate"


function usage_and_exit()
    println(stderr, "uso: julia translate_reference.jl <modelo_ollama> <idioma_objetivo>")
    println(stderr, "ejemplo: julia translate_reference.jl gemma3:1b-it-q4_K_M English")
    exit(1)
end


# --- Lectura de referencia ---------------------------------------------

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


const JSON_ESCAPES = Dict(
    'n'  => '\n',
    't'  => '\t',
    'r'  => '\r',
    '"'  => '"',
    '\\' => '\\',
    '/'  => '/'
)

function json_extract_string(json::AbstractString, key::String)
    marker = "\"$key\":\""
    range = findfirst(marker, json)

    range === nothing &&
        error("Campo '$key' no encontrado en la respuesta de Ollama: $json")

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


# --- Traducción vía Ollama --------------------------------------------

function translate(text::AbstractString, model::String, target_lang::String)
    prompt = "Translate the following text into $target_lang. Output ONLY " *
             "the translated text, with no explanations, quotation marks, " *
             "or additional commentary.\n\nText: $text"

    payload = """{"model":"$(json_escape(model))","prompt":"$(json_escape(prompt))",""" *
              """"stream":false,"think":false,"options":{"temperature":0,"seed":67}}"""

    cmd = Cmd([
        "curl", "-s", "-X", "POST", OLLAMA_URL,
        "-H", "Content-Type: application/json",
        "--data-binary", "@-"
    ])

    out = IOBuffer()
    run(pipeline(cmd; stdin=IOBuffer(payload), stdout=out))
    response = String(take!(out))

    return strip(json_extract_string(response, "response"))
end


# --- Métrica chrF -----------------------------------------------------
# Mismo código utilizado por el profesor

function char_ngrams(s::AbstractString, n::Int)
    chars = collect(s)
    counts = Dict{String,Int}()

    for i in 1:(length(chars) - n + 1)
        g = String(chars[i:i+n-1])
        counts[g] = get(counts, g, 0) + 1
    end

    return counts
end


function chrf(reference::AbstractString, hypothesis::AbstractString; max_n::Int=6, beta::Float64=2.0)
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


# --- Main -------------------------------------------------------------

function main()
    length(ARGS) == 2 || usage_and_exit()

    model, target_lang = ARGS[1], ARGS[2]

    # Referencias en español e inglés
    refs_es = read_groundtruth(GROUNDTRUTH_ES)
    refs_en = read_groundtruth(GROUNDTRUTH_EN)

    println(rpad("archivo", 12), rpad("chrF", 10))
    println("-"^22)

    scores = Float64[]

    for wav_name in sort(collect(keys(refs_es)))

        # Texto correcto en español
        reference_es = refs_es[wav_name]

        # Traducción humana correcta
        reference_en = get(refs_en, wav_name, nothing)

        if reference_en === nothing
            @warn "Sin referencia en groundtruth_en.txt" wav_name
            continue
        end

        # Traducir SOLO la referencia española
        llm_translation = translate(reference_es, model, target_lang)

        # Comparar traducción humana vs traducción del LLM
        score = chrf(reference_en, llm_translation)
        push!(scores, score)

        # Misma idea de salida del código original
        println(rpad(wav_name, 12), rpad(round(score; digits=2), 10))
        println("  ref (es):          ", reference_es)
        println("  ref ($target_lang): ", reference_en)
        println("  LLM -> $target_lang: ", llm_translation)
        println()
    end

    println("-"^22)
    println("chrF promedio: ", round(sum(scores) / length(scores); digits=2))
end


main()
