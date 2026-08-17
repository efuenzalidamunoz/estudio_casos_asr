#!/usr/bin/env julia
# Pipeline ASR: transcribe con whisper-cli, lee resultados y calcula métricas WER/CER.

const WHISPER_CLI = joinpath(@__DIR__, "whisper.cpp", "build", "bin", "whisper-cli")
const LANGUAGE    = "es"
const AUDIO_DIR   = joinpath(@__DIR__, "audio")
const WAV_FILES   = ["test1.wav", "test2.wav", "test3.wav","test4.wav","test5.wav","test6.wav","test7.wav","test8.wav","test9.wav","test10.wav"]
const GROUNDTRUTH = joinpath(@__DIR__, "groundtruth.txt")
const OUT_DIR     = joinpath(@__DIR__, "out")

function usage_and_exit()
    println(stderr, "uso: julia asr_pipeline.jl <ruta-al-modelo.bin>")
    println(stderr, "ejemplo: julia asr_pipeline.jl whisper.cpp/models/ggml-base.bin")
    exit(1)
end

# --- Duración de audio (lee el header WAV, sin dependencias externas) -----

function wav_duration_seconds(path::String)
    open(path, "r") do io
        read(io, 12) # "RIFF" + tamaño + "WAVE"
        byte_rate = 0
        data_size = 0
        while !eof(io)
            chunk_id = String(read(io, 4))
            length(chunk_id) < 4 && break
            chunk_size = Int(reinterpret(UInt32, read(io, 4))[1])
            if chunk_id == "fmt "
                fmt_data = read(io, chunk_size)
                byte_rate = Int(reinterpret(UInt32, fmt_data[9:12])[1])
            elseif chunk_id == "data"
                data_size = chunk_size
                skip(io, chunk_size)
            else
                skip(io, chunk_size)
            end
            isodd(chunk_size) && skip(io, 1)
        end
        byte_rate == 0 && error("No se pudo leer el chunk 'fmt ' de $path")
        return data_size / byte_rate
    end
end

# --- Transcripción -----------------------------------------------------

function transcribe(wav_file::String, model::String)
    base = splitext(basename(wav_file))[1]
    out_prefix = joinpath(OUT_DIR, base)
    cmd = `$WHISPER_CLI -m $model -f $wav_file -l $LANGUAGE -nt -np -otxt -of $out_prefix -t 4` # usar 4 hilos
    elapsed = @elapsed run(pipeline(cmd; stdout=devnull, stderr=devnull))

    hypothesis = strip(read(out_prefix * ".txt", String))
    rtf = elapsed / wav_duration_seconds(wav_file)

    return hypothesis, rtf
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

# --- Normalización y métricas ----------------------------------------------

function normalize(text::AbstractString)
    t = lowercase(text)
    t = replace(t, r"[^\p{L}\p{N}\s]" => "")
    t = replace(t, r"\s+" => " ")
    return strip(t)
end

# Distancia de Levenshtein genérica sobre vectores (palabras o caracteres)
function levenshtein(a::AbstractVector, b::AbstractVector)
    n, m = length(a), length(b)
    prev = collect(0:m)
    curr = zeros(Int, m + 1)
    for i in 1:n
        curr[1] = i
        for j in 1:m
            cost = a[i] == b[j] ? 0 : 1
            curr[j+1] = min(prev[j+1] + 1, curr[j] + 1, prev[j] + cost)
        end
        prev, curr = curr, prev
    end
    return prev[m+1]
end

function wer(reference::AbstractString, hypothesis::AbstractString)
    ref_words = split(normalize(reference))
    hyp_words = split(normalize(hypothesis))
    isempty(ref_words) && return hyp_words == ref_words ? 0.0 : Inf
    return levenshtein(ref_words, hyp_words) / length(ref_words)
end

function cer(reference::AbstractString, hypothesis::AbstractString)
    ref_chars = collect(normalize(reference))
    hyp_chars = collect(normalize(hypothesis))
    isempty(ref_chars) && return hyp_chars == ref_chars ? 0.0 : Inf
    return levenshtein(ref_chars, hyp_chars) / length(ref_chars)
end

# --- Main --------------------------------------------------------------

function main()
    length(ARGS) == 1 || usage_and_exit()
    model = ARGS[1]
    isfile(model) || (println(stderr, "modelo no encontrado: $model"); exit(1))

    isdir(OUT_DIR) || mkdir(OUT_DIR)
    refs = read_groundtruth(GROUNDTRUTH)

    println(rpad("archivo", 12), rpad("WER", 10), rpad("CER", 10), rpad("RTF", 10))
    println("-"^42)

    total_wer = Float64[]
    total_cer = Float64[]
    total_rtf = Float64[]

    for wav_file in WAV_FILES
        reference = get(refs, wav_file, nothing)
        if reference === nothing
            @warn "Sin referencia en groundtruth.txt" wav_file
            continue
        end

        hypothesis, rtf = transcribe(joinpath(AUDIO_DIR, wav_file), model)
        w = wer(reference, hypothesis)
        c = cer(reference, hypothesis)
        push!(total_wer, w)
        push!(total_cer, c)
        push!(total_rtf, rtf)

        println(rpad(wav_file, 12), rpad(round(w; digits=3), 10), rpad(round(c; digits=3), 10), rpad(round(rtf; digits=3), 10))
        println("  ref: ", reference)
        println("  hyp: ", hypothesis)
    end

    println("-"^42)
    println("WER promedio: ", round(sum(total_wer) / length(total_wer); digits=3))
    println("CER promedio: ", round(sum(total_cer) / length(total_cer); digits=3))
    println("RTF promedio: ", round(sum(total_rtf) / length(total_rtf); digits=3))
end

main()
