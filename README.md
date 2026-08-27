# Estudio de Casos — Computación Numérica

Análisis del impacto de la cuantización numérica en un pipeline de reconocimiento automático de habla (ASR) y su propagación hacia un modelo de lenguaje (LLM) para traducción.

**Asignatura:** Computación Numérica — 6° Semestre  
**Universidad:** Universidad Católica del Maule

---

## Descripción

Este proyecto implementa un pipeline completo de evaluación que conecta dos etapas:

1. **ASR (Automatic Speech Recognition):** Transcripción de audio en español usando [whisper.cpp](https://github.com/ggerganov/whisper.cpp) con modelos en distintos formatos de cuantización.
2. **Traducción con LLM:** Traducción de las transcripciones al inglés usando [Ollama](https://ollama.com/) con el modelo Gemma 3 1B en distintas precisiones numéricas.

El objetivo central es cuantificar cómo la reducción de precisión numérica (cuantización) en los pesos del modelo ASR afecta la calidad de la transcripción (medida con WER/CER) y cómo ese error se **propaga** a través del LLM de traducción (medido con chrF), aplicando el modelo de **propagación de errores de Gauss**.

---

## Estructura del Proyecto

```
estudio_casos_asr/
│
├── asr_pipeline.jl            # Pipeline ASR (CPU, sin GPU)
├── asr_pipeline_igpu.jl       # Pipeline ASR (iGPU Intel via Vulkan)
├── translate_pipeline.jl      # Traducción ASR → LLM y evaluación chrF
├── translate_reference.jl     # Traducción referencia → LLM (error intrínseco del LLM)
├── calcular_d.py              # Extracción del factor de escala d de modelos GGML
├── calcular_propagacion.jl    # Propagación del error de Gauss (ASR → LLM)
├── grafico.py                 # Gráfico WER vs. formato numérico
│
├── groundtruth.txt            # Transcripciones de referencia (español)
├── groundtruth_en.txt         # Traducciones de referencia (inglés)
│
├── audio/                     # 10 archivos WAV del corpus (16 kHz, mono)
│   ├── test1.wav
│   └── ...test10.wav
│
├── out/                       # Salida de whisper-cli (transcripciones .txt)
├── translations/              # Traducciones generadas por el LLM
│
├── scripts/                   # Scripts de automatización (bash)
│   ├── r1_a.sh                # R1: Cuantización de modelos
│   ├── calcular_bits_efectivos.sh  # R1: Cálculo de bits efectivos por peso
│   ├── r2_b.sh                # R2: Pipeline completo ASR + traducción
│   ├── medir_ram.sh           # R2: Medición de consumo de memoria RAM
│   ├── r3_a.sh                # R3: Traducción con distintos LLM
│   ├── r3_b.sh                # R3: Comparación referencia vs. ASR cuantizado
│   └── propagacion_error.sh   # R3: Matriz completa ASR × LLM (18 combinaciones)
│
├── scripts_out/               # Resultados de los experimentos
│   ├── R1/                    # Cuantización y análisis de factores de escala
│   ├── R2/                    # Precisión (WER/CER), RAM, gráficos
│   └── R3/                    # Propagación del error ASR → LLM
│
├── whisper.cpp/               # Dependencia externa (no incluida, ver instalación)
└── estudio_casos_unidad1-2.pdf  # Enunciado del caso de estudio
```

---

## Modelos Evaluados

### ASR (Whisper)

| Formato | Bits efectivos/peso | Descripción |
|---------|--------------------:|-------------|
| **Base** | 32.00 | Modelo original FP32 (74 M parámetros) |
| **Tiny** | 16.00 | Modelo más pequeño FP16 (39 M parámetros) |
| **Base-Q8_0** | 8.50 | Base cuantizado a 8 bits + escala FP16 |
| **Base-Q5_1** | 6.00 | Base cuantizado a 5 bits + escala y mín. FP16 |
| **Base-Q4_0** | 4.50 | Base cuantizado a 4 bits + escala FP16 |

### LLM (Gemma 3 1B vía Ollama)

| Variante | Precisión |
|----------|-----------|
| `gemma3:1b-it-fp16` | FP16 |
| `gemma3:1b-it-q8_0` | Q8_0 |
| `gemma3:1b-it-q4_K_M` | Q4_K_M |

---

## Metodología

### Corpus

10 frases en español del dominio clínico-hospitalario, grabadas como audio WAV (16 kHz, mono, ~10 s cada una). Cada frase tiene una transcripción de referencia (`groundtruth.txt`) y una traducción humana al inglés (`groundtruth_en.txt`).

### Métricas

| Métrica | Qué mide | Definición |
|---------|----------|------------|
| **WER** | Error de transcripción | Distancia de Levenshtein a nivel de palabras / total de palabras de referencia |
| **CER** | Error de transcripción | Distancia de Levenshtein a nivel de caracteres / total de caracteres de referencia |
| **RTF** | Velocidad de inferencia | Tiempo de procesamiento / duración del audio |
| **chrF** | Calidad de traducción | F-score de n-gramas de caracteres (Popović, 2015), β = 2 |

### Pipeline de Evaluación

```
Audio WAV ──→ whisper.cpp (ASR) ──→ Transcripción ──→ Ollama (LLM) ──→ Traducción
                  │                      │                                   │
                  │                 WER / CER                              chrF
                  │                                                          │
                  └──── Cuantización Q8_0 / Q5_1 / Q4_0 ────────────────────┘
                                                                    Propagación de Gauss
```

### Propagación del Error

Se utiliza la **propagación de errores de Gauss** para modelar cómo la incertidumbre del ASR (σ_WER) se transmite al sistema completo:

```
σ_sistema = √( (α · σ_WER)² + σ_chrF² )
```

Donde:
- **α** = |pendiente| de la regresión lineal chrF vs. WER (sensibilidad del LLM al error ASR)
- **σ_WER** = desviación estándar del WER sobre los 10 audios
- **σ_chrF** = variabilidad intrínseca del LLM (medida con la condición Referencia → LLM)

---

## Resultados Empíricos

### R1 — Cuantización

**Factor de escala `d` por formato:**

| Formato | Bloques | d mínimo | d máximo | d promedio | d mediana |
|---------|--------:|----------|----------|------------|-----------|
| Q8_0 | 2 206 096 | 1.48e-05 | 7.68e-03 | 4.93e-04 | 4.50e-04 |
| Q5_1 | 2 206 096 | 1.11e-04 | 3.35e-02 | 3.12e-03 | 2.90e-03 |
| Q4_0 | 2 206 096 | 2.36e-04 | 1.22e-01 | 7.83e-03 | 7.14e-03 |

A menor número de bits → mayor factor de escala → mayor error de cuantización.

### R2 — Precisión y Recursos

**WER promedio por modelo:**

| Modelo | Bits/peso | WER (%) |
|--------|----------:|--------:|
| Base | 32.00 | 10.0 |
| Tiny | 16.00 | 22.1 |
| Base-Q8_0 | 8.50 | 9.6 |
| Base-Q5_1 | 6.00 | 11.5 |
| Base-Q4_0 | 4.50 | 13.9 |

**Consumo de memoria RAM (iGPU Vulkan, mediana, n=30):**

| Modelo | ΔRAM mediana (MiB) | RSS promedio (MiB) |
|--------|-------------------:|-------------------:|
| Base | 404.69 | 126.17 |
| Tiny | 275.36 | 112.64 |
| Base-Q8_0 | 333.31 | 112.57 |
| Base-Q5_1 | 318.46 | 114.69 |
| Base-Q4_0 | 296.30 | 113.21 |

### R3 — Propagación ASR → LLM

**chrF por combinación (18 combinaciones):**

| LLM | Condición ASR | WER (%) | chrF |
|-----|---------------|--------:|-----:|
| Gemma3:1B FP16 | Referencia | 0.0 | 79.44 |
| Gemma3:1B FP16 | Base | 10.0 | 84.33 |
| Gemma3:1B FP16 | Base-Q8_0 | 9.6 | 83.40 |
| Gemma3:1B FP16 | Base-Q5_1 | 11.5 | 83.95 |
| Gemma3:1B FP16 | Base-Q4_0 | 13.9 | 81.05 |
| Gemma3:1B FP16 | Tiny | 22.1 | 75.13 |
| Gemma3:1B Q8_0 | Referencia | 0.0 | 78.57 |
| Gemma3:1B Q8_0 | Base | 10.0 | 83.86 |
| Gemma3:1B Q8_0 | Base-Q8_0 | 9.6 | 82.95 |
| Gemma3:1B Q8_0 | Base-Q5_1 | 11.5 | 84.21 |
| Gemma3:1B Q8_0 | Base-Q4_0 | 13.9 | 81.01 |
| Gemma3:1B Q8_0 | Tiny | 22.1 | 75.59 |
| Gemma3:1B Q4_K_M | Referencia | 0.0 | 78.89 |
| Gemma3:1B Q4_K_M | Base | 10.0 | 84.68 |
| Gemma3:1B Q4_K_M | Base-Q8_0 | 9.6 | 83.34 |
| Gemma3:1B Q4_K_M | Base-Q5_1 | 11.5 | 84.87 |
| Gemma3:1B Q4_K_M | Base-Q4_0 | 13.9 | 80.39 |
| Gemma3:1B Q4_K_M | Tiny | 22.1 | 74.15 |

**Propagación de Gauss — σ_sistema:**

| LLM | α | σ_chrF intrínseco | σ_sistema (Base) | σ_sistema (Q4_0) |
|-----|----:|------------------:|-----------------:|-----------------:|
| Gemma3:1B FP16 | 0.2085 | 3.949 | 4.060 | 4.069 |
| Gemma3:1B Q8_0 | 0.1457 | 4.382 | 4.431 | 4.435 |
| Gemma3:1B Q4_K_M | 0.2332 | 4.733 | 4.849 | 4.858 |

---

## Requisitos

- **Julia** ≥ 1.6
- **Python** ≥ 3.8 (con `matplotlib`, `numpy`)
- **whisper.cpp** compilado (ver sección de instalación)
- **Ollama** con los modelos Gemma 3 1B descargados
- **Linux** (las mediciones de RAM usan `/proc/meminfo`)
- **GNU time** (`/usr/bin/time -v`)

---

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/efuenzalidamunoz/estudio_casos_asr.git
cd estudio_casos_asr
```

### 2. Compilar whisper.cpp

```bash
git clone https://github.com/ggerganov/whisper.cpp.git
cd whisper.cpp

# Build CPU
cmake -B build
cmake --build build --config Release

# Build Vulkan (para iGPU Intel)
cmake -B build-vulkan -DGGML_VULKAN=ON
cmake --build build-vulkan --config Release

cd ..
```

### 3. Descargar modelos Whisper

```bash
cd whisper.cpp/models
./download-ggml-model.sh base
./download-ggml-model.sh tiny
cd ../..
```

### 4. Instalar modelos Ollama

```bash
ollama pull gemma3:1b-it-fp16
ollama pull gemma3:1b-it-q8_0
ollama pull gemma3:1b-it-q4_K_M
```

---

## Uso

### Cuantización de modelos (R1)

```bash
# Cuantizar el modelo base a Q8_0, Q5_1 y Q4_0
bash scripts/r1_a.sh

# Calcular bits efectivos por peso
bash scripts/calcular_bits_efectivos.sh

# Extraer el factor de escala d de los modelos cuantizados
python3 calcular_d.py
```

### Evaluación de precisión y recursos (R2)

```bash
# Pipeline completo ASR + traducción (5 modelos)
bash scripts/r2_b.sh

# Medición de consumo de memoria RAM (30 repeticiones por modelo)
bash scripts/medir_ram.sh

# Generar gráfico WER vs. formato numérico
python3 grafico.py
```

### Propagación del error (R3)

```bash
# Traducción con distintos LLM
bash scripts/r3_a.sh

# Comparación referencia vs. ASR cuantizado
bash scripts/r3_b.sh

# Matriz completa: 5 condiciones ASR × 3 LLM (18 combinaciones)
bash scripts/propagacion_error.sh

# Calcular propagación de Gauss
julia calcular_propagacion.jl
```

---

## Scripts Principales

| Script | Lenguaje | Descripción |
|--------|----------|-------------|
| `asr_pipeline.jl` | Julia | Transcripción con whisper-cli (CPU, 4 hilos, `-ng`) y cálculo de WER/CER/RTF |
| `asr_pipeline_igpu.jl` | Julia | Igual que el anterior pero usando la iGPU Intel vía Vulkan (`MESA_VK_DEVICE_SELECT`) |
| `translate_pipeline.jl` | Julia | Traduce transcripciones ASR con Ollama y calcula chrF contra la traducción de la referencia |
| `translate_reference.jl` | Julia | Traduce el groundtruth con Ollama y calcula chrF contra la traducción humana (mide error intrínseco del LLM) |
| `calcular_d.py` | Python | Lee los binarios GGML bloque a bloque y extrae el factor de escala `d` (FP16) |
| `calcular_propagacion.jl` | Julia | Aplica propagación de errores de Gauss sobre la matriz completa ASR → LLM |
| `grafico.py` | Python | Genera gráfico SVG del WER en función del formato numérico |

---

## Implementación Técnica

### Distancia de Levenshtein

Implementada desde cero en Julia con programación dinámica (dos filas), usada tanto a nivel de palabras (WER) como de caracteres (CER).

### chrF (Character F-score)

Implementación propia de la métrica de Popović (2015) con n-gramas de caracteres de orden 1 a 6 y β = 2 (mayor peso al recall).

### Comunicación con Ollama

Se usa la API REST `/api/generate` de Ollama directamente con `curl`, sin dependencias externas de JSON. La serialización/deserialización JSON se implementa manualmente en Julia.

### Lectura de modelos GGML

`calcular_d.py` parsea el formato binario GGML de whisper.cpp, recorriendo el header, vocabulario y cada tensor para extraer los factores de escala `d` de los bloques cuantizados (Q4_0, Q5_1, Q8_0).

### Medición de RAM

`medir_ram.sh` combina dos métodos:
- **ΔRAM del sistema:** Diferencia entre `MemAvailable` (baseline) y el mínimo observado durante la ejecución, con ventana de suavizado de 5 muestras.
- **RSS del proceso:** `Maximum resident set size` reportado por GNU time.

---

## Licencia

Proyecto académico — Universidad Católica del Maule, 2026.
