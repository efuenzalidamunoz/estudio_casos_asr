# 🎙️ Estudio de Casos: Cuantización y Propagación del Error en una Cascada ASR → LLM

> **Computación Numérica — Universidad Católica del Maule, 2026-2**

## 📋 Descripción del Problema

Un centro de atención de urgencia de la Región del Maule desea implementar un **sistema de traducción en tiempo real** para atender a pacientes que no hablan español. Por razones de **confidencialidad clínica**, el sistema debe ejecutarse completamente en el dispositivo local (sin GPU dedicada) con latencia inferior al tiempo real (*real-time factor* < 1).

La arquitectura propuesta es una **cascada de dos etapas**:

```
Audio (español) ──► whisper.cpp (ASR) ──► Texto español ──► Ollama (LLM) ──► Traducción
```

1. **Reconocimiento automático del habla (ASR)** con [whisper.cpp](https://github.com/ggml-org/whisper.cpp), implementación en C/C++ del modelo Whisper sobre la biblioteca ggml.
2. **Traducción del texto** con un modelo de lenguaje local servido mediante [Ollama](https://ollama.com).

El equipo disponible carece de memoria suficiente para ejecutar ambos modelos en precisión completa, por lo que se recurre a la **cuantización**: los pesos (originalmente en IEEE-754 binary32) se convierten a formatos de menor ancho de palabra (F16, Q8_0, Q5_1, Q4_0). En los esquemas ggml, los pesos se agrupan en bloques de 32 valores que comparten un factor de escala *d* en FP16:

$$\tilde{w}_i = d \cdot q_i + m, \quad q_i \in \mathbb{Z} \cap [q_{\min},\, q_{\max}]$$

El objetivo del estudio es responder: **¿cuánta precisión numérica se puede sacrificar antes de que la traducción deje de ser clínicamente confiable?**

### 🖥️ Nota sobre el uso de la GPU integrada (iGPU)

El equipo de pruebas cuenta con dos dispositivos gráficos:

| Dispositivo | Tipo | VRAM |
|---|---|---|
| **NVIDIA GeForce GTX 1650** | GPU dedicada | 4 GB |
| **Intel UHD Graphics (CML GT2)** | GPU integrada (iGPU) | Memoria compartida con RAM |

La GTX 1650 no dispone de VRAM suficiente para cargar los modelos de Whisper y Gemma 3 simultáneamente, por lo que se **forzó el uso exclusivo de la GPU integrada Intel** en ambos componentes del pipeline:

- **whisper.cpp**: se compiló con soporte **Vulkan** y se utiliza la variable de entorno `MESA_VK_DEVICE_SELECT=8086:9bc4!` para que whisper-cli solo vea la iGPU Intel. El sufijo `!` oculta los demás dispositivos Vulkan (GTX 1650 y llvmpipe), garantizando que toda la inferencia ASR se ejecute en la iGPU.

- **Ollama**: se configura para usar la iGPU Intel mediante las variables de entorno:
  ```bash
  export OLLAMA_INTEL_GPU=true
  export HSA_OVERRIDE_GFX_VERSION=0  # Evita intentar usar la GPU dedicada
  ```
  Esto asegura que Gemma 3 se ejecute sobre la GPU integrada, aprovechando la memoria RAM compartida en lugar de la VRAM limitada de la GTX 1650.

> **¿Por qué no usar la GTX 1650?** Con solo 4 GB de VRAM, cargar un modelo Whisper (~300 MB) y un LLM Gemma 3 fp16 (~2.4 GB) excedería la capacidad disponible. La iGPU Intel, al compartir la RAM del sistema (~16 GB), permite cargar modelos más grandes a costa de un menor ancho de banda de memoria.

## 🏗️ Arquitectura del Experimento

```
┌─────────────────────────────────────────────────────────────┐
│                    PIPELINE COMPLETO                        │
│                                                             │
│  audio/*.wav                                                │
│       │                                                     │
│       ▼                                                     │
│  whisper-cli  (modelo cuantizado, iGPU Intel via Vulkan)    │
│       │         ├── base (F32, 32 bits/peso)                │
│       │         ├── tiny (F16, 16 bits/peso)                │
│       │         ├── Q8_0 (8.5 bits/peso)                    │
│       │         ├── Q5_1 (6.0 bits/peso)                    │
│       │         └── Q4_0 (4.5 bits/peso)                    │
│       ▼                                                     │
│  out/*.txt  (transcripción en español)                      │
│       │                                                     │
│       ▼                                                     │
│  Ollama / Gemma 3 1B  (LLM cuantizado, iGPU Intel)         │
│       │         ├── fp16                                    │
│       │         ├── q8_0                                    │
│       │         └── q4_K_M                                  │
│       ▼                                                     │
│  translations/*_hyp.txt  (traducción al inglés)             │
│                                                             │
│  Métricas: WER · CER · chrF · RTF · RAM                    │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Estructura del Proyecto

```
estudio_casos_asr/
├── asr_pipeline_igpu.jl     # Pipeline ASR con iGPU Intel (Vulkan): transcripción + WER/CER/RTF
├── asr_pipeline.jl          # Pipeline ASR solo CPU (sin GPU, flag -ng): transcripción + WER/CER/RTF
├── translate_pipeline.jl    # Traducción vía Ollama + evaluación chrF (ref_LLM vs hyp_LLM)
├── translate_reference.jl   # Traduce solo la referencia y compara contra groundtruth_en.txt (chrF)
├── calcular_d.py            # Extrae el factor de escala d de modelos cuantizados GGML → CSV
├── grafico.py               # Genera gráfico SVG de WER vs bits por peso (matplotlib)
│
├── scripts/
│   ├── r1_a.sh                    # R1a: cuantiza el modelo base a Q8_0, Q5_1, Q4_0
│   ├── calcular_bits_efectivos.sh # R1b: calcula bits efectivos por peso → CSV
│   ├── r2_b.sh                    # R2b: ejecuta ASR + traducción con cada modelo
│   ├── medir_ram.sh               # R2b: mide ΔRAM del sistema y RSS con monitor + GNU time
│   ├── r3_a.sh                    # R3a: traduce con distintas cuantizaciones del LLM
│   └── r3_b.sh                    # R3b: tres condiciones experimentales de propagación
│
├── groundtruth.txt          # Transcripciones de referencia (español, 10 segmentos)
├── groundtruth_en.txt       # Traducciones de referencia humana (inglés, 10 segmentos)
├── audio/                   # 10 archivos WAV (test1.wav – test10.wav, ~15-30s c/u)
├── out/                     # Transcripciones generadas por whisper-cli
├── translations/            # Traducciones generadas (*_ref.txt y *_hyp.txt)
│
├── scripts_out/             # Resultados organizados por requerimiento
│   ├── R1/
│   │   ├── a/               #   Logs de cuantización (q8_0, q5_1, q4_0)
│   │   └── b/               #   bits_efectivos.csv, quantization_steps.csv (factor d)
│   ├── R2/
│   │   ├── b/               #   Salidas completas ASR+traducción por modelo
│   │   ├── c/               #   datos_precision.csv, wer_porcentual_ordenado.svg
│   │   └── ram/             #   memory_all_models.csv, ram_promedios.csv
│   └── R3/
│       ├── a/               #   chrF por cuantización del LLM (fp16, q8_0, q4_K_M)
│       └── b/               #   Tres condiciones: referencia→LLM, ASR_alta→LLM, ASR_Q4→LLM
│
├── estudio_casos_unidad1-2.pdf  # Enunciado del caso de estudio
├── whisper.cpp/             # Subproyecto whisper.cpp (compilar localmente)
└── .gitignore
```

## ⚙️ Requisitos Previos

| Herramienta | Versión sugerida | Propósito |
|---|---|---|
| **Julia** | ≥ 1.10 | Pipeline ASR y traducción |
| **Python 3** | ≥ 3.10 | Análisis de cuantización y gráficos |
| **CMake** | ≥ 3.14 | Compilación de whisper.cpp |
| **GCC / Clang** | — | Compilación de whisper.cpp |
| **Ollama** | latest | Servir modelos de lenguaje localmente |
| **GNU time** | `/usr/bin/time` | Medición de RAM (RSS) |
| **matplotlib** + **numpy** | — | Generación de gráficos |
| **Drivers Vulkan Intel** | — | Aceleración por iGPU (mesa-vulkan-drivers) |

## 🚀 Instalación y Compilación

### 1. Clonar el repositorio

```bash
git clone https://github.com/<usuario>/estudio_casos_asr.git
cd estudio_casos_asr
```

### 2. Compilar whisper.cpp (con soporte Vulkan para iGPU)

```bash
git clone https://github.com/ggml-org/whisper.cpp.git
cd whisper.cpp
cmake -B build-vulkan -DGGML_VULKAN=ON
cmake --build build-vulkan --config Release
cd ..
```

> **Nota:** La opción `-DGGML_VULKAN=ON` habilita la aceleración por GPU vía Vulkan. El pipeline `asr_pipeline_igpu.jl` fuerza automáticamente el uso de la iGPU Intel mediante `MESA_VK_DEVICE_SELECT=8086:9bc4!`.

Si se desea compilar también la versión solo CPU (usada por `asr_pipeline.jl`):

```bash
cd whisper.cpp
cmake -B build
cmake --build build --config Release
cd ..
```

### 3. Descargar el modelo base de Whisper

```bash
cd whisper.cpp
bash models/download-ggml-model.sh base
bash models/download-ggml-model.sh tiny
cd ..
```

### 4. Instalar Ollama y descargar modelos

```bash
# Instalar Ollama (https://ollama.com)
curl -fsSL https://ollama.com/install.sh | sh

# Descargar modelos de lenguaje necesarios
ollama pull gemma3:1b-it-q4_K_M
ollama pull gemma3:1b-it-q8_0
ollama pull gemma3:1b-it-fp16
```

**Forzar uso de la iGPU Intel en Ollama** (agregar al `.bashrc` o ejecutar antes de `ollama serve`):

```bash
export OLLAMA_INTEL_GPU=true
export HSA_OVERRIDE_GFX_VERSION=0
```

### 5. Dependencias de Python

```bash
pip install matplotlib numpy
```

## 🔬 Ejecución de los Experimentos

### R1 — Representación numérica y cuantización

**R1a) Generar modelos cuantizados:**
```bash
bash scripts/r1_a.sh
```
Genera `ggml-base-q8_0.bin`, `ggml-base-q5_1.bin` y `ggml-base-q4_0.bin` dentro de `whisper.cpp/models/`. Los logs se guardan en `scripts_out/R1/a/`.

**R1b) Calcular bits efectivos por peso:**
```bash
bash scripts/calcular_bits_efectivos.sh
```
Genera `scripts_out/R1/b/bits_efectivos.csv` con el desglose: pesos por bloque, bits de datos, bits de metadata y bits efectivos por peso.

**R1b) Extraer el factor de escala *d* de cada modelo:**
```bash
python3 calcular_d.py
```
Analiza los bloques cuantizados de cada modelo GGML y calcula estadísticas (mín, máx, promedio, mediana) del factor de escala *d* que determina el paso de cuantización. Los resultados se guardan en `scripts_out/R1/b/quantization_steps.csv`.

### R2 — Efecto de la cuantización sobre la precisión

**R2a) Transcribir audio con un modelo específico (iGPU):**
```bash
julia asr_pipeline_igpu.jl whisper.cpp/models/ggml-base.bin
```
Transcribe los 10 archivos de audio usando la iGPU Intel vía Vulkan, calcula WER, CER y RTF para cada uno y muestra promedios.

**R2a) Transcribir audio con un modelo específico (solo CPU):**
```bash
julia asr_pipeline.jl whisper.cpp/models/ggml-base.bin
```
Versión que ejecuta `whisper-cli` con el flag `-ng` (no GPU), usando el build CPU en `whisper.cpp/build/`.

**R2b) Ejecutar pipeline completo (ASR + traducción) con todos los modelos:**
```bash
bash scripts/r2_b.sh
```
Ejecuta `asr_pipeline_igpu.jl` y luego `translate_pipeline.jl` con `gemma3:1b-it-q4_K_M` para cada uno de los 5 modelos Whisper. Los resultados se guardan en `scripts_out/R2/b/`.

**R2b) Medir consumo de RAM:**
```bash
bash scripts/medir_ram.sh
```
Mide el consumo de RAM de whisper-cli con la iGPU Intel. Para cada modelo ejecuta 3 repeticiones × 10 audios (30 mediciones). Utiliza un monitor de `MemAvailable` con ventana de suavizado y también captura el RSS máximo vía `GNU time -v`. Incluye una fase de calentamiento (la primera ejecución no se mide) y cooldown entre mediciones. Los resultados se guardan en:
- `scripts_out/R2/ram/memory_all_models.csv` (mediciones individuales)
- `scripts_out/R2/ram/ram_promedios.csv` (media, mediana, desv. estándar por modelo)

**R2c) Generar gráfico de precisión vs bits por peso:**
```bash
python3 grafico.py
```
Lee `scripts_out/R2/c/datos_precision.csv` y genera `scripts_out/R2/c/wer_porcentual_ordenado.svg`.

### R3 — Propagación del error en la cascada

**R3a) Traducir con distintas cuantizaciones del LLM:**
```bash
bash scripts/r3_a.sh
```
Ejecuta `translate_pipeline.jl` con tres variantes de Gemma 3 (`fp16`, `q8_0`, `q4_K_M`), manteniendo fijo el modelo ASR (usa las transcripciones ya generadas en `out/`). Los resultados se guardan en `scripts_out/R3/a/`.

**R3b) Tres condiciones experimentales (error aislado vs compuesto):**
```bash
bash scripts/r3_b.sh
```
Ejecuta tres condiciones con el LLM fijo (`gemma3:1b-it-q4_K_M`):
1. **Referencia → LLM**: traduce la transcripción perfecta (`translate_reference.jl`), aísla el error del LLM
2. **ASR alta precisión → LLM**: transcribe con el modelo base F32 y traduce la salida
3. **ASR cuantizado → LLM**: transcribe con el modelo Q4_0 y traduce (error compuesto)

Los resultados se guardan en `scripts_out/R3/b/`.

**Traducción de referencia aislada (valida calidad del LLM):**
```bash
julia translate_reference.jl gemma3:1b-it-q4_K_M English
```
Traduce solo el texto de referencia en español y compara contra la traducción humana de referencia (`groundtruth_en.txt`) usando chrF.

## 📊 Métricas Utilizadas

| Métrica | Qué mide | Fórmula / Descripción |
|---|---|---|
| **WER** | Tasa de error de palabra | Distancia de Levenshtein a nivel de palabras / total palabras de referencia |
| **CER** | Tasa de error de carácter | Distancia de Levenshtein a nivel de caracteres / total caracteres de referencia |
| **chrF** | Calidad de traducción | F-score de n-gramas de caracteres (Popović, 2015), β=2, n=1..6 |
| **RTF** | Real-Time Factor | Tiempo de procesamiento / duración del audio |
| **ΔRAM** | Consumo de memoria del sistema | Caída en `MemAvailable` (mediana, suavizada) durante la inferencia |
| **RSS** | Resident Set Size | RSS máximo del proceso `whisper-cli`, medido con `GNU time -v` |

## 📈 Resultados Resumidos

### Precisión de transcripción (WER promedio)

| Formato | Bits/peso | WER (%) | CER (%) | RTF |
|---|---|---|---|---|
| base (F32) | 32.0 | 10.0 | 2.3 | 0.19 |
| tiny (F16) | 16.0 | 22.1 | 5.9 | 0.12 |
| Q8_0 | 8.5 | 9.6 | 2.2 | 0.22 |
| Q5_1 | 6.0 | 11.5 | 2.8 | 0.47 |
| Q4_0 | 4.5 | 13.9 | 3.6 | 0.34 |

> **Observación:** Q8_0 mantiene incluso mejor precisión que el modelo base, con un WER de 9.6% vs 10.0%. A partir de Q5_1 el error comienza a crecer de manera apreciable. Todos los modelos logran RTF < 1 (tiempo real).

### Consumo de memoria (ΔRAM del sistema, iGPU Intel)

| Formato | ΔRAM mediana (MiB) | RSS promedio (MiB) | n |
|---|---|---|---|
| base (F32) | 404.69 | 126.17 | 30 |
| tiny (F16) | 275.36 | 112.64 | 30 |
| Q8_0 | 333.31 | 112.57 | 30 |
| Q5_1 | 318.46 | 114.69 | 30 |
| Q4_0 | 296.30 | 113.21 | 30 |

> **Nota:** ΔRAM refleja el consumo total del sistema (incluye buffers de la iGPU asignados desde la RAM compartida). RSS mide solo la memoria del proceso en espacio de usuario.

### Factor de escala *d* (paso de cuantización)

| Formato | Bloques | d mín | d máx | d promedio | d mediana |
|---|---|---|---|---|---|
| Q8_0 | 2 206 096 | 1.48×10⁻⁵ | 7.68×10⁻³ | 4.93×10⁻⁴ | 4.50×10⁻⁴ |
| Q5_1 | 2 206 096 | 1.11×10⁻⁴ | 3.35×10⁻² | 3.12×10⁻³ | 2.90×10⁻³ |
| Q4_0 | 2 206 096 | 2.36×10⁻⁴ | 1.22×10⁻¹ | 7.83×10⁻³ | 7.14×10⁻³ |

> **Observación:** El paso de cuantización *d* crece con la agresividad de la cuantización. En Q4_0, la mediana de *d* es ~16× mayor que en Q8_0, lo que explica la mayor pérdida de precisión.

### Calidad de traducción (chrF, LLM fijo: q4_K_M)

| Formato ASR | WER ASR (%) | chrF traducción |
|---|---|---|
| base (F32) | 10.0 | 84.68 |
| tiny (F16) | 22.1 | 74.15 |
| Q8_0 | 9.6 | 83.34 |
| Q5_1 | 11.5 | 84.87 |
| Q4_0 | 13.9 | 80.39 |

### Efecto de la cuantización del LLM (R3a, ASR fijo)

| Modelo LLM | chrF promedio |
|---|---|
| gemma3:1b-it-fp16 | 81.05 |
| gemma3:1b-it-q8_0 | 81.01 |
| gemma3:1b-it-q4_K_M | 80.39 |

> **Observación:** La cuantización del LLM tiene un efecto menor que la del ASR. La diferencia entre fp16 y q4_K_M es de solo ~0.7 puntos chrF.

### Propagación del error en la cascada (R3b)

| Condición | Descripción | chrF |
|---|---|---|
| Referencia → LLM | Texto perfecto traducido por LLM (aísla error del LLM) | 78.89 |
| ASR alta (F32) → LLM | Transcripción F32 + traducción | 84.68 |
| ASR cuantizado (Q4_0) → LLM | Transcripción Q4_0 + traducción (error compuesto) | 80.39 |

> **Observación:** La condición "Referencia → LLM" usa chrF contra `groundtruth_en.txt` (traducción humana), mientras que las condiciones 2 y 3 usan chrF entre la traducción de la referencia y la traducción de la hipótesis generadas por el mismo LLM. Los errores del ASR se amplifican al pasar por el LLM: la caída de chrF entre F32 y Q4_0 es de ~4.3 puntos.

## 📝 Corpus de Audio

El corpus está compuesto por **10 segmentos de audio** (15–30 segundos cada uno) con temática clínica de urgencias, que simulan interacciones reales entre personal médico y pacientes:

- Toma de signos vitales
- Anamnesis y síntomas
- Localización del dolor
- Dificultad respiratoria
- Historial de fiebre y síntomas respiratorios
- Antecedentes médicos
- Medicación actual
- Alergias
- Ingesta reciente
- Instrucciones de espera

Cada segmento tiene su transcripción de referencia en español (`groundtruth.txt`) y traducción de referencia al inglés (`groundtruth_en.txt`).

## 🔧 Descripción de los Scripts Principales

### Julia

| Script | Descripción |
|---|---|
| `asr_pipeline_igpu.jl` | Transcribe 10 audios con whisper-cli usando la **iGPU Intel** (Vulkan, `MESA_VK_DEVICE_SELECT=8086:9bc4!`). Calcula WER, CER y RTF. Lee encabezados WAV para obtener la duración sin dependencias externas. Usa `build-vulkan/`. |
| `asr_pipeline.jl` | Versión **solo CPU** del pipeline. Usa el flag `-ng` (no GPU) y el build en `build/`. Misma lógica de métricas. |
| `translate_pipeline.jl` | Traduce cada transcripción de `out/` y la referencia correspondiente de `groundtruth.txt` usando Ollama. Calcula chrF entre ambas traducciones (aísla el error del ASR). Usa temperatura 0 y semilla 67 para reproducibilidad. |
| `translate_reference.jl` | Traduce solo las referencias en español (`groundtruth.txt`) y compara contra las traducciones humanas (`groundtruth_en.txt`) usando chrF. Mide el error puro del LLM. |

### Python

| Script | Descripción |
|---|---|
| `calcular_d.py` | Parsea archivos GGML (header + tensores), extrae el valor `d` (FP16) de cada bloque cuantizado (Q8_0, Q5_1, Q4_0) y calcula estadísticas. Guarda en `scripts_out/R1/b/quantization_steps.csv`. |
| `grafico.py` | Lee `scripts_out/R2/c/datos_precision.csv` y genera un gráfico SVG con WER (%) vs formato/bits por peso, ordenado por WER ascendente. |

### Bash

| Script | Descripción |
|---|---|
| `scripts/r1_a.sh` | Ejecuta `whisper-quantize` sobre el modelo base para generar Q8_0, Q5_1 y Q4_0. |
| `scripts/calcular_bits_efectivos.sh` | Calcula bits efectivos por peso para cada formato (incluyendo metadata: factor *d* y offset *m*). |
| `scripts/r2_b.sh` | Ejecuta `asr_pipeline_igpu.jl` + `translate_pipeline.jl` con los 5 modelos Whisper (base, tiny, Q8_0, Q5_1, Q4_0) y LLM `gemma3:1b-it-q4_K_M`. |
| `scripts/medir_ram.sh` | Mide ΔRAM y RSS para cada modelo con 3 repeticiones × 10 audios. Incluye calentamiento, baseline estable (mediana), monitor con ventana de suavizado, y resumen estadístico. Ejecuta con iGPU via Vulkan. |
| `scripts/r3_a.sh` | Ejecuta `translate_pipeline.jl` con los 3 modelos LLM (fp16, q8_0, q4_K_M), ASR fijo. |
| `scripts/r3_b.sh` | Ejecuta las 3 condiciones experimentales: referencia→LLM, ASR_F32→LLM, ASR_Q4_0→LLM. |

## 🛠️ Tecnologías

- **[whisper.cpp](https://github.com/ggml-org/whisper.cpp)** — Motor ASR en C/C++ sobre ggml (backend Vulkan para iGPU)
- **[Ollama](https://ollama.com)** — Servidor local de modelos de lenguaje
- **[Gemma 3 1B](https://ai.google.dev/gemma)** — Modelo de lenguaje usado para traducción (variantes fp16, q8_0, q4_K_M)
- **Julia** — Pipeline de transcripción, traducción y métricas (WER, CER, chrF, RTF)
- **Python** — Análisis de cuantización (factor *d*) y visualización (matplotlib)
- **Bash** — Orquestación de experimentos, cuantización y medición de recursos (RAM)

## 📄 Licencia

Proyecto académico desarrollado para el curso de Computación Numérica — UCM 2026-2.
