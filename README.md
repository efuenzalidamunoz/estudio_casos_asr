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
│  whisper-cli  (modelo cuantizado)                           │
│       │         ├── base (F32, 32 bits/peso)                │
│       │         ├── tiny (F16, 16 bits/peso)                │
│       │         ├── Q8_0 (8.5 bits/peso)                    │
│       │         ├── Q5_1 (6.0 bits/peso)                    │
│       │         └── Q4_0 (4.5 bits/peso)                    │
│       ▼                                                     │
│  out/*.txt  (transcripción en español)                      │
│       │                                                     │
│       ▼                                                     │
│  Ollama / Gemma 3  (LLM cuantizado)                        │
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
├── asr_pipeline_igpu.jl     # Pipeline principal: transcripción + métricas WER/CER/RTF (iGPU)
├── translate_pipeline.jl    # Traducción vía Ollama + evaluación chrF
├── translate_reference.jl   # Traduce solo la referencia para aislar error del LLM
├── calcular_d.py            # Extrae el factor de escala d de modelos cuantizados GGML
├── grafico.py               # Genera gráfico WER vs bits por peso (matplotlib)
├── medir_ram.sh             # Mide consumo de RAM (RSS) por modelo con GNU time
│
├── scripts/
│   ├── r1_a.sh              # R1a: cuantiza el modelo base a Q8_0, Q5_1, Q4_0
│   ├── calcular_bits_efectivos.sh  # R1b: calcula bits efectivos por peso
│   ├── r2_b.sh              # R2b: ejecuta ASR + traducción con cada modelo
│   ├── r3_a.sh              # R3a: traduce con distintas cuantizaciones del LLM
│   └── r3_b.sh              # R3b: tres condiciones experimentales de propagación
│
├── groundtruth.txt          # Transcripciones de referencia (español)
├── groundtruth_en.txt       # Traducciones de referencia (inglés)
├── audio/                   # 10 archivos WAV (test1.wav – test10.wav, ~15-30s c/u)
├── out/                     # Transcripciones generadas por whisper-cli
├── translations/            # Traducciones generadas (*_ref.txt y *_hyp.txt)
├── scripts_out/             # Resultados organizados por requerimiento
│   ├── R1/                  #   Logs de cuantización, bits efectivos, factor d
│   ├── R2/                  #   Precisión, RAM, gráficos
│   └── R3/                  #   Propagación del error en la cascada
│
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

> **Nota:** La opción `-DGGML_VULKAN=ON` habilita la aceleración por GPU vía Vulkan. El pipeline fuerza automáticamente el uso de la iGPU Intel mediante `MESA_VK_DEVICE_SELECT`.

### 3. Descargar el modelo base de Whisper

```bash
cd whisper.cpp
bash models/download-ggml-model.sh base
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
Genera `ggml-base-q8_0.bin`, `ggml-base-q5_1.bin` y `ggml-base-q4_0.bin` dentro de `whisper.cpp/models/`.

**R1b) Calcular bits efectivos por peso:**
```bash
bash scripts/calcular_bits_efectivos.sh
```

**R1b) Extraer el factor de escala *d* de cada modelo:**
```bash
python3 calcular_d.py
```
Analiza los bloques cuantizados de cada modelo GGML y calcula estadísticas (mín, máx, promedio, mediana) del factor de escala *d* que determina el paso de cuantización.

### R2 — Efecto de la cuantización sobre la precisión

**R2a) Transcribir audio con un modelo específico:**
```bash
julia asr_pipeline_igpu.jl whisper.cpp/models/ggml-base.bin
```
Transcribe los 10 archivos de audio, calcula WER, CER y RTF para cada uno y muestra promedios.

**R2b) Ejecutar pipeline completo (ASR + traducción) con todos los modelos:**
```bash
bash scripts/r2_b.sh
```

**R2b) Medir consumo de RAM:**
```bash
bash medir_ram.sh
```

**R2c) Generar gráfico de precisión vs bits por peso:**
```bash
python3 grafico.py
```

### R3 — Propagación del error en la cascada

**R3a) Traducir con distintas cuantizaciones del LLM:**
```bash
bash scripts/r3_a.sh
```

**R3b) Tres condiciones experimentales (error aislado vs compuesto):**
```bash
bash scripts/r3_b.sh
```
Ejecuta tres condiciones:
1. **Referencia → LLM**: traduce la transcripción perfecta (aísla el error del LLM)
2. **ASR alta precisión → LLM**: traduce la salida del modelo base F32
3. **ASR cuantizado → LLM**: traduce la salida del modelo Q4_0 (error compuesto)

**Traducción de referencia aislada (valida calidad del LLM):**
```bash
julia translate_reference.jl gemma3:1b-it-q4_K_M English
```

## 📊 Métricas Utilizadas

| Métrica | Qué mide | Fórmula / Descripción |
|---|---|---|
| **WER** | Tasa de error de palabra | Distancia de Levenshtein a nivel de palabras / total palabras de referencia |
| **CER** | Tasa de error de carácter | Distancia de Levenshtein a nivel de caracteres / total caracteres de referencia |
| **chrF** | Calidad de traducción | F-score de n-gramas de caracteres (Popović, 2015), β=2, n=1..6 |
| **RTF** | Real-Time Factor | Tiempo de procesamiento / duración del audio |
| **RAM** | Consumo de memoria | RSS máximo medido con GNU `time -v` |

## 📈 Resultados Resumidos

### Precisión de transcripción (WER promedio)

| Formato | Bits/peso | WER (%) | RAM (MiB) |
|---|---|---|---|
| base (F32) | 32.0 | 9.6 | 288.4 |
| tiny (F16) | 16.0 | 21.6 | 180.3 |
| Q8_0 | 8.5 | 9.6 | 222.5 |
| Q5_1 | 6.0 | 11.5 | 201.7 |
| Q4_0 | 4.5 | 14.4 | 189.2 |

> **Observación:** Q8_0 mantiene la misma precisión que el modelo base con un ahorro de ~23% en RAM. A partir de Q5_1 el error comienza a crecer de manera apreciable.

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

## 🛠️ Tecnologías

- **[whisper.cpp](https://github.com/ggml-org/whisper.cpp)** — Motor ASR en C/C++ sobre ggml
- **[Ollama](https://ollama.com)** — Servidor local de modelos de lenguaje
- **[Gemma 3](https://ai.google.dev/gemma)** — Modelo de lenguaje usado para traducción
- **Julia** — Pipeline de transcripción, traducción y métricas
- **Python** — Análisis de cuantización y visualización
- **Bash** — Orquestación de experimentos y medición de recursos

## 📄 Licencia

Proyecto académico desarrollado para el curso de Computación Numérica — UCM 2026-2.
