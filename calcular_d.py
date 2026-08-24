#!/usr/bin/env python3

import os
import struct
import math
import csv

ROOT = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(
    ROOT,
    "whisper.cpp",
    "models"
)

OUTPUT_DIR = os.path.join(
    ROOT,
    "scripts_out",
    "R1",
    "b"
)

OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    "quantization_steps.csv"
)

# ============================================================
# MODELOS
#
# Tipos GGML usados:
# 2 = Q4_0
# 7 = Q5_1
# 8 = Q8_0
# ============================================================

MODELS = [
    (
        "q8_0",
        os.path.join(MODEL_DIR, "ggml-base-q8_0.bin"),
        8
    ),
    (
        "q5_1",
        os.path.join(MODEL_DIR, "ggml-base-q5_1.bin"),
        7
    ),
    (
        "q4_0",
        os.path.join(MODEL_DIR, "ggml-base-q4_0.bin"),
        2
    ),
]


# INFORMACION DE LOS TIPOS GGML
# tipo: (nombre, elementos_por_bloque, bytes_por_bloque)

TYPE_INFO = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),

    # Q4_0:
    # fp16 d      = 2 bytes
    # 32 x 4 bits = 16 bytes
    # total       = 18 bytes
    2: ("Q4_0", 32, 18),

    # Q5_0
    6: ("Q5_0", 32, 22),

    # Q5_1:
    # fp16 d      = 2 bytes
    # fp16 m      = 2 bytes
    # qh          = 4 bytes
    # qs          = 16 bytes
    # total       = 24 bytes
    7: ("Q5_1", 32, 24),

    # Q8_0:
    # fp16 d      = 2 bytes
    # 32 x int8   = 32 bytes
    # total       = 34 bytes
    8: ("Q8_0", 32, 34),
}

def read_i32(f):
    data = f.read(4)

    if len(data) != 4:
        raise EOFError

    return struct.unpack("<i", data)[0]


def read_u32(f):
    data = f.read(4)

    if len(data) != 4:
        raise EOFError

    return struct.unpack("<I", data)[0]


def fp16_from_bytes(data):
    """
    Convierte 2 bytes IEEE-754 binary16
    a float de Python.
    """
    return struct.unpack("<e", data)[0]


# saltar header
def skip_header(f):
    # Magic number "ggml"
    magic = read_u32(f)

    if magic != 0x67676D6C:
        raise RuntimeError(
            f"Magic GGML inesperado: 0x{magic:08x}"
        )

    # Hyperparameters de Whisper
    hparams = [
        read_i32(f)
        for _ in range(11)
    ]

    # Filtros Mel
    n_mel = read_i32(f)
    n_fft = read_i32(f)

    filter_count = n_mel * n_fft

    f.seek(
        filter_count * 4,
        os.SEEK_CUR
    )

    # Vocabulario
    vocab_size = read_i32(f)

    for _ in range(vocab_size):

        length = read_u32(f)

        f.seek(
            length,
            os.SEEK_CUR
        )


# calcula el tamaño de un tensor
def tensor_nbytes(ne, ggml_type):

    if ggml_type not in TYPE_INFO:
        raise RuntimeError(
            f"Tipo GGML no soportado: {ggml_type}"
        )

    _, block_size, type_size = TYPE_INFO[ggml_type]

    n_elements = math.prod(ne)

    if n_elements % block_size != 0:
        raise RuntimeError(
            f"Número de elementos {n_elements} "
            f"no divisible por bloque {block_size}"
        )

    return (
        n_elements // block_size
    ) * type_size


# extraer d de cadad bloque
def extract_d_from_tensor(data, ggml_type):

    values = []

    # --------------------------------------------------------
    # Q4_0
    #
    # block_q4_0:
    #
    # fp16 d
    # uint8 qs[16]
    #
    # 18 bytes por bloque
    # --------------------------------------------------------

    if ggml_type == 2:

        block_bytes = 18

        for offset in range(
            0,
            len(data),
            block_bytes
        ):

            block = data[
                offset:offset + block_bytes
            ]

            if len(block) != block_bytes:
                break

            d = fp16_from_bytes(
                block[0:2]
            )

            if math.isfinite(d):
                values.append(abs(d))

    # --------------------------------------------------------
    # Q5_1
    #
    # block_q5_1:
    #
    # fp16 d
    # fp16 m
    # uint8 qh[4]
    # uint8 qs[16]
    #
    # 24 bytes por bloque
    # --------------------------------------------------------

    elif ggml_type == 7:

        block_bytes = 24

        for offset in range(
            0,
            len(data),
            block_bytes
        ):

            block = data[
                offset:offset + block_bytes
            ]

            if len(block) != block_bytes:
                break

            d = fp16_from_bytes(
                block[0:2]
            )

            if math.isfinite(d):
                values.append(abs(d))

    # --------------------------------------------------------
    # Q8_0
    #
    # block_q8_0:
    #
    # fp16 d
    # int8 qs[32]
    #
    # 34 bytes por bloque
    # --------------------------------------------------------

    elif ggml_type == 8:

        block_bytes = 34

        for offset in range(
            0,
            len(data),
            block_bytes
        ):

            block = data[
                offset:offset + block_bytes
            ]

            if len(block) != block_bytes:
                break

            d = fp16_from_bytes(
                block[0:2]
            )

            if math.isfinite(d):
                values.append(abs(d))

    return values


# ============================================================
# CALCULAR MEDIANA
# ============================================================

def median(values):

    sorted_values = sorted(values)

    n = len(sorted_values)

    middle = n // 2

    if n % 2 == 0:
        return (
            sorted_values[middle - 1]
            + sorted_values[middle]
        ) / 2

    return sorted_values[middle]


# ============================================================
# ANALIZAR UN MODELO COMPLETO
# ============================================================

def analyze_model(path, wanted_type):

    all_d = []

    tensor_count = 0
    quant_tensor_count = 0

    with open(path, "rb") as f:

        skip_header(f)

        while True:

            # ------------------------------------------------
            # Header del tensor
            # ------------------------------------------------

            header = f.read(12)

            if len(header) == 0:
                break

            if len(header) != 12:
                raise RuntimeError(
                    "Archivo terminó en medio "
                    "del encabezado de un tensor"
                )

            n_dims, name_length, ggml_type = struct.unpack(
                "<iii",
                header
            )

            if n_dims <= 0 or n_dims > 4:
                raise RuntimeError(
                    f"n_dims inválido: {n_dims}"
                )

            # ------------------------------------------------
            # Dimensiones
            # ------------------------------------------------

            ne = []

            for _ in range(n_dims):
                ne.append(
                    read_i32(f)
                )

            # ------------------------------------------------
            # Nombre del tensor
            # ------------------------------------------------

            name = f.read(name_length).decode(
                "utf-8",
                errors="replace"
            )

            # ------------------------------------------------
            # Tamaño del tensor
            # ------------------------------------------------

            size = tensor_nbytes(
                ne,
                ggml_type
            )

            tensor_count += 1

            # ------------------------------------------------
            # Leer datos
            # ------------------------------------------------

            data = f.read(size)

            if len(data) != size:
                raise RuntimeError(
                    f"Tensor incompleto: {name}"
                )

            # ------------------------------------------------
            # Solo extraer d del tipo que nos interesa
            # ------------------------------------------------

            if ggml_type == wanted_type:

                quant_tensor_count += 1

                ds = extract_d_from_tensor(
                    data,
                    ggml_type
                )

                all_d.extend(ds)

    # ========================================================
    # VALIDACIÓN
    # ========================================================

    if not all_d:
        raise RuntimeError(
            f"No se encontraron bloques "
            f"del tipo GGML {wanted_type}"
        )

    # ========================================================
    # ESTADÍSTICAS
    # ========================================================

    d_min = min(all_d)

    d_max = max(all_d)

    d_mean = (
        sum(all_d)
        / len(all_d)
    )

    d_median = median(all_d)

    return {
        "blocks": len(all_d),
        "tensors": quant_tensor_count,

        "d_min": d_min,
        "d_max": d_max,
        "d_mean": d_mean,
        "d_median": d_median,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    results = []

    # ========================================================
    # ANALIZAR MODELOS
    # ========================================================

    for name, path, wanted_type in MODELS:

        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"No se encontró el modelo:\n{path}"
            )

        print(
            f"Analizando {name}..."
        )

        result = analyze_model(
            path,
            wanted_type
        )

        results.append({
            "formato": name,
            **result
        })

    # ========================================================
    # GUARDAR CSV
    # ========================================================

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as csvfile:

        writer = csv.writer(
            csvfile
        )

        writer.writerow([
            "formato",
            "bloques",
            "tensores_cuantizados",
            "d_min",
            "d_max",
            "d_promedio",
            "d_mediana"
        ])

        for r in results:

            writer.writerow([
                r["formato"],
                r["blocks"],
                r["tensors"],

                f'{r["d_min"]:.10e}',
                f'{r["d_max"]:.10e}',
                f'{r["d_mean"]:.10e}',
                f'{r["d_median"]:.10e}',
            ])
    print()
    print(
        f"Resultados guardados en:"
    )
    print(
        OUTPUT_CSV
    )


if __name__ == "__main__":
    main()