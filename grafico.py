import csv
import numpy as np
import matplotlib.pyplot as plt

INPUT_CSV = "scripts_out/R2/c/datos_precision.csv"
OUTPUT_SVG = "scripts_out/R2/c/wer_porcentual_ordenado.svg"

# ------------------------------------------------------------
# Leer datos
# ------------------------------------------------------------

datos = []

with open(INPUT_CSV, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:
        wer = float(row["wer_promedio"])

        # El CSV puede guardar WER como fracción:
        # 0.096 -> 9.6 %
        if wer <= 1.0:
            wer *= 100

        datos.append({
            "formato": row["formato"],
            "bits": float(row["bits_efectivos"]),
            "wer": wer
        })


# ------------------------------------------------------------
# Posiciones visuales
#
# Base -> Tiny -> Q8_0 -> Q5_1 -> Q4_0
#
# Tiny se muestra entre Base y Q8_0, pero NO forma parte
# de la línea de cuantización del modelo Base.
# ------------------------------------------------------------

posiciones = {
    "base": 0,
    "tiny": 1,
    "q8_0": 2,
    "q5_1": 3,
    "q4_0": 4
}

orden_visual = [
    "base",
    "tiny",
    "q8_0",
    "q5_1",
    "q4_0"
]

orden_curva = [
    "base",
    "q8_0",
    "q5_1",
    "q4_0"
]


# ------------------------------------------------------------
# Obtener datos de la curva Base -> cuantizaciones
# ------------------------------------------------------------

curva = []

for nombre in orden_curva:
    encontrado = next(
        (
            d for d in datos
            if d["formato"].lower() == nombre
        ),
        None
    )

    if encontrado is not None:
        curva.append(encontrado)


# Tiny como modelo independiente
tiny = next(
    (
        d for d in datos
        if d["formato"].lower() == "tiny"
    ),
    None
)


# ------------------------------------------------------------
# Preparar datos
# ------------------------------------------------------------

x_curva = np.array([
    posiciones[d["formato"].lower()]
    for d in curva
])

wer_curva = np.array([
    d["wer"]
    for d in curva
])


# ------------------------------------------------------------
# Estilo
# ------------------------------------------------------------

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9
})


fig, ax = plt.subplots(
    figsize=(6.6, 3.55)
)


# ------------------------------------------------------------
# Curva principal
#
# Base -> Q8_0 -> Q5_1 -> Q4_0
# ------------------------------------------------------------

ax.plot(
    x_curva,
    wer_curva,
    marker="o",
    linewidth=1.8,
    markersize=6
)


# ------------------------------------------------------------
# Tiny independiente
# ------------------------------------------------------------

if tiny is not None:

    ax.plot(
        posiciones["tiny"],
        tiny["wer"],
        marker="o",
        markersize=6,
        linestyle="none"
    )


# ------------------------------------------------------------
# Títulos y ejes
# ------------------------------------------------------------

ax.set_title(
    "Error de transcripción según representación numérica"
)

ax.set_ylabel(
    "WER promedio (%)"
)

ax.set_xlabel(
    "Formato y bits efectivos por peso"
)


# ------------------------------------------------------------
# Etiquetas del eje X
# ------------------------------------------------------------

xticks = []
xticklabels = []

for nombre in orden_visual:

    dato = next(
        (
            d for d in datos
            if d["formato"].lower() == nombre
        ),
        None
    )

    if dato is None:
        continue

    xticks.append(
        posiciones[nombre]
    )

    xticklabels.append(
        f'{dato["formato"]}\n{dato["bits"]:g} bits/peso'
    )


ax.set_xticks(
    xticks
)

ax.set_xticklabels(
    xticklabels
)


# ------------------------------------------------------------
# Escala del eje Y
# ------------------------------------------------------------

todos_wer = [
    d["wer"]
    for d in datos
]

max_wer = max(todos_wer)

ax.set_ylim(
    0,
    max_wer * 1.22
)


# ------------------------------------------------------------
# Rejilla
# ------------------------------------------------------------

ax.grid(
    axis="y",
    alpha=0.25
)


# ------------------------------------------------------------
# Etiquetas sobre los puntos de la curva
# ------------------------------------------------------------

for d in curva:

    xi = posiciones[
        d["formato"].lower()
    ]

    ax.annotate(
        f'{d["wer"]:.1f}%',
        xy=(
            xi,
            d["wer"]
        ),
        xytext=(
            0,
            7
        ),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=8.5
    )


# ------------------------------------------------------------
# Etiqueta de Tiny
# ------------------------------------------------------------

if tiny is not None:

    ax.annotate(
        f'{tiny["wer"]:.1f}%',
        xy=(
            posiciones["tiny"],
            tiny["wer"]
        ),
        xytext=(
            0,
            7
        ),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=8.5
    )


# ------------------------------------------------------------
# Apariencia final
# ------------------------------------------------------------

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.tick_params(
    axis="x",
    length=0
)

fig.tight_layout(
    pad=1.2
)


# ------------------------------------------------------------
# Guardar
# ------------------------------------------------------------

fig.savefig(
    OUTPUT_SVG,
    bbox_inches="tight"
)

plt.show()